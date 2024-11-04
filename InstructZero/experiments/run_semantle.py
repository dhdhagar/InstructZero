import random
import torch
import numpy as np
import copy
from automatic_prompt_engineer import ape, data
from data.instruction_induction.load_data import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from automatic_prompt_engineer import evaluate, config, template, data
import os
import re
import json
from misc import get_test_conf, get_conf, plot_posterior, get_gp, get_coupled_kernel_data, extract_json

from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cosine_similarity
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from sentence_transformers import SentenceTransformer
from instruction_coupled_kernel import *
import time
import warnings

from misc import set_all_seed, TASKS, tkwargs

from args import parse_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LMForwardAPI:
    def __init__(self, model_name=None, bbox_model_name=None, target=None, warmstart=None, random_proj=None,
                 intrinsic_dim=None, n_prompt_tokens=None, args=None, device='cuda', dtype=torch.float16):
        self.target = target
        self.intrinsic_dim = intrinsic_dim
        self.args = args

        # Load generator model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            torch_dtype=dtype,
            token=args.hf_access_token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.decoding_kwargs = {
            "num_return_sequences": self.args.num_return_sequences,
            "top_p": 0.9,
            "temperature": self.args.temperature,
            "do_sample": self.args.do_sample,
            "repetition_penalty": self.args.repetition_penalty,
        }
        assert self.args.num_return_sequences == 1 or self.args.do_sample, "num_return_sequences must be 1 if not sampling"

        # Load bbox model
        self.bbox_model = SentenceTransformer(bbox_model_name,
                                              trust_remote_code=True,
                                              token=args.hf_access_token,
                                              model_kwargs={"torch_dtype": dtype},
                                              device=args.device)
        self.bbox_prompt = "What is a %s?"
        self.bbox_instruction = f"""Instruct: Given the following English-language word, retrieve only those words \
that are similar to it in meaning.\n\nWord: """

        # Target word
        self.target_embed = self.get_scores(guesses=[self.target], target=None)

        # Warmstart text that defines the task
        self.warmstart = warmstart
        if type(self.warmstart[0]) not in [list, tuple]:
            # Get scores for warmstart
            warmstart_scores = self.get_scores(guesses=self.warmstart, target=self.target_embed)
            self.warmstart = list(zip(self.warmstart, warmstart_scores))

        # Get the textual prompt and embedding
        self.embeddings = self.model.get_input_embeddings().weight.clone().detach()
        # Get the hull of the embeddings across dimensions
        self.embeddings_hull = torch.stack([self.embeddings.min(dim=0).values, self.embeddings.max(dim=0).values])
        self.hidden_size = self.embeddings.shape[-1]
        self.text_prompt, input_ids = self.create_semantle_prompt(examples=self.warmstart,
                                                                  n_return=args.guesses_per_prompt)
        self.text_prompt_embed = self.embeddings[input_ids].view(1, -1, self.hidden_size).detach()

        # Soft-prompts
        self.n_prompt_tokens = n_prompt_tokens

        if random_proj == "none":
            self.linear = None
            self.intrinsic_dim = self.hidden_size
            print("Setting intrinsic dim to hidden size")
        else:
            self.linear = torch.nn.Linear(self.intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False,
                                          dtype=dtype, device=device)
            if random_proj == 'normal':
                torch.nn.init.normal_(self.linear.weight, -1, 1)
            elif random_proj == 'uniform':
                torch.nn.init.uniform_(self.linear.weight, -1, 1)

        self.guesses_raw = []
        self.guesses = []
        self.soft_prompts = []
        self.opt_soft_prompt = None
        self.scores_best_mean_var = []  # per soft prompt
        self.unique_guesses = set([x[0] for x in warmstart])
        self.repeats = 0
        self.evals = 0
        self.generation_errors = []
        self.n_skipped_cos_error = 0
        self.n_skipped_cos_cma_bound_error = 0
        self.best_warmstart = sorted(self.warmstart, key=lambda x: -x[1])[0]
        self.best_so_far = (self.best_warmstart[0], self.best_warmstart[1], None)  # word, score, prompt
        self.last_best = (self.best_warmstart[0], self.best_warmstart[1], None)  # word, score, prompt
        self.opt_found = False

    def eval(self, soft_prompts=None, no_prompt=False):
        add_decoding_kwargs = {}
        self.soft_prompts.append(soft_prompts.to('cpu'))
        if not no_prompt:
            soft_prompts = soft_prompts.to(device=self.text_prompt_embed.device, dtype=self.text_prompt_embed.dtype)
            if self.linear is not None:
                soft_prompts = self.linear(soft_prompts)
            soft_prompts = soft_prompts.view(-1, self.n_prompt_tokens, self.hidden_size)
            input_embed = torch.cat((soft_prompts, self.text_prompt_embed.repeat(soft_prompts.shape[0], 1, 1)), dim=1)
        else:
            input_embed = self.text_prompt_embed
            add_decoding_kwargs = {
                "num_return_sequences": self.decoding_kwargs["num_return_sequences"] * len(soft_prompts),
                "do_sample": True
            }

        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            outputs = self.model.generate(inputs_embeds=input_embed,
                                          max_new_tokens=self.args.max_new_tokens,
                                          pad_token_id=self.tokenizer.eos_token_id,
                                          **{**self.decoding_kwargs, **add_decoding_kwargs})
        guesses_raw_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.guesses_raw.append(guesses_raw_batch)

        iter_guesses_scores = []
        iter_last_best = []
        iter_scores_best_mean_var = []
        iter_generation_errors = []
        for i, guesses_raw in enumerate(guesses_raw_batch):
            # Get unique guesses
            guesses, errors = self.extract_guesses(guesses_raw)
            if guesses is None:
                self.n_skipped_cos_error += 1
                iter_scores_best_mean_var.append((-1., -1., 0))
                iter_guesses_scores.append([("", -1.)] * self.args.guesses_per_prompt)
                iter_last_best.append(("", -1., self.soft_prompts[-1][i]))
                continue
            guesses = guesses[:self.args.guesses_per_prompt]
            iter_generation_errors.append(errors)
            _len_unique_guesses = len(self.unique_guesses)
            self.unique_guesses.update(set(guesses))
            self.repeats += len(guesses) - (len(self.unique_guesses) - _len_unique_guesses)
            # Add ideal number of new guesses
            self.evals += len(guesses)

            # Get scores
            scores = self.get_scores(guesses=guesses, target=self.target_embed)
            scores_best = max(scores)
            scores_mean = np.mean(scores)
            scores_var = np.var(scores)
            iter_scores_best_mean_var.append((scores_best, scores_mean, scores_var))

            guesses_scores = sorted(list(zip(guesses, scores)), key=lambda x: x[1])
            iter_guesses_scores.append(guesses_scores)
            iter_last_best.append((guesses_scores[-1][0], guesses_scores[-1][1], self.soft_prompts[-1][i]))

            # Update best so far
            if guesses_scores[-1][1] > self.best_so_far[1]:
                self.best_so_far = (guesses_scores[-1][0], guesses_scores[-1][1], self.soft_prompts[-1][i])

            # Check if target is found
            if not self.opt_found and self.best_so_far[1] == 1.0:
                print(f"""\nOPTIMUM ("{self.target}") FOUND at iteration #{len(self.guesses)}\n""")
                self.opt_found = True
                self.opt_soft_prompt = self.soft_prompts[-1][i]

        self.guesses.append(iter_guesses_scores)
        self.last_best = sorted(iter_last_best, key=lambda x: x[1])[-1]
        self.scores_best_mean_var.append(iter_scores_best_mean_var)
        self.generation_errors.append(iter_generation_errors)

        return iter_scores_best_mean_var, [[__gs[1] for __gs in _gs] for _gs in iter_guesses_scores]

    def get_scores(self, guesses, target=None, batch_size=16, bbox_prompt=None, bbox_instruction=None):
        prompts = [(self.bbox_prompt if bbox_prompt is None else bbox_prompt) % guess for guess in guesses]

        embeds = self.bbox_model.encode(prompts,
                                        prompt=self.bbox_instruction if bbox_instruction is None else bbox_instruction,
                                        convert_to_tensor=True,
                                        normalize_embeddings=True,
                                        show_progress_bar=False,
                                        batch_size=batch_size)

        if target is None:
            return embeds.squeeze()

        scores = cosine_similarity(embeds, target.unsqueeze(0)).squeeze().tolist()
        return scores if type(scores) is list else [scores]

    def create_semantle_prompt(self, examples, n_return=1, sort=True):
        system = """You are a helpful chatbot with high attention to detail who is not talkative and responds only \
with the answer and no additional conversation. All your responses should be in JSON format, i.e. {key: value}, where \
the key is always \"response\" and the value can be a string, int, list, or dict, depending on the context."""

        user = """Your task is to guess a hidden word from the English dictionary. Use the below series of your \
previous guesses (in increasing order of their similarity to the hidden word in terms of their *meaning*) to make \
a new guess. Your new guess should not have been made before and should score higher than your previous guesses. \
Analyze your previous guesses to decide what word to guess next. If you guess an invalid word (i.e., not in the \
dictionary or a repeat guess), you will get no score, so stick to proper, single-word English words, and do not \
repeat your previous guesses!\n\nHere are your top previous guesses (from worst to best): """

        _examples = sorted(examples, key=lambda x: x[1]) if sort else examples
        user += ", ".join([x[0] for x in _examples]) + ".\n\n"

        user += f"""Now, guess exactly n={n_return} new word{"s" if n_return > 1 else ""} that is likely to be more \
similar to the hidden word than the previous guesses. (Note: give only a list of words in the provided JSON format, \
e.g. {{\"response\": [\"word1\", \"word2\",...]}})"""

        prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        prompt_templatized = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_templatized_tkns = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                                                     return_tensors="pt")

        return prompt_templatized, prompt_templatized_tkns

    def extract_guesses(self, guesses_raw, unique=True, response_key="response"):
        if type(guesses_raw) is str:
            guesses_raw = [guesses_raw]
        guesses = [guess.strip().lower() for guess in guesses_raw]
        parsed = []
        errors = []
        for idx, guess in enumerate(guesses):
            try:
                extracted = extract_json(guess)
                words = extracted[response_key]
                if unique:
                    words = list(set(words))
                parsed.append(words)
            except:
                errors.append((idx, guess))
                parsed.append(None)
                continue

        return parsed if len(guesses) > 1 else parsed[0], errors


def evaluate_soft_prompts(X, model_forward_api, args, initial=False, no_prompt=False):
    Y_best_mean_var, Y_scores = model_forward_api.eval(X, no_prompt=no_prompt)
    Y = [_Y[1] for _Y in Y_best_mean_var]
    Yvar = [_Y[2] for _Y in Y_best_mean_var]
    Ybest = [_Y[0] for _Y in Y_best_mean_var]

    if args.coupled_kernel == "scores":
        X_struct = torch.tensor(Y_scores)
    elif args.coupled_kernel == "instruct-embed":
        X_struct = model_forward_api.get_scores(guesses=model_forward_api.guesses_raw[-1], bbox_prompt="%s",
                                                bbox_instruction="")
    else:
        X_struct = torch.zeros((len(X), 1))

    X = X.to(**tkwargs)
    X_struct = X_struct.to(**tkwargs)
    Y = torch.tensor(Y).unsqueeze(-1).to(**tkwargs)
    Yvar = torch.tensor(Yvar).unsqueeze(-1).to(**tkwargs)

    if initial:
        print(f"\nBest initial point (mean): {Y.max().item():.4f}")
        print(f"Best initial point (max): {max(Ybest):.4f}\n")

    return X, X_struct, Y, Yvar


def run(args):
    # Temp hard coded warmstart
    warmstart = [["tuxedo", 0.24267719686031342], ["thundercloud", 0.29079192876815796],
                 ["outpatient", 0.2909504473209381], ["painter", 0.2917482256889343], ["group", 0.3045214116573334],
                 ["riverbank", 0.31173431873321533], ["maid", 0.32068318128585815], ["chapstick", 0.3297225832939148],
                 ["brick", 0.35291898250579834], ["hatchback", 0.38664406538009644], ["drawer", 0.3957057595252991],
                 ["hornet", 0.4009988605976105], ["spinout", 0.4301176965236664], ["clot", 0.4447917342185974],
                 ["understanding", 0.45225825905799866], ["money", 0.45715004205703735],
                 ["bureau", 0.46322184801101685], ["file", 0.5185025930404663], ["wires", 0.5240160226821899],
                 ["fixer", 0.5351479053497314]]

    if args.visualize_posterior:
        posterior_vals, viz_observed = {}, []
        viz_repr, viz_scores = torch.load(args.visualize_posterior)
        # sort in ascending order of y
        _argsort = viz_scores.argsort(dim=0).squeeze()
        viz_repr, viz_scores = viz_repr[_argsort], viz_scores[_argsort]

    model_forward_api = LMForwardAPI(model_name=args.model_name, bbox_model_name=args.bbox_model,
                                     random_proj=args.random_proj, intrinsic_dim=args.intrinsic_dim,
                                     n_prompt_tokens=args.n_prompt_tokens, warmstart=warmstart, target=args.task,
                                     args=args)

    # Start BO

    # Get warmstart points
    sobol = SobolEngine(dimension=model_forward_api.intrinsic_dim, scramble=True, seed=args.seed)  # from [0,1]^d
    with torch.no_grad():
        X = sobol.draw(args.n_init).to(**tkwargs)
    X, X_struct, Y, Yvar = evaluate_soft_prompts(X, model_forward_api, args, initial=True, no_prompt=args.no_prompt)

    # Set bounds
    bounds = None
    min_bounds = torch.ones(X.shape[1]).to(X.device) * -6.
    max_bounds = torch.ones(X.shape[1]).to(X.device) * 6.
    bounds = torch.stack([min_bounds, max_bounds])

    # Get kernel hyperparameters
    kernel_hparams = {
        **{k: v for k, v in {"lengthscale": args.kernel_lengthscale,
                             "lengthscale_prior_concentration": args.kernel_lengthscale_prior_concentration,
                             "lengthscale_prior_rate": args.kernel_lengthscale_prior_rate,
                             "lengthscale_instr": args.kernel_lengthscale_instr,
                             "lengthscale_prior_concentration_instr": args.kernel_lengthscale_prior_concentration_instr,
                             "lengthscale_prior_rate_instr": args.kernel_lengthscale_prior_rate_instr,
                             "outputscale": args.kernel_outputscale,
                             "outputscale_prior_concentration": args.kernel_outputscale_prior_concentration,
                             "outputscale_prior_rate": args.kernel_outputscale_prior_rate,
                             "mean": args.kernel_mean,
                             "mean_prior_mean": args.kernel_mean_prior_mean,
                             "mean_prior_std": args.kernel_mean_prior_std}.items() if v is not None and v != -100},
        "coupled_kernel": args.coupled_kernel
    }

    # Get GP model
    gp_model, gp_mll, requires_optim = get_gp(X, Y, X_struct, kernel_hparams,
                                              y_train_var=Yvar, standardize_outputs=True, normalize_inputs=True,
                                              bounds=bounds, bounds_margin=1, symmetric_bounds=False)

    for i in (pbar := tqdm(range(args.n_iterations))):
        pbar.set_description(f"Iteration {i + 1}")
        pbar.set_postfix({
            "last_best": (model_forward_api.last_best[0], round(model_forward_api.last_best[1], 4)),
            "best_so_far": (model_forward_api.best_so_far[0], round(model_forward_api.best_so_far[1], 4)),
        })

        if model_forward_api.opt_found:
            break

        # Fit the GP after the new set of observations
        if requires_optim:
            fit_gpytorch_model(gp_mll)
        if args.verbose:
            print(f"\nLearned GP mean = {gp_model.mean_module.constant.item()}")
            if args.coupled_kernel != "none":
                print(f"Learned GP lengthscale = {gp_model.covar_module.base_kernel.base_latent_kernel.lengthscale}")
                print(
                    f"Learned GP lengthscale (instruction) = {gp_model.covar_module.base_kernel.instruction_kernel.lengthscale}")
            else:
                print(f"Learned GP lengthscale = {gp_model.covar_module.base_kernel.lengthscale}")
            print(f"Learned GP outputscale = {gp_model.covar_module.outputscale.item()}\n")

        EI = ExpectedImprovement(gp_model, best_f=Y.max().item())

        # Sample new points to evaluate
        starting_idxs = torch.argsort(-1 * Y.squeeze())[:args.batch_size]
        starting_points = X[starting_idxs]
        best_points = []
        best_vals = []
        for starting_point_for_cma in starting_points:
            # Check that each dim of starting_point_for_cma is within bounds
            if torch.any(starting_point_for_cma < bounds[0]) or torch.any(starting_point_for_cma > bounds[1]):
                model_forward_api.n_skipped_cos_cma_bound_error += 1
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                newp, newv = cma_es_concat(starting_point_for_cma, EI, tkwargs, bounds=bounds, silent=True)
            best_points.append(newp)
            best_vals.append(newv)
        # print(f"best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}")
        # print(f"Time for CMA-ES {time.time() - start_time}")
        if args.random_prompt:
            # Sample a random soft prompt instead of using the BO proposal
            with torch.no_grad():
                X_next = sobol.draw(len(best_vals))
        else:
            X_next = torch.from_numpy(np.array(best_points)[np.argsort(-1 * np.array(best_vals))]).float()
        X_next, X_next_struct, Y_next, Yvar_next = evaluate_soft_prompts(X_next, model_forward_api, args, initial=False,
                                                                         no_prompt=args.no_prompt)

        X = torch.cat([X, X_next])
        X_struct = torch.cat([X_struct, X_next_struct])
        Y = torch.cat([Y, Y_next])
        Yvar = torch.cat([Yvar, Yvar_next])

        # Get GP model
        gp_model, gp_mll, requires_optim = get_gp(X, Y, X_struct, kernel_hparams,
                                                  y_train_var=Yvar, standardize_outputs=True, normalize_inputs=True,
                                                  bounds=bounds, bounds_margin=1, symmetric_bounds=False)

    return model_forward_api


if __name__ == '__main__':
    args = parse_args()

    # Temp
    args.task = "computer"

    print("Script arguments:")
    print(args.__dict__)

    # Get res dirs
    res_dirname = f"semantle/{args.out_file + '_' if args.out_file is not None else ''}{args.model_name}_{args.bbox_model}"
    global OUT_DIR
    OUT_DIR = f"results/{res_dirname}/{args.task}"

    # evaluation budget
    print(f"\nUsing a total of {args.n_init + args.batch_size * args.n_iterations} soft-prompts")
    print(
        f"\nUsing a total of {(args.n_init + args.batch_size * args.n_iterations) * args.guesses_per_prompt} bbox evaluations")
    print("\n" + set_all_seed(args.seed) + "\n")
    runner_obj = run(args=args)

    os.makedirs(OUT_DIR, exist_ok=True)
    res_fpath = f"{OUT_DIR}/seed-{args.seed}.json"

    print("\nFinished!")

    results = {
        "optimized": runner_obj.opt_found,
        "best_so_far": (runner_obj.best_so_far[0], runner_obj.best_so_far[1]),
        "best_warmstart": (runner_obj.best_warmstart[0], runner_obj.best_warmstart[1]),
        "n_unique_guesses": len(runner_obj.unique_guesses),
        "max_bbox_evaluations": (args.n_init + args.batch_size * args.n_iterations) * args.guesses_per_prompt,
        "max_soft_prompts": args.n_init + args.batch_size * args.n_iterations,
        "n_repeats": runner_obj.repeats,
        "n_skipped_cos_error": runner_obj.n_skipped_cos_error,
        "n_skipped_cos_cma_bound_error": runner_obj.n_skipped_cos_cma_bound_error,
        "n_evals": runner_obj.evals,
        "guesses": runner_obj.guesses,
        "args": args.__dict__
    }

    with open(res_fpath, 'w') as fh:
        fh.write(json.dumps(results, indent=2))
    # Print summary of the shorter version of the results
    print(f"\nResults:")
    print(json.dumps({k: v for k, v in results.items() if k in [
        "optimized", "best_so_far", "best_warmstart", "n_unique_guesses", "n_repeats", "max_bbox_evaluations",
        "max_soft_prompts", "n_skipped_cos_error", "n_skipped_cos_cma_bound_error", "n_evals"
    ]}, indent=2))

    print(f"\nSaved results to: {res_fpath}\n\n")

    if args.debug:
        breakpoint()
