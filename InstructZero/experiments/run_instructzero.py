import random
import torch
import numpy as np
import copy
import logging
from automatic_prompt_engineer import ape, data
from data.instruction_induction.load_data import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from automatic_prompt_engineer import evaluate, config, template, data
import os
import re
from misc import get_test_conf, get_conf
from evaluation.instruction_induction.exec_accuracy import exec_evaluator

from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from instruction_coupled_kernel import *
import time
from datetime import datetime

from args import parse_args
from misc import set_all_seed, TASKS, tkwargs, N_INIT, BATCH_SIZE, N_ITERATIONS

import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_token = "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"

# Get the current timestamp and format it
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logfile_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)


class LMForwardAPI:
    def __init__(
        self,
        model_name=None,
        eval_data=None,
        init_prompt=None,
        init_qa=None,
        conf=None,
        base_conf=None,
        prompt_gen_data=None,
        random_proj=None,
        intrinsic_dim=None,
        n_prompt_tokens=None,
        few_shot_data=None,
        HF_cache_dir=None,
        args=None,
    ):
        p = torch.ones(10)

        kwargs = {"torch_dtype": torch.float16, "use_cache": True}
        self.ops_model = model_name
        # import pdb; pdb.set_trace()
        if self.ops_model in ["vicuna", "wizardlm", "openchat", "gemma", "llama3-8B"]:
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_cache_dir,
                low_cpu_mem_usage=True,
                device_map="auto",
                token=hf_token,
                **kwargs,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                HF_cache_dir,
                model_max_length=1024,
                padding_side="left",
                use_fast=False,
                token=hf_token,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError

        self.init_token = init_prompt[0] + init_qa[0]
        if self.ops_model in ["wizardlm", "vicuna", "openchat", "gemma", "llama3-8B"]:
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(
                init_prompt, return_tensors="pt"
            ).input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]

        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        logging.info(f"Shape of initial prompt embedding: {self.init_prompt.shape}")

        # self.init_prompt = self.init_prompt.reshape(self.n_prompt_tokens * self.hidden_size)
        # Create the template for Vicuna and WizardLM
        self.count = 0
        self.linear = torch.nn.Linear(
            intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False
        )
        if self.ops_model in ["vicuna", "wizardlm", "gemma", "llama3-8B"]:
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ["USER:", "ASSISTANT:"]
        elif self.ops_model == "alpaca":
            self.system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.role = ["### Instruction:", "### Response:"]
        else:
            NotImplementedError

        if random_proj == "normal":
            # calculate std for normal distribution
            if model_name in ["wizardlm", "vicuna", "openchat", "gemma", "llama3-8B"]:
                logging.info("Get the embedding firstly to avoid issues")
            else:
                raise NotImplementedError
            mu_hat = self.embedding.reshape(-1).mean().item()
            std_hat = self.embedding.reshape(-1).std().item()
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(intrinsic_dim) * args.sigma)

            logging.info(
                f"[Embedding] mu: {mu_hat} | std: {std_hat} [RandProj]  mu: {mu} | std: {std}"
            )
            torch.nn.init.normal_(self.linear.weight, -1, 1)
        elif random_proj == "uniform":
            torch.nn.init.uniform_(self.linear.weight, -1, 1)

        # eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate(
            "Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]"
        )
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")

        if args.api_model in ["llama3-8b", "flan-t5"]:
            self.api_model = exec_evaluator(args.api_model, self.conf)
        else:
            self.api_model = args.api_model

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()

    def eval(self, prompt_embedding=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim

        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            # if self.init_prompt is not None:
            #     prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        elif isinstance(prompt_embedding, torch.Tensor):
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f"[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead."
            )
        # create the input text with the system prompt
        input_text = f"{self.system_prompt} USER:{self.init_token} ASSISTANT:"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )

        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(
            device=input_embed.device, dtype=input_embed.dtype
        )
        input_embed = torch.cat((prompt_embedding, input_embed), 1)

        prompt_length = prompt_embedding.shape[1]
        extended_attention_mask = torch.cat(
            [
                torch.ones(
                    (attention_mask.shape[0], prompt_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ],
            dim=1,
        )

        outputs = self.model.generate(
            inputs_embeds=input_embed,
            max_new_tokens=128,
            attention_mask=extended_attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # postprocess instruction
        # instruction[0] = 'The instruction was to ' + instruction[0]
        # import pdb; pdb.set_trace()
        # start = instruction[0].find('The instruction was to')
        # end = instruction[0].find('Comment:')
        # if end == -1:
        #     instruction[0] = instruction[0][start:]
        # else:
        #     instruction[0] = instruction[0][start: end]

        # sentences = re.split(r' *[\.\?!][\'"\)\]]* *', instruction[0])
        # search_string = 'The instruction was to'
        # for sentence in sentences:
        #     if sentence.startswith(search_string):
        #         instruction[0] = sentence
        #         break

        # print post-processed instruction
        logging.info(f"Instruction: {instruction}")

        if instruction[0] in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
        else:
            if self.api_model in ["chatgpt"]:
                dev_perf, instruction_score = evaluate.evaluate_prompts(
                    instruction,
                    self.eval_template,
                    self.eval_data,
                    self.demos_template,
                    self.few_shot_data,
                    self.conf["evaluation"]["method"],
                    self.conf["evaluation"],
                )
                dev_perf = dev_perf.sorted()[1][0]
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            # We will fix the bugs for other api models. Stay tuned!
            elif self.api_model.api_model_name in ["llama", "flan-t5", "llama3-8b"]:
                dev_perf, instruction_score = self.api_model.evaluate(
                    instruction,
                    self.eval_template,
                    self.eval_data,
                    self.demos_template,
                    self.few_shot_data,
                    self.conf["evaluation"],
                )
                dev_perf = dev_perf.sorted()[1][0]
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            else:
                raise NotImplementedError

        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_instruction = instruction

        logging.info(
            f"Dev loss: {round(float(dev_perf), 4)}. Dev perf: {round(float(dev_perf), 4)}. Best dev perf: {round(float(self.best_dev_perf), 4)}"
        )
        logging.info("********* Done *********")

        return dev_perf, instruction_score, outputs, instruction[0]

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set


def run(args):
    task, HF_cache_dir = args.task, args.HF_cache_dir
    random_proj, intrinsic_dim, n_prompt_tokens = (
        args.random_proj,
        args.intrinsic_dim,
        args.n_prompt_tokens,
    )

    assert args.task in TASKS, "Task not found!"

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    induce_data, test_data = load_data("induce", task), load_data("eval", task)

    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size), 100)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data, eval_data = data.create_split(induce_data, prompt_gen_size)

    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data = prompt_gen_data[0], [
        random.sample(output, 1)[0] for output in prompt_gen_data[1]
    ]
    # import pdb; pdb.set_trace()
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOUTPUT: [OUTPUT]"  # change the evaluation template
    init_prompt = ["\n"]
    # TODO : Can we write a better prompt?
    prompt_gen_template = "[full_DEMO]\n\nThe instruction was to?"
    # prompt_gen_template = "[full_DEMO]\n\nWhat w  as the instruction for the task?"
    # prompt_gen_template = "[full_DEMO]\n\n Please generate appropriate instructions for the task."

    if args.api_model == "chatgpt":
        base_conf = "../configs/instruction_induction.yaml"
    elif args.api_model == "llama3-8b":
        base_conf = "../configs/instruction_induction_llama3-8b.yaml"
    conf = get_conf(task, eval_data)

    # make the demo automatically
    subsampled_data = data.subsample_data(
        prompt_gen_data, conf["generation"]["num_demos"]
    )
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]


    model_forward_api = LMForwardAPI(
        model_name=args.model_name,
        eval_data=eval_data,
        init_prompt=init_prompt,
        init_qa=init_qa,
        conf=conf,
        base_conf=base_conf,
        prompt_gen_data=prompt_gen_data,
        random_proj=random_proj,
        intrinsic_dim=intrinsic_dim,
        n_prompt_tokens=n_prompt_tokens,
        HF_cache_dir=HF_cache_dir,
        args=args,
    )

    # start bayesian optc
    X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(N_INIT)
    X_return = [model_forward_api.eval(x) for x in X]
    x_vec = [x[2] for x in X_return]
    x_text = [x[3] for x in X_return]

    Y = [X[0] for X in X_return]
    Y_scores = [X[1].squeeze() for X in X_return]

    X = X.to(**tkwargs)
    Y = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
    Y_scores = torch.FloatTensor(np.array(Y_scores)).to(**tkwargs)
    logging.info(f"Best initial point: {Y.max().item():.3f}")

    # standardization Y (no standardization for X)
    X_train = X
    y_train = (Y - Y.mean(dim=-2)) / (Y.std(dim=-2) + 1e-9)
    x_vec = torch.nn.utils.rnn.pad_sequence(
        [x.squeeze(0) for x in x_vec], batch_first=True, padding_value=0
    ).to(dtype=torch.float64)


    # define matern kernel
    matern_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=X_train.shape[-1],
        lengthscale_prior=GammaPrior(3.0, 6.0),
    )
    if args.instruct_method == "default":
        matern_kernel_instruction = MaternKernel(
            nu=2.5,
            ard_num_dims=Y_scores.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        vector_train = None
    elif args.instruct_method == "vec_sim":
        matern_kernel_instruction = VectorSimilarityKernel()
        vector_train = x_vec
    elif args.instruct_method == "txt_sim":
        matern_kernel_instruction = TextSimilarityKernel(similarity_model)
        vector_train = x_text

    covar_module = ScaleKernel(
        base_kernel=CombinedStringKernel(
            base_latent_kernel=matern_kernel,
            instruction_kernel=matern_kernel_instruction,
            latent_train=X_train.double(),
            instruction_train=Y_scores,
            vector_train=vector_train,  # Shape: (num_points, num_features)
        )
    )

    # covar_module = ScaleKernel(VectorSimilarityKernel())
    gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
    gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    for i in range(N_ITERATIONS):  # 5 iterations, 25 batch size.
        # print([p for p in model_forward_api.linear.parameters()][0])
        # breakpoint()
        logging.info(f"X_train shape {X_train.shape}")
        # logging.info(f"y_train shape {y_train.shape}")

        start_time = time.time()

        fit_gpytorch_model(gp_mll)  # , options = {'maxiter':10})
        logging.info(f"Fitting done in {time.time()-start_time}")
        start_time = time.time()
        EI = ExpectedImprovement(gp_model, best_f=y_train.max().item())
        starting_idxs = torch.argsort(-1 * y_train.squeeze())[:BATCH_SIZE]
        print(starting_idxs)
        starting_points = X_train[starting_idxs]
        print("Starting points shape:", starting_points.shape)
        best_points = []
        best_vals = []
        for starting_point_for_cma in starting_points:
            if (
                torch.max(starting_point_for_cma) > 1
                or torch.min(starting_point_for_cma) < -1
            ):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI, tkwargs)
            # newp, newv = adam_optimizer(starting_point_for_cma, EI, tkwargs)
            best_points.append(newp)
            best_vals.append(newv)
        print(best_vals)
        print(best_points)
        logging.info(
            f"best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}"
        )
        logging.info(f"Time for CMA-ES {time.time() - start_time}")
        for idx in np.argsort(-1 * np.array(best_vals)):
            X_next_point = torch.from_numpy(best_points[idx]).float().unsqueeze(0)
            # Y_next_point = [model_forward_api.eval(X_next_point)]

            X_next_points_return = [model_forward_api.eval(X_next_point)]
            Y_next_point = [X[0] for X in X_next_points_return]
            Y_scores_next_points = [X[1].squeeze() for X in X_next_points_return]
            x_vec_next_point = [x[2] for x in X_next_points_return]
            x_text_next_point = [x[3] for x in X_next_points_return]

            x_vec = torch.nn.utils.rnn.pad_sequence(
                [x.squeeze(0) for x in x_vec]
                + [x.squeeze(0) for x in x_vec_next_point],
                batch_first=True,
                padding_value=0,
            ).to(dtype=torch.float64)

            X_next_point = X_next_point.to(**tkwargs)
            Y_next_point = torch.FloatTensor(Y_next_point).unsqueeze(-1).to(**tkwargs)
            Y_scores_next_points = torch.FloatTensor(np.array(Y_scores_next_points)).to(
                **tkwargs
            )
            x_text = x_text + x_text_next_point

            X = torch.cat([X, X_next_point])
            Y = torch.cat([Y, Y_next_point])
            Y_scores = torch.cat([Y_scores, Y_scores_next_points])

        # standardization Y
        X_train = X.clone()
        y_train = (Y - Y.mean(dim=-2)) / (Y.std(dim=-2) + 1e-9)

        matern_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=X_train.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )

        if args.instruct_method == "default":
            matern_kernel_instruction = MaternKernel(
                nu=2.5,
                ard_num_dims=Y_scores.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
            vector_train = None
        elif args.instruct_method == "vec_sim":
            matern_kernel_instruction = VectorSimilarityKernel()
            vector_train = x_vec
        elif args.instruct_method == "txt_sim":
            matern_kernel_instruction = TextSimilarityKernel(similarity_model)
            vector_train = x_text

        covar_module = ScaleKernel(
            base_kernel=CombinedStringKernel(
                base_latent_kernel=matern_kernel,
                # instruction_kernel=TextSimilarityKernel(similarity_model),
                instruction_kernel=matern_kernel_instruction,
                latent_train=X_train.double(),
                instruction_train=Y_scores,
                vector_train=vector_train,  # Shape: (num_points, num_features)
            )
        )
        gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        logging.info(f"Best value found till now: {torch.max(Y)}")

    logging.info("Evaluate on test data...")
    prompts = model_forward_api.return_best_prompt()
    logging.info("Best instruction is:")
    logging.info(prompts)

    logging.info("The final instruction set is:")
    logging.info(model_forward_api.return_prompts_set())

    # Evaluate on test data
    logging.info("Evaluating on test data...")

    test_conf = get_test_conf(task, test_data)

    test_res = ape.evaluate_prompts(
        prompts=prompts,
        eval_template=eval_template,
        eval_data=test_data,
        few_shot_data=prompt_gen_data,
        demos_template=demos_template,
        conf=test_conf,
        base_conf=base_conf,
    )
    test_res = test_res[0]
    test_score = test_res.sorted()[1][0]
    return test_score
    # print(f'Test score on ChatGPT: {test_score}')


if __name__ == "__main__":
    args = parse_args()
    # evaluation budget
    logging.info(
        f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations"
    )
    logging.info(set_all_seed(args.seed))
    test_score = run(args=args)
    logging.info("Finished!!!")
    logging.info(f"Test score on {args.api_model}: {test_score}")
