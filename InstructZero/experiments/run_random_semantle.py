"""
SUMMARY:
Given a set of soft prompts, we randomly initialize them using SobelEngine
We then generate a word using LLaMA or WizardLM given the soft prompts and the instruction.
We then compute the similarity between the generated word and the target word.
This score is used to pick up new initialization for the soft prompts using Bayesian Optimization.
Repeat the process until convergence.
"""

import json
import torch
import numpy as np
import copy
import logging
from automatic_prompt_engineer import ape, data
from data.instruction_induction.load_data import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from automatic_prompt_engineer import evaluate, config, template, data
import os
from misc import get_test_conf
from evaluation.instruction_induction.exec_accuracy import (
    exec_evaluator,
    exec_accuracy_evaluator,
)

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
from torch.nn.functional import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_token = "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"


def semantle_evaluator(model, instruction, target_word):
    # Generate the word
    word = instruction[0]
    if not word:
        word = "None"
    emb1 = torch.tensor(model.encode(word))
    emb2 = torch.tensor(model.encode(target_word))
    score = cosine_similarity(emb1, emb2, dim=0).item()
    return score, [[score]]


def get_conf(task):
    conf = {
        "generation": {
            "num_subsamples": 1,
            "num_demos": 10,
            "num_prompts_per_subsample": 20,
            "model": {
                "gpt_config": {
                    # 'model': 'text-ada-001'
                }
            },
        },
        "evaluation": {
            "method": semantle_evaluator,
            "task": task,
            "num_samples": 0,
            "model": {
                "gpt_config": {
                    # 'model': 'text-ada-001'
                }
            },
        },
    }
    return conf


class LMForwardAPI:
    def __init__(
        self,
        model_name=None,
        init_prompt=None,
        init_qa=None,
        conf=None,
        base_conf=None,
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
        if self.ops_model in ["vicuna", "wizardlm", "gemma"]:
            self.system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.role = ["USER:", "ASSISTANT:"]
        elif self.ops_model == "alpaca":
            self.system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.role = ["### Instruction:", "### Response:"]
        elif self.ops_model == "llama3-8B":
            self.messages = []
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

        if args.api_model in ["llama3-8b", "flan-t5"]:
            self.api_model = exec_evaluator(args.api_model, self.conf)
        else:
            self.api_model = args.api_model

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()

    def eval(self, similarity_model, prompt_embedding=None, target_word=""):
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
        if self.ops_model == "llama3-8B":
            messages = [{"role": "user", "content": self.init_token}]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ).replace("<|begin_of_text|><|start_header_id|>user<|end_header_id|>", "")
        else:
            input_text = f"{self.system_prompt} USER:{self.init_token} ASSISTANT:"

        inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )
        prefix = self.tokenizer(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
            return_tensors="pt",
        )
        prefix_emb = self.embedding[prefix.input_ids]
        prompt_embedding = torch.cat((prefix_emb, prompt_embedding.cuda()), 1)

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
            max_new_tokens=256,
            attention_mask=extended_attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # print post-processed instruction
        logging.info(f"Instruction: {instruction}")

        if instruction[0] in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
        else:
            dev_perf, instruction_score = semantle_evaluator(
                similarity_model, instruction, target_word
            )
            self.prompts_set[instruction[0]] = (dev_perf, instruction_score)

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

    prompt_gen_template = "\nGenerate only one word.\n\n"

    if args.api_model == "chatgpt":
        base_conf = "../configs/instruction_induction.yaml"
    elif args.api_model == "llama3-8b":
        base_conf = "../configs/instruction_induction_llama3-8b.yaml"
    conf = get_conf(task)

    model_forward_api = LMForwardAPI(
        model_name=args.model_name,
        init_prompt=[""],
        init_qa=[prompt_gen_template],
        conf=conf,
        base_conf=base_conf,
        random_proj=random_proj,
        intrinsic_dim=intrinsic_dim,
        n_prompt_tokens=n_prompt_tokens,
        HF_cache_dir=HF_cache_dir,
        args=args,
    )

    # start bayesian optc
    itr = 0
    X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(
        N_INIT * (N_ITERATIONS + 1)
    )
    _ = [
        model_forward_api.eval(similarity_model, x, args.semantle_word)
        for x in X[itr : (itr + N_INIT)]
    ]
    itr += N_INIT

    for i in range(N_ITERATIONS):  # 5 iterations, 25 batch size.

        for idx in range(N_INIT):
            X_next_point = X[idx + itr]

            _ = [
                model_forward_api.eval(
                    similarity_model, X_next_point, args.semantle_word
                )
            ]
        itr += N_INIT

    logging.info("Evaluate on test data...")
    prompts = model_forward_api.return_best_prompt()
    logging.info("Best instruction is:")
    logging.info(prompts)

    logging.info("The final instruction set is:")
    logging.info(model_forward_api.return_prompts_set())

    # Evaluate on test data
    logging.info("Evaluating on test data...")

    test_res, _ = semantle_evaluator(similarity_model, prompts, args.semantle_word)
    return test_res
    # print(f'Test score on ChatGPT: {test_score}')


if __name__ == "__main__":

    args = parse_args()

    # Get the current timestamp and format it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"random_semantle_{args.task}_{args.instruct_method}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )

    # evaluation budget
    logging.info(
        f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations"
    )
    logging.info(set_all_seed(args.seed))
    test_score = run(args=args)
    logging.info("Finished!!!")
    logging.info(f"Test score on {args.api_model}: {test_score}")
