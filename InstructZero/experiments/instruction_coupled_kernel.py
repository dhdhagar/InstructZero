# gpytorch for substring kernel implemented in https://github.com/henrymoss/BOSS/tree/master/boss/code/kernels/string
from gpytorch.kernels.kernel import Kernel
from gpytorch.constraints import Positive
import torch
import numpy as np
from tqdm.auto import tqdm, trange
from pathlib import Path
import json
import numpy as np
import torch
import os
import cma
import torch.optim as optim


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class VectorSimilarityKernel(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.register_parameter(name="scale", parameter=torch.nn.Parameter(torch.ones(1)))

    def forward(self, x1, x2, **params):
        pooled_x1 = torch.mean(x1, dim=1).unsqueeze(0)
        pooled_x2 = torch.mean(x2, dim=1).unsqueeze(0)

        norm_x1 = pooled_x1 / pooled_x1.norm(dim=1, keepdim=True)
        norm_x2 = pooled_x2 / pooled_x2.norm(dim=1, keepdim=True)

        cosine_similarity = torch.mm(norm_x1.t(), norm_x2)  # * self.scale
        return cosine_similarity


class TextSimilarityKernel(Kernel):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        # self.register_parameter(
        #     name="raw_scale", parameter=torch.nn.Parameter(torch.zeros(1))
        # )
        # self.register_constraint("raw_scale", Positive())

    def forward(self, x1, x2, **params):
        # Convert texts to embeddings
        emb1 = torch.tensor(self.model.encode(x1))
        emb2 = torch.tensor(self.model.encode(x2))

        # Normalize the embeddings
        norm_emb1 = emb1 / emb1.norm(dim=1, keepdim=True)
        norm_emb2 = emb2 / emb2.norm(dim=1, keepdim=True)

        # Compute the cosine similarity
        cosine_similarity = torch.mm(norm_emb1, norm_emb2.t())

        return cosine_similarity.cuda()


class CombinedStringKernel(Kernel):
    def __init__(
        self,
        base_latent_kernel,
        instruction_kernel,
        latent_train,
        instruction_train,
        vector_train,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_latent_kernel = (
            base_latent_kernel  # Kernel on the latent space (Matern Kernel)
        )
        self.instruction_kernel = (
            instruction_kernel  # Kernel on the latent space (Matern Kernel)
        )
        self.latent_train = latent_train  # normalized training input
        self.lp_dim = self.latent_train.shape[-1]
        self.instruction_train = instruction_train  # SMILES format training input #self.get_smiles(self.latent_train)#.clone())
        self.vector_train = vector_train

    def forward(self, z1, z2, **params):
        # z1 and z2 are unnormalized
        check_dim = 0
        if len(z1.shape) > 2:
            check_dim = z1.shape[0]
            z1 = z1.squeeze(1)
        if len(z2.shape) > 2:
            check_dim = z2.shape[0]
            z2 = z2[0]
        latent_train_z1 = z1[:, : self.lp_dim]
        latent_train_z2 = z2[:, : self.lp_dim]

        if self.vector_train is not None:
            K_train_instruction = self.instruction_kernel.forward(
                self.vector_train, self.vector_train, **params
            )
        else:
            K_train_instruction = self.instruction_kernel.forward(
                self.instruction_train, self.instruction_train, **params
            )
        latent_space_kernel = self.base_latent_kernel.forward(
            self.latent_train, self.latent_train, **params
        )
        K_z1_training = self.base_latent_kernel.forward(
            latent_train_z1, self.latent_train, **params
        )
        K_z2_training = self.base_latent_kernel.forward(
            latent_train_z2, self.latent_train, **params
        )
        latent_space_kernel_inv = torch.inverse(
            latent_space_kernel
            + 0.0001 * torch.eye(len(self.latent_train)).to(latent_space_kernel.device)
        )

        kernel_val = (
            K_z1_training
            @ latent_space_kernel_inv
            @ (K_train_instruction)
            @ latent_space_kernel_inv
            @ K_z2_training.T
        )
        if check_dim > 0:
            kernel_val = kernel_val.unsqueeze(1)
        return kernel_val


def cma_es_concat(starting_point_for_cma, EI, tkwargs):
    if starting_point_for_cma.type() == "torch.cuda.DoubleTensor":
        starting_point_for_cma = starting_point_for_cma.detach().cpu().squeeze()
    es = cma.CMAEvolutionStrategy(
        x0=starting_point_for_cma,
        sigma0=0.8,
        inopts={"bounds": [-1, 1], "popsize": 50},
    )
    iter = 1
    while not es.stop():
        iter += 1
        xs = es.ask()
        X = torch.tensor(np.array(xs)).float().unsqueeze(1).to(**tkwargs)
        with torch.no_grad():
            Y = -1 * EI(X)
        es.tell(xs, Y.cpu().numpy())  # return the result to the optimizer
        print("current best")
        print(f"{es.best.f}")
        if iter > 10:
            break

    return es.best.x, -1 * es.best.f


def adam_optimizer(starting_point_for_adam, EI, tkwargs):
    starting_point_for_adam = (
        starting_point_for_adam.clone().detach().requires_grad_(True).to(**tkwargs)
    )
    optimizer = optim.Adam([starting_point_for_adam], lr=0.1)

    for _ in range(10):
        optimizer.zero_grad()
        loss = -1 * EI(starting_point_for_adam)
        loss.backward()
        optimizer.step()
        print(f"Current best: {-loss.item()}")

    return starting_point_for_adam.detach().cpu().numpy(), -loss.item()
