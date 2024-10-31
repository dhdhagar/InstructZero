import torch
import random
import numpy as np
from evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
import os
import math
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior, NormalPrior
from instruction_coupled_kernel import *
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP

TASKS = [
    'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
    'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation',
    'num_to_verbal', 'active_to_passive', 'singular_to_plural', 'rhymes',
    'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
    'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
    'translation_en-fr', 'word_in_context', 'auto_categorization', 'auto_debugging', 'ascii', 'cs_algorithms',
    'periodic_elements', 'word_sorting', 'word_unscrambling', 'odd_one_out', 'object_count'
]

SMOKE_TEST = os.environ.get("SMOKE_TEST")
## bayesian opt
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

N_INIT = 25 if not SMOKE_TEST else 5
N_ITERATIONS = 5 if not SMOKE_TEST else 1
BATCH_SIZE = 25 if not SMOKE_TEST else 1


def get_test_conf(task, test_data):
    test_conf = {
        'generation': {
            'num_subsamples': 3,
            'num_demos': 5,
            'num_prompts_per_subsample': 0,
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            # option: accuracy (cannot use likelihood here due to the textual outputs from ChatGPT do not have log prob)
            'num_samples': min(100, len(test_data[0])),
            'task': task,
            'model': {
                "name": "GPT_forward",
                'gpt_config': {
                    'model': 'gpt-3.5-turbo',  # gpt-3.5-turbo
                }
            }
        }
    }
    return test_conf


def get_conf(task, eval_data):
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 10,
            'num_prompts_per_subsample': 20,
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            'model': {
                "name": "GPT_forward",
                'gpt_config': {
                    'model': 'gpt-3.5-turbo'
                }
            }
        }
    }
    return conf


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return f"Set all the seeds to {seed} successfully!"


def plot_posterior(posterior_vals, posterior_cands, path, animate=False, anim_interval=300, anim_repeat=True,
                   obs_xy=None, trend_over_unobserved=True, conf_intervals=[95], top_k=None):
    posterior_cands_x = [c[0] for c in posterior_cands]

    # Bayesian credible interval to std map
    conf_to_std = {
        95: 1.96,
        99: 2.576
    }
    all_vals = np.array([posterior_vals[k] for k in posterior_vals])
    plt.clf()
    fig, ax = plt.subplots()
    obs_xy_in_viz = []
    warmstart_obs_xy_in_viz = []

    def update(frame):
        ax.clear()
        vals = all_vals[len(all_vals) - 1 if not animate else (frame if frame < len(all_vals) else (frame - 1))]
        _obs_xy = obs_xy[frame] if obs_xy is not None else []
        y, mean, std = vals[:, 0], vals[:, 1], vals[:, 2]
        x = 1 + np.arange(len(y))
        ax.plot(x, y, label="True")
        ax.plot(x, mean, label="Posterior", color="orange", alpha=0.8)
        # Plot observations
        _obs_xy_in_viz = []
        for xy in _obs_xy:
            try:
                _obs_xy_in_viz.append((posterior_cands_x.index(xy[0]), xy[1]))
            except:
                continue
        if frame == 0:
            warmstart_obs_xy_in_viz.extend(_obs_xy_in_viz)
        else:
            obs_xy_in_viz.extend(_obs_xy_in_viz)
        if len(warmstart_obs_xy_in_viz) > 0:
            obs_x, obs_y = zip(*warmstart_obs_xy_in_viz)
            ax.scatter(obs_x, obs_y, label="Warmstart", color="gray")
        if len(obs_xy_in_viz) > 0:
            obs_x, obs_y = zip(*obs_xy_in_viz)
            ax.scatter(obs_x, obs_y, label="BO Observation", color="red")
        # Plot uncertainty
        for conf_interval in conf_intervals:
            ax.fill_between(x,
                            mean - (conf_to_std[conf_interval] * std),
                            mean + (conf_to_std[conf_interval] * std),
                            alpha=0.4, color="orange")
        # Plot trend
        if trend_over_unobserved:
            obs_idxs = [o[0] for o in obs_xy_in_viz]
            unobs_idxs = np.array([i for i in range(len(x)) if i not in obs_idxs]).astype(int)
        posterior_trend = np.polyfit(x[unobs_idxs] if trend_over_unobserved else x,
                                     mean[unobs_idxs] if trend_over_unobserved else mean,
                                     8)
        posterior_trend = np.poly1d(posterior_trend)
        posterior_trend_y = posterior_trend(x)
        ax.plot(x, posterior_trend_y, "r--", label=f"Posterior Mean Trend")  # (s={_posterior_trend[0]})
        ax.legend()
        ax.set_title(f"t = {frame}")
        ax.set_ylabel("Objective")
        # Set y limit to be the same for all frames
        # ax.set_ylim([-3.1, 2])
        ax.set_ylim([-0.1, 1.1])  # ax.set_ylim([-5, 5] if args.surrogate_fn == "laplace" else [-0.1, 1.1])
        ax.set_xlabel("Candidates (sorted by ground-truth scores)")
        if top_k is not None:
            ax.set_xlim([0, top_k + 1])
        ax.grid()

    update(frame=0)  # initial plot

    if animate:
        ani = animation.FuncAnimation(fig=fig, func=update, frames=range(len(all_vals) + 1), interval=anim_interval,
                                      repeat=anim_repeat, repeat_delay=50)
        ani.save(path.replace(".json", ".gif"), writer="pillow")  # imagemagick
    else:
        plt.savefig(path.replace(".json", ".png"), bbox_inches="tight")

def get_gp(X_train, y_train, Y_scores, kernel_hparams):
    # define matern kernel
    matern_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=X_train.shape[-1],
        **{k: v for k, v in {"lengthscale_prior": GammaPrior(
            kernel_hparams.get('lengthscale_prior_concentration'),
            kernel_hparams.get('lengthscale_prior_rate')) if (kernel_hparams.get(
            'lengthscale_prior_concentration') is not None and kernel_hparams.get(
            'lengthscale_prior_rate') is not None) else None}.items() if
           v is not None}
    )
    matern_kernel_instruction = MaternKernel(
        nu=2.5,
        ard_num_dims=Y_scores.shape[-1],
        **{k: v for k, v in {"lengthscale_prior": GammaPrior(
            kernel_hparams.get('lengthscale_prior_concentration'),
            kernel_hparams.get('lengthscale_prior_rate')) if (kernel_hparams.get(
            'lengthscale_prior_concentration') is not None and kernel_hparams.get(
            'lengthscale_prior_rate') is not None) else None}.items() if
           v is not None}
    )

    covar_module = ScaleKernel(
        base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction,
                                         latent_train=X_train.double(), instruction_train=Y_scores),
        **{k: v for k, v in {"outputscale_prior": GammaPrior(
            kernel_hparams.get('outputscale_prior_concentration'),
            kernel_hparams.get('outputscale_prior_rate')) if (kernel_hparams.get(
            'outputscale_prior_concentration') is not None and kernel_hparams.get(
            'outputscale_prior_rate') is not None) else None}.items() if
           v is not None}
    )
    gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module,
                            **{k: v for k, v in {"mean_module": ConstantMean(
                                constant_prior=NormalPrior(
                                    kernel_hparams.get('mean_prior_mean'),
                                    kernel_hparams.get('mean_prior_std')
                                )
                            ) if (kernel_hparams.get('mean_prior_mean') is not None and
                                  kernel_hparams.get('mean_prior_std') is not None) else None}.items() if
                               v is not None})

    if kernel_hparams.get("mean", None) is not None:
        # Set to the constant value and don't optimize
        gp_model.mean_module.constant = kernel_hparams["mean"]
        gp_model.mean_module.constant.requires_grad_(False)
    if kernel_hparams.get("lengthscale", None) is not None:
        # Set to the constant value and don't optimize
        gp_model.covar_module.base_kernel.base_latent_kernel.lengthscale = kernel_hparams["lengthscale"]
        gp_model.covar_module.base_kernel.base_latent_kernel.raw_lengthscale.requires_grad_(False)
        gp_model.covar_module.base_kernel.instruction_kernel.lengthscale = kernel_hparams["lengthscale"]
        gp_model.covar_module.base_kernel.instruction_kernel.raw_lengthscale.requires_grad_(False)
    if kernel_hparams.get("outputscale", None) is not None:
        # Set to the constant value and don't optimize
        gp_model.covar_module.outputscale = kernel_hparams["outputscale"]
        gp_model.covar_module.raw_outputscale.requires_grad_(False)

    requires_optim = False
    for name, param in gp_model.named_parameters():
        if param.requires_grad:
            requires_optim = True
            # print(f"Requires optim: {(name, param)}")
            break

    gp_mll = None
    if requires_optim:
        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    return gp_model, gp_mll, requires_optim