from __future__ import absolute_import, division, print_function

import argparse
from collections import namedtuple
import time
import torch
import pickle
import numpy as np
import datetime

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import (
    vi_ape, naive_rainforth_eig, accelerated_rainforth_eig, donsker_varadhan_eig, gibbs_y_eig,
    gibbs_y_re_eig, amortized_lfire_eig, lfire_eig, iwae_eig, laplace_vi_ape
)
from pyro.contrib.util import lexpand
from pyro.contrib.oed.util import (
    linear_model_ground_truth, vi_eig_lm, vi_eig_mc, ba_eig_lm, ba_eig_mc, normal_inverse_gamma_ground_truth,
    laplace_vi_eig_mc, 
)
from pyro.contrib.glmm import (
    group_assignment_matrix, normal_inverse_gamma_linear_model, sigmoid_model_fixed, known_covariance_linear_model,
    logistic_regression_model
)
from pyro.contrib.glmm.guides import (
    LinearModelPosteriorGuide, NormalInverseGammaPosteriorGuide, SigmoidPosteriorGuide, GuideDV, LogisticPosteriorGuide,
    LogisticMarginalGuide, LogisticLikelihoodGuide, SigmoidMarginalGuide, SigmoidLikelihoodGuide,
    NormalMarginalGuide, NormalLikelihoodGuide,
    LinearModelLaplaceGuide,
)
from pyro.contrib.glmm.classifiers import LinearModelAmortizedClassifier, LinearModelBootstrapClassifier, LinearModelClassifier

"""
Expected information gain estimation benchmarking
-------------------------------------------------
Dials to turn:
- the model
- the design space
- which parameters to consider as targets

Models:
- linear model
- normal-inverse gamma model
- linear mixed effects
- logistic regression
- sigmoid regression

Designs:
- A/B test
- location finding

Estimation techniques:
    Core
    - analytic EIG (where available)
    - Nested Monte Carlo
    - Posterior
    - Marginal / marginal + likelihood

    Old / deprecated
    - iterated variational inference
    - Donsker-Varadhan

    TODO
    - Laplace approximation
    - LFIRE

"""

#########################################################################################
# Designs
#########################################################################################
# All design tensors have shape: batch x n x p
# AB test
AB_test_11d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in range(0, 11)])
AB_test_11d_20n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 20-n])) for n in np.linspace(0, 20, 11)])
AB_test_2d_10n_2p = torch.stack([group_assignment_matrix(torch.tensor([n, 10-n])) for n in [0, 5]])
AB_test_11d_10n_12p = torch.cat([AB_test_11d_10n_2p, lexpand(torch.eye(10), 11)], dim=-1)

# Design on S^1
item_thetas = torch.linspace(0., np.pi, 10).unsqueeze(-1)
X_circle_10d_1n_2p = torch.stack([item_thetas.cos(), -item_thetas.sin()], dim=-1)
item_thetas_small = torch.linspace(0., np.pi/2, 5).unsqueeze(-1)
X_circle_5d_1n_2p = torch.stack([item_thetas_small.cos(), -item_thetas_small.sin()], dim=-1)

# Location finding designs
loc_15d_1n_2p = torch.stack([torch.linspace(-30., 30., 15), torch.ones(15)], dim=-1).unsqueeze(-2)
loc_4d_1n_2p = torch.tensor([[-5., 1], [-4.9, 1.], [4.9, 1], [5., 1.]]).unsqueeze(-2)

########################################################################################
# Aux
########################################################################################

elbo = TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss

Estimator = namedtuple("EIGEstimator",[
    "name",
    "tags",
    "method"
])

truth_lm = Estimator("Ground truth", ["truth", "lm", "standard"], linear_model_ground_truth)
truth_nigam = Estimator("Ground truth", ["truth", "nigam", "standard"], normal_inverse_gamma_ground_truth)
nmc = Estimator("Nested Monte Carlo", ["nmc", "naive_rainforth", "standard"], naive_rainforth_eig)
nnmc = Estimator("Non-nested Monte Carlo", ["nnmc", "accelerated_rainforth", "standard"], accelerated_rainforth_eig)
posterior_lm = Estimator("Posterior", ["posterior", "gibbs", "ba", "lm", "standard"], ba_eig_lm)
posterior_mc = Estimator("Posterior", ["posterior", "gibbs", "ba", "standard"], ba_eig_mc)
marginal = Estimator("Marginal", ["marginal", "gibbs", "standard"], gibbs_y_eig)
marginal_re = Estimator("Marginal + likelihood", ["marginal_re", "marginal_likelihood", "gibbs", "standard"],
                        gibbs_y_re_eig)
alfire = Estimator("Amortized LFIRE", ["alfire"], amortized_lfire_eig)
lfire = Estimator("LFIRE", ["lfire"], lfire_eig)
iwae = Estimator("IWAE", ["iwae"], iwae_eig)
laplace = Estimator("Laplace", ["laplace", "diag_laplace"], laplace_vi_eig_mc)

Case = namedtuple("EIGBenchmarkingCase", [
    "title",
    "model",
    "design",
    "observation_label",
    "target_label",
    "estimator_argslist",
    "tags"
])

CASES = [
    #############################################################################################################
    # Normal inverse gamma, with AB testing
    #############################################################################################################
    Case(
        "Normal inverse gamma model, information on w, tau",
        (normal_inverse_gamma_linear_model, {"coef_means": torch.tensor(0.),
                                             "coef_sqrtlambdas": torch.tensor([.1, .55]),
                                             "alpha": torch.tensor(3.),
                                             "beta": torch.tensor(2.)}),
        AB_test_11d_10n_2p,
        "y",
        ["w", "tau"],
        [
            (nmc, {"N": 120*120, "M": 120}),
            (posterior_mc,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "guide": (NormalInverseGammaPosteriorGuide, {"mf": True, "correct_gamma": False, "alpha_init": 3.,
                                                           "b0_init": 2., "tikhonov_init": -2.,
                                                           "scale_tril_init": torch.tensor([[10., 0.], [0., 1/.55]])}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (iwae,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500, "M": 1,
              "guide": (NormalInverseGammaPosteriorGuide, {"mf": True, "correct_gamma": False, "alpha_init": 3.,
                                                           "b0_init": 2., "tikhonov_init": -2.,
                                                           "scale_tril_init": torch.tensor([[10., 0.], [0., 1/.55]])}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal,
             {"num_samples": 10, "num_steps": 1500, "final_num_samples": 500,
              "guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (lfire,
             {"num_theta_samples": 20, "num_y_samples": 2, "num_steps": 500, "final_num_samples": 100,
              "classifier": (LinearModelClassifier, {"scale_tril_init": 1/3., "ntheta": 20}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_nigam, {}),
        ],
        ["nigam", "ground_truth", "no_re", "ab_test", "explicit_grid"]
    ),
    Case(
        "Normal inverse gamma model, information on w only",
        (normal_inverse_gamma_linear_model, {"coef_means": torch.tensor(0.),
                                             "coef_sqrtlambdas": torch.tensor([.1, 10.]),
                                             "alpha": torch.tensor(3.),
                                             "beta": torch.tensor(2.)}),
        AB_test_11d_10n_2p,
        "y",
        "w",
        [
            # Caution, Rainforth does not work correctly in this case because we must
            # compute p(psi | theta)
            # TODO: Use LFIRE instead
            (posterior_mc,
             {"num_samples": 10, "num_steps": 1000, "final_num_samples": 500,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal_re,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "marginal_guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "cond_guide": (NormalLikelihoodGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_lm, {}),
        ],
        ["nigam", "ground_truth", "re", "ab_test"]
    ),
    #############################################################################################################
    # Linear model, with AB testing
    #############################################################################################################
    Case(
        "Linear regression model",
        (known_covariance_linear_model, {"coef_means": torch.tensor(0.),
                                         "coef_sds": torch.tensor([10., .1]),
                                         "observation_sd": torch.tensor(1.)}),
        AB_test_11d_10n_2p,
        "y",
        "w",
        [
            (nmc, {"N": 60*60, "M": 60}),
            (posterior_lm,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            # TODO: fix analytic entropy
            # (Estimator("Posterior with analytic entropy", ["posterior", "gibbs", "ba", "ae"], ba_eig_lm),
            #  {"num_samples": 10, "num_steps": 1200, "final_num_samples": 50, "analytic_entropy": True,
            #   "guide": (LinearModelGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
            #   "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (iwae,
             {"num_samples": 10, "num_steps": 800, "final_num_samples": 500, "M": 1,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            # (marginal_re,
            #  {"num_samples": 10, "num_steps": 600, "final_num_samples": 500,
            #   "marginal_guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
            #   "cond_guide": (NormalLikelihoodGuide, {"mu_init": 0., "sigma_init": 3.}),
            #   "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (alfire,
             {"num_samples": 10, "num_steps": 1500, "final_num_samples": 500,
              "classifier": (LinearModelAmortizedClassifier, {"scale_tril_init": 1/3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (Estimator("ALFIRE 2", ["alfire2"], amortized_lfire_eig),
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "classifier": (LinearModelBootstrapClassifier, {"scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (lfire,
             {"num_theta_samples": 60, "num_y_samples": 2, "num_steps": 1200, "final_num_samples": 100,
              "classifier": (LinearModelClassifier, {"scale_tril_init": 1/3., "ntheta": 60}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_lm, {}),
            (laplace,
             {"num_samples": 1,
              "num_steps": 1000, 
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}}),
              "loss": TraceEnum_ELBO(max_iarange_nesting=2).differentiable_loss,
              "guide": (LinearModelLaplaceGuide, {}),
              "final_num_samples": 10}),
        ],
        ["lm", "ground_truth", "no_re", "ab_test", "small_n", "lmab"]
    ),
    Case(
        "Linear model with random effects",
        (known_covariance_linear_model, {"coef_means": [torch.tensor(0.), torch.tensor(0.)],
                                         "coef_sds": [torch.tensor([10., .1]), torch.ones(10)],
                                         "observation_sd": torch.tensor(1.),
                                         "coef_labels": ["ab", "re"]}),
        AB_test_11d_10n_12p,
        "y",
        "ab",
        [
            (nmc, {"N": 50, "M": 50, "M_prime": 50, "independent_priors": True}),
            (posterior_lm,
             {"num_samples": 10, "num_steps": 150, "final_num_samples": 500,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal_re,
             {"num_samples": 10, "num_steps": 600, "final_num_samples": 500,
              "marginal_guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "cond_guide": (NormalLikelihoodGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_lm, {})
        ],
        ["lm", "re", "ab_test", "small_n"]
    ),

    Case(
        "Linear regression model (large dim(y))",
        (known_covariance_linear_model, {"coef_means": torch.tensor(0.),
                                         "coef_sds": torch.tensor([10., .1]),
                                         "observation_sd": torch.tensor(1.)}),
        AB_test_11d_20n_2p,
        "y",
        "w",
        [
            (nmc, {"N": 60*60, "M": 60}),
            (posterior_lm,
             {"num_samples": 10, "num_steps": 1000, "final_num_samples": 500,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (iwae,
             {"num_samples": 10, "num_steps": 800, "final_num_samples": 500, "M": 1,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal,
             {"num_samples": 10, "num_steps": 700, "final_num_samples": 500,
              "guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_lm, {})
        ],
        ["lm", "ground_truth", "no_re", "ab_test", "large_n", "large_dim_y"],
    ),
    #############################################################################################################
    # Sigmoid regression location finding
    #############################################################################################################
    Case(
        "Sigmoid regression model",
        (sigmoid_model_fixed, {"coef_means": torch.tensor([1., 10.]),
                               "coef_sds": torch.tensor([.25, 8.]),
                               "observation_sd": torch.tensor(2.)}),
        loc_15d_1n_2p,
        "y",
        "w",
        [
            (nmc, {"N": 80*80, "M": 80}),
            (posterior_mc,
             {"num_samples": 10, "num_steps": 1500, "final_num_samples": 500,
              "guide": (SigmoidPosteriorGuide, {"mu_init": torch.tensor([1., 10.]),
                                                "scale_tril_init": torch.tensor([[1., 0.], [0., 20.]]),
                                                "tikhonov_init": -2.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            # TODO why is iwae performance here worse than NMC? Must be a badly chosen posterior
            # Yes look at the prior means!
            (iwae,
             {"num_samples": [5, 1], "num_steps": 1000, "final_num_samples": [20*20, 60],
              "guide": (SigmoidPosteriorGuide, {"mu_init": torch.tensor([1., 10.]),
                                                "scale_tril_init": torch.tensor([[1., 0.], [0., 20.]]),
                                                "tikhonov_init": -2.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal,
             {"num_samples": 10, "num_steps": 3000, "final_num_samples": 500,
              "guide": (SigmoidMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
        ],
        ["sigmoid", "no_re", "location", "explicit_grid"]
    ),
    Case(
        "Sigmoid with random effects",
        (sigmoid_model_fixed, {"coef_means": [torch.tensor([1.]), torch.tensor([10.])],
                               "coef_sds": [torch.tensor([.25]), torch.tensor([8.])],
                               "observation_sd": torch.tensor(2.),
                               "coef_labels": ["coef", "loc"]}),
        loc_15d_1n_2p,
        "y",
        "loc",
        [
            (nmc, {"N": 50, "M": 50, "M_prime": 50, "independent_priors": True}),
            (posterior_mc,
             {"num_samples": 10, "num_steps": 1500, "final_num_samples": 500,
              "guide": (SigmoidPosteriorGuide, {"mu_init": 0., "scale_tril_init": 20., "tikhonov_init": -2.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal_re,
             {"num_samples": 10, "num_steps": 2000, "final_num_samples": 500,
              "marginal_guide": (SigmoidMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "cond_guide": (SigmoidLikelihoodGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
        ],
        ["sigmoid", "re", "location"]
    ),
    #############################################################################################################
    # Logistic regression location finding
    #############################################################################################################
    Case(
        "Logistic regression",
        (logistic_regression_model, {"coef_means": torch.tensor([1., 10.]),
                                     "coef_sds": torch.tensor([.25, 8.])}),
        loc_15d_1n_2p,
        "y",
        "w",
        [
            (nnmc, {"N": 2000, "yspace": {"y": torch.tensor([0., 1.])}}),
            (posterior_mc,
             {"num_samples": 10, "num_steps": 800, "final_num_samples": 500,
              "guide": (LogisticPosteriorGuide, {"mu_init": 0.,
                                                 "scale_tril_init": torch.tensor([[1., 0.], [0., 20.]])}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}
             ),
            # Do not apply here- not a direct competitor to NNMC
            # (iwae,
            #  {"num_samples": 10, "num_steps": 800, "final_num_samples": 500, "M": 1,
            #   "guide": (LogisticPosteriorGuide, {"mu_init": 0.,
            #                                      "scale_tril_init": torch.tensor([[1., 0.], [0., 20.]])}),
            #   "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal,
             {"num_samples": 10, "num_steps": 500, "final_num_samples": 500,
              "guide": (LogisticMarginalGuide, {"p_logit_init": 0.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}
             )
        ],
        ["logistic", "no_re", "location"]
    ),
    Case(
        "Logistic with random effects",
        (logistic_regression_model, {"coef_means": [torch.tensor([1.]), torch.tensor([10.])],
                                     "coef_sds": [torch.tensor([.25]), torch.tensor([8.])],
                                     "coef_labels": ["coef", "loc"]}),
        loc_15d_1n_2p,
        "y",
        "loc",
        [
            (nnmc, {"N": 200, "M_prime": 200, "yspace": {"y": torch.tensor([0., 1.])}}),
            (posterior_mc,
             {"num_samples": 10, "num_steps": 800, "final_num_samples": 500,
              "guide": (LogisticPosteriorGuide, {"mu_init": 0., "scale_tril_init": 20.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}
             ),
            (marginal_re,
             {"num_samples": 10, "num_steps": 1000, "final_num_samples": 500,
              "marginal_guide": (LogisticMarginalGuide, {"p_logit_init": 0.}),
              "cond_guide": (LogisticLikelihoodGuide, {"p_logit_init": 0.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}
             )
        ],
        ["logistic", "re", "location"]
    ),
    #############################################################################################################
    # Linear models with circular designs
    #############################################################################################################
    Case(
        "Linear model with designs on S^1",
        (known_covariance_linear_model, {"coef_means": torch.tensor(0.),
                                         "coef_sds": torch.tensor([10., 2.]),
                                         "observation_sd": torch.tensor(1.)}),
        X_circle_10d_1n_2p,
        "y",
        "w",
        [
            (nmc, {"N": 60*60, "M": 60}),
            (posterior_lm,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (iwae,
             {"num_samples": 10, "num_steps": 800, "final_num_samples": 500, "M": 1,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_lm, {})
        ],
        ["lm", "ground_truth", "no_re", "circle", "small_n"]
    ),
    Case(
        "Linear model with designs on S^1, targetting one parameter only",
        (known_covariance_linear_model, {"coef_means": [torch.tensor(0.), torch.tensor(0.)],
                                         "coef_sds": [torch.tensor([10.]), torch.tensor([2.])],
                                         "observation_sd": torch.tensor(1.),
                                         "coef_labels": ["w1", "w2"]}),
        X_circle_10d_1n_2p,
        "y",
        "w1",
        [
            (nmc, {"N": 60*60, "M": 60, "M_prime": 60, "independent_priors": True}),
            (posterior_lm,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "guide": (LinearModelPosteriorGuide, {"tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal_re,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "marginal_guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "cond_guide": (NormalLikelihoodGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_lm, {})
        ],
        ["lm", "ground_truth", "re", "circle", "small_n"]
    ),
]


def main(case_tags, estimator_tags, num_runs, num_parallel, experiment_name):
    output_dir = "./run_outputs/eig_benchmark/"
    if not experiment_name:
        experiment_name = output_dir+"{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir+experiment_name
    results_file = experiment_name+'.result_stream.pickle'

    print("Experiment", experiment_name)
    case_tags = case_tags.split(",")
    estimator_tags = estimator_tags.split(",")
    if "*" in case_tags or "all" in case_tags:
        cases = CASES
    else:
        cases = [c for c in CASES if all(tag in c.tags for tag in case_tags)]
    for case in cases:
        # Create the model in a way that allows us to pickle its params
        model_func, model_params = case.model
        model = model_func(**model_params)
        for estimator, kwargs, *others in case.estimator_argslist:
            # Filter estimators
            if others:
                estimator_name = others[0]
            else:
                estimator_name = estimator.name
            if ("*" in estimator_tags) or ("all" in estimator_tags) or any(tag in estimator.tags for tag in estimator_tags):
                for run in range(1, num_runs+1):
                    pyro.clear_param_store()
                    print(case.title, "|", estimator_name)

                    # Handles the parallelization
                    if "truth" not in estimator.tags:
                        expanded_design = lexpand(case.design, num_parallel)
                    else:
                        expanded_design = lexpand(case.design, 1)

                    # Begin collecting settings of this run, for pickle
                    results = {
                        "case": case.title,
                        "run_num": run,
                        "obs_labels": case.observation_label,
                        "target_labels": case.target_label,
                        "model_params": model_params,
                        "model_name": model_func.__name__,
                        "design": case.design,
                        "num_parallel": num_parallel,
                        "estimator_name": estimator_name,
                        "estimator_params": {},
                    }

                    for key, value in list(kwargs.items()):
                        if isinstance(value, tuple):
                            param_func, param_params = value
                            # Communicate some size attributes to the guide
                            if "guide" in key or "classifier" in key:
                                param_params.update({"d": expanded_design.shape[:2],
                                                     "w_sizes": model.w_sizes,
                                                     "y_sizes": {model.observation_label: expanded_design.shape[-2]}})
                            results["estimator_params"][key] = param_params
                            kwargs[key] = param_func(**param_params)
                        else:
                            results["estimator_params"][key] = value

                    t = time.time()
                    eig_surface = estimator.method(model, expanded_design,
                                                   case.observation_label,
                                                   case.target_label, **kwargs)
                    elapsed = time.time() - t
                    print("Finished in", elapsed, "seconds")

                    results["surface"] = eig_surface
                    results["elapsed"] = elapsed
                    with open(results_file, 'ab') as f:
                        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIG estimation benchmarking")
    # Note: for case-tags, we take the intersection of matching cases; for estimator-tags we take the union of matching
    # estimators
    # In both cases, blank = all
    # This may seem weird, but it corresponds best to common usage
    parser.add_argument("--case-tags", nargs="?", default="*", type=str)
    parser.add_argument("--estimator-tags", nargs="?", default="*", type=str)
    parser.add_argument("--num-runs", nargs="?", default=1, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=5, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    args = parser.parse_args()
    main(args.case_tags, args.estimator_tags, args.num_runs, args.num_parallel, args.name)
