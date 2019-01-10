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
    vi_ape, naive_rainforth_eig, accelerated_rainforth_eig, donsker_varadhan_eig, barber_agakov_ape, gibbs_y_eig,
    gibbs_y_re_eig
)
from pyro.contrib.util import lexpand
from pyro.contrib.oed.util import (
    linear_model_ground_truth, vi_eig_lm, ba_eig_lm, ba_eig_mc, normal_inverse_gamma_ground_truth
)
from pyro.contrib.glmm import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix,
    normal_inverse_gamma_linear_model, normal_inverse_gamma_guide, group_linear_model,
    group_normal_guide, sigmoid_model_gamma, sigmoid_model_fixed, rf_group_assignments,
    known_covariance_linear_model, logistic_regression_model
)
from pyro.contrib.glmm.guides import (
    LinearModelPosteriorGuide, NormalInverseGammaPosteriorGuide, SigmoidPosteriorGuide, GuideDV, LogisticPosteriorGuide,
    LogisticMarginalGuide, LogisticLikelihoodGuide, SigmoidMarginalGuide, SigmoidLikelihoodGuide,
    NormalMarginalGuide, NormalLikelihoodGuide
)

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

#########################################################################################
# Models
#########################################################################################
# Linear models
# basic_2p_linear_model_sds_10_2pt5, basic_2p_guide = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]))
# _, basic_2p_guide_w1 = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]), coef_label="w1")
# basic_2p_linear_model_sds_10_0pt1, _ = zero_mean_unit_obs_sd_lm(torch.tensor([10., .1]))
# basic_2p_ba_guide = lambda d: LinearModelGuide(d, {"w": 2})  # noqa: E731
# normal_response_est = lambda d: NormalResponseEst(d, {"y": 10})
# normal_response_est_20 = lambda d: NormalResponseEst(d, {"y": 20})
# normal_likelihood_est = lambda d: NormalLikelihoodEst(d, {"ab": 2, "re": 10}, {"y": 10})
# normal_likelihood_est2 = lambda d: NormalLikelihoodEst(d, {"w": 2}, {"y": 10})
# group_2p_linear_model_sds_10_2pt5 = group_linear_model(torch.tensor(0.), torch.tensor([10.]), torch.tensor(0.),
#                                                        torch.tensor([2.5]), torch.tensor(1.))
# normal_re = group_linear_model(torch.tensor(0.), torch.tensor([10., .1]), torch.tensor(0.),
#                                torch.ones(10), torch.tensor(1.), coef1_label="ab", coef2_label="re")
# group_2p_guide = group_normal_guide(torch.tensor(1.), (1,), (1,))
# group_2p_ba_guide = lambda d: LinearModelGuide(d, {"w1": 1, "w2": 1})  # noqa: E731
# nig_2p_linear_model_3_2 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([.1, 10.]),
#                                                             torch.tensor([3.]), torch.tensor([2.]))
# nig_2p_linear_model_15_14 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([.1, 10.]),
#                                                               torch.tensor([15.]), torch.tensor([14.]))
# nig_re_3_2 = normal_inverse_gamma_linear_model([torch.tensor(0.), torch.tensor(0.)],
#                                                [torch.tensor([.1, 10.]), torch.ones(10)],
#                                                torch.tensor([3.]), torch.tensor([2.]),
#                                                coef_labels=["ab", "re"])
#
# re_guide = lambda d: LinearModelGuide(d, {"ab": 2, "re": 10})
# nig_2p_guide = normal_inverse_gamma_guide((2,), mf=True)
# nig_2p_ba_guide = lambda d: NormalInverseGammaGuide(d, {"w": 2})  # noqa: E731
# nig_2p_ba_mf_guide = lambda d: NormalInverseGammaGuide(d, {"w": 2}, mf=True)  # noqa: E731
#
# sigmoid_2p_model = sigmoid_model_fixed(torch.tensor([1., 10.]), torch.tensor([.25, 8.]),
#                                        torch.tensor(2.))
# sigmoid_re_model = sigmoid_model_fixed([torch.tensor([1.]), torch.tensor([10.])],
#                                                   [torch.tensor([.25]), torch.tensor([8.])],
#                                                   torch.tensor(2.), coef_labels=["coef", "loc"])
# loc_2p_model = known_covariance_linear_model(torch.tensor([1., 10.]), torch.tensor([.25, 8.]), torch.tensor(2.),
#                                              coef_label="w1")
# logistic_2p_model = logistic_regression_model(torch.tensor([1., 10.]), torch.tensor([.25, 8.]), coef_labels="w1")
# logistic_random_effects = logistic_regression_model([torch.tensor([1.]), torch.tensor([10.])],
#                                                     [torch.tensor([.25]), torch.tensor([8.])],
#                                                     coef_labels=["coef", "loc"])
# loc_ba_guide = lambda d: LinearModelGuide(d, {"w1": 2})  # noqa: E731
# logistic_guide  = lambda d: LogisticGuide(d, {"w1": 2})
# logistic_random_effect_guide = lambda d: LogisticGuide(d, {"loc": 1})
# logistic_response_est = lambda d: LogisticResponseEst(d, ["y"])
# logistic_likelihood_est = lambda d: LogisticLikelihoodEst(d, {"coef": 1, "loc": 1}, ["y"])
# sigmoid_response_est = lambda d: SigmoidResponseEst(d, ["y"])
# sigmoid_likelihood_est = lambda d: SigmoidLikelihoodEst(d, {"coef": 1, "loc": 1}, ["y"])
# sigmoid_low_guide = lambda d: SigmoidGuide(d, {"w": 2})  # noqa: E731
# sigmoid_high_guide = lambda d: SigmoidGuide(d, {"w": 2})  # noqa: E731
# sigmoid_random_effect_guide = lambda d: SigmoidGuide(d, {"coef": 1, "loc": 1})

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
    Case(
        "Normal inverse gamma model, information on w, tau",
        (normal_inverse_gamma_linear_model, {"coef_means": torch.tensor(0.),
                                             "coef_sqrtlambdas": torch.tensor([.1, 10.]),
                                             "alpha": torch.tensor(3.),
                                             "beta": torch.tensor(2.)}),
        AB_test_11d_10n_2p,
        "y",
        ["w", "tau"],
        [
            (nmc, {"N": 60*60, "M": 60}),
            (posterior_mc,
             {"num_samples": 10, "num_steps": 800, "final_num_samples": 500,
              "guide": (NormalInverseGammaPosteriorGuide, {"mf": True, "alpha_init": 10., "b0_init": 10.,
                                                           "tikhonov_init": -2., "scale_tril_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_nigam, {}),
        ],
        ["nigam", "ground_truth", "no_re", "ab_test"]
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
             {"num_samples": 10, "num_steps": 800, "final_num_samples": 500,
              "guide": (NormalInverseGammaPosteriorGuide, {"mf": True, "alpha_init": 10., "b0_init": 10.,
                                                           "tikhonov_init": -2., "scale_tril_init": 3.}),
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
            (marginal,
             {"num_samples": 10, "num_steps": 1200, "final_num_samples": 500,
              "guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_lm, {})
        ],
        ["lm", "ground_truth", "no_re", "ab_test", "small_n"]
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
            (marginal,
             {"num_samples": 10, "num_steps": 700, "final_num_samples": 500,
              "guide": (NormalMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (truth_lm, {})
        ],
        ["lm", "ground_truth", "no_re", "ab_test", "large_n", "large_dim_y"],
    ),
    Case(
        "Sigmoid regression model",
        (sigmoid_model_fixed, {"coef_means": torch.tensor([1., 10.]),
                               "coef_sds": torch.tensor([.25, 8.]),
                               "observation_sd": torch.tensor(2.)}),
        loc_15d_1n_2p,
        "y",
        "w",
        [
            (nmc, {"N": 50, "M": 50}),
            (posterior_mc,
             {"num_samples": 10, "num_steps": 1500, "final_num_samples": 500,
              "guide": (SigmoidPosteriorGuide, {"mu_init": 0., "scale_tril_init": torch.tensor([[1., 0.], [0., 20.]]),
                                                "tikhonov_init": -2.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
            (marginal,
             {"num_samples": 10, "num_steps": 2000, "final_num_samples": 500,
              "guide": (SigmoidMarginalGuide, {"mu_init": 0., "sigma_init": 3.}),
              "optim": (optim.Adam, {"optim_args": {"lr": 0.05}})}),
        ],
        ["sigmoid", "no_re", "location"]
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
#
# CMP_TEST_CASES = [
#     T(
#         "Sigmoid regression model",
#         sigmoid_2p_model,
#         loc_15d_1n_2p,
#         "y",
#         "w",
#         [
#             (naive_rainforth_eig, [75*75, 75]),
#             (gibbs_y_eig,
#              [20, 2800, sigmoid_response_est((10, 15)), optim.Adam({"lr": 0.05}),
#               False, None, 500]),
#             (ba_eig_mc,
#              [10, 180, sigmoid_high_guide((10, 15)), optim.Adam({"lr": 0.05}),
#               False, None, 500]),
#             #(donsker_varadhan_eig,
#             # [400, 80, GuideDV(sigmoid_high_guide(15)),
#             #  optim.Adam({"lr": 0.05}), False, None, 500])
#         ]
#     ),
#     T(
#         "Normal inverse gamma version of A/B test, with random effects",
#         nig_re_3_2,
#         AB_test_11d_10n_12p,
#         "y",
#         "ab",
#         [
#             (gibbs_y_re_eig,
#              [10, 1100, normal_response_est((10, 11)), normal_likelihood_est((10, 11)),
#               optim.Adam({"lr": 0.05}), False, None, 500]),
#             (naive_rainforth_eig, [50*50, 50, 50, True]),
#             (ba_eig_mc,
#              [20, 600, re_guide((10, 11)), optim.Adam({"lr": 0.05}),
#               False, None, 500])
#         ]
#     ),
#     T(
#         "Sigmoid with random effects",
#         sigmoid_re_model,
#         loc_15d_1n_2p,
#         "y",
#         "loc",
#         [
#             (gibbs_y_re_eig,
#              [40, 1600, sigmoid_response_est((10, 15)), sigmoid_likelihood_est((10, 15)),
#               optim.Adam({"lr": 0.05}), False, None, 500]),
#             (ba_eig_mc,
#              [10, 400, sigmoid_random_effect_guide((10, 15)), optim.Adam({"lr": 0.05}),
#               False, None, 500]),
#             (naive_rainforth_eig, [60*60, 60, 60, True])
#         ]
#     ),
#     T(
#         "A/B test linear model with known observation variance",
#         basic_2p_linear_model_sds_10_2pt5,
#         AB_test_11d_10n_2p,
#         "y",
#         "w",
#         [
#             (linear_model_ground_truth, []),
#             (naive_rainforth_eig, [2000, 2000]),
#             (vi_eig_lm,
#              [{"guide": basic_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
#                "num_steps": 1000}, {"num_samples": 1}]),
#             (donsker_varadhan_eig,
#              [400, 800, GuideDV(basic_2p_ba_guide((11,))),
#               optim.Adam({"lr": 0.025}), False, None, 500]),
#             (ba_eig_lm,
#              [20, 400, basic_2p_ba_guide((11,)), optim.Adam({"lr": 0.05}),
#               False, None, 500])
#         ]
#     ),
#     # TODO: make this example work better
#     T(
#         "A/B testing with unknown covariance (Gamma(15, 14))",
#         nig_2p_linear_model_15_14,
#         AB_test_11d_10n_2p,
#         "y",
#         ["w", "tau"],
#         [
#             (naive_rainforth_eig, [2000, 2000]),
#             (vi_ape,
#              [{"guide": nig_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
#                "num_steps": 1000}, {"num_samples": 4}]),
#             (barber_agakov_ape,
#              [20, 800, nig_2p_ba_guide((11,)), optim.Adam({"lr": 0.05}),
#               False, None, 500]),
#             (barber_agakov_ape,
#              [20, 800, nig_2p_ba_mf_guide((11,)), optim.Adam({"lr": 0.05}),
#               False, None, 500])
#         ]
#     ),
#     # TODO: make this example work better
#     T(
#         "A/B testing with unknown covariance (Gamma(3, 2))",
#         nig_2p_linear_model_3_2,
#         AB_test_11d_10n_2p,
#         "y",
#         ["w", "tau"],
#         [
#             (naive_rainforth_eig, [2000, 2000]),
#             (vi_ape,
#              [{"guide": nig_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
#                "num_steps": 1000}, {"num_samples": 4}]),
#             (barber_agakov_ape,
#              [20, 800, nig_2p_ba_guide((11,)), optim.Adam({"lr": 0.05}),
#               False, None, 500]),
#             (barber_agakov_ape,
#              [20, 800, nig_2p_ba_mf_guide((11,)), optim.Adam({"lr": 0.05}),
#               False, None, 500])
#         ]
#     ),
#     # TODO: make VI work here (non-mean-field guide)
#     T(
#         "Linear model targeting one parameter",
#         group_2p_linear_model_sds_10_2pt5,
#         X_circle_10d_1n_2p,
#         "y",
#         "w1",
#         [
#             (linear_model_ground_truth, []),
#             (naive_rainforth_eig, [200, 200, 200]),
#             (vi_eig_lm,
#              [{"guide": group_2p_guide, "optim": optim.Adam({"lr": 0.05}), "loss": elbo,
#                "num_steps": 1000}, {"num_samples": 1}]),
#             (donsker_varadhan_eig,
#              [400, 400, GuideDV(group_2p_ba_guide((10,))),
#               optim.Adam({"lr": 0.05}), False, None, 500]),
#             (ba_eig_lm,
#              [20, 400, group_2p_ba_guide((10,)), optim.Adam({"lr": 0.05}),
#               False, None, 500])
#         ]
#     ),

# ]
#
#
# @pytest.mark.parametrize("title,model,design,observation_label,target_label,arglist", TRUTH_TEST_CASES)
# def test_eig_ground_truth(title, model, design, observation_label, target_label, arglist):
#     """
#     Computes EIG estimates from a number of estimators and compares them with the ground truth, assumed
#     to be the last estimator. This is done 10 times and standard deviations and biases are estimated.
#     The results are plotted.
#     """
#     ep = 0.05
#     ys = []
#     means = []
#     sds = []
#     names = []
#     elapseds = []
#     markers = ['x', '+', 'o', 'D', 'v', '^']
#     print(title)
#     for n, (estimator, args) in enumerate(arglist):
#         y, elapsed = time_eig(estimator, model, lexpand(design, NREPS) if n<(len(arglist)-1) else lexpand(design,1), observation_label, target_label, args)
#         y = y.detach().numpy()
#         y[np.isinf(y)] = np.nan
#         ys.append(y)
#         means.append(np.nanmean(y, 0))
#         sds.append(2*np.nanstd(y, 0))
#         elapseds.append(elapsed)
#         names.append(estimator.name)
#
#     bias = [np.nanmean(m - means[-1]) for m in means]
#     variance = [np.nanmean(s - sds[-1]) for s in sds]
#     print('bias: ', bias, '2std: ', variance)
#
#     if PLOT:
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10, 5))
#         x = np.arange(0, means[0].shape[0])
#         for n, (y, s) in enumerate(zip(means[:-1], sds[:-1])):
#             plt.errorbar(x+n*ep, y-means[-1], yerr=s, linestyle='-', marker=markers[n], markersize=10)
#         plt.title(title, fontsize=18)
#         plt.legend(names, loc=2, fontsize=16)
#         plt.axhline(color='k')
#         plt.xlabel("Design", fontsize=18)
#         plt.ylabel("EIG estimation error", fontsize=18)
#         plt.show()


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
                            if "guide" in key:
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
