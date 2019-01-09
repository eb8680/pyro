from __future__ import absolute_import, division, print_function

import argparse
from collections import namedtuple
import time
import torch
import pytest
import numpy as np
from functools import partial

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
    LinearModelGuide, NormalInverseGammaGuide, SigmoidGuide, GuideDV, LogisticGuide,
    LogisticResponseEst, LogisticLikelihoodEst, SigmoidResponseEst, SigmoidLikelihoodEst,
    NormalResponseEst, NormalLikelihoodEst
)

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
basic_2p_linear_model_sds_10_2pt5, basic_2p_guide = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]))
_, basic_2p_guide_w1 = zero_mean_unit_obs_sd_lm(torch.tensor([10., 2.5]), coef_label="w1")
basic_2p_linear_model_sds_10_0pt1, _ = zero_mean_unit_obs_sd_lm(torch.tensor([10., .1]))
basic_2p_ba_guide = lambda d: LinearModelGuide(d, {"w": 2})  # noqa: E731
normal_response_est = lambda d: NormalResponseEst(d, {"y": 10})
normal_response_est_20 = lambda d: NormalResponseEst(d, {"y": 20})
normal_likelihood_est = lambda d: NormalLikelihoodEst(d, {"ab": 2, "re": 10}, {"y": 10})
normal_likelihood_est2 = lambda d: NormalLikelihoodEst(d, {"w": 2}, {"y": 10})
group_2p_linear_model_sds_10_2pt5 = group_linear_model(torch.tensor(0.), torch.tensor([10.]), torch.tensor(0.),
                                                       torch.tensor([2.5]), torch.tensor(1.))
normal_re = group_linear_model(torch.tensor(0.), torch.tensor([10., .1]), torch.tensor(0.),
                               torch.ones(10), torch.tensor(1.), coef1_label="ab", coef2_label="re")
group_2p_guide = group_normal_guide(torch.tensor(1.), (1,), (1,))
group_2p_ba_guide = lambda d: LinearModelGuide(d, {"w1": 1, "w2": 1})  # noqa: E731
nig_2p_linear_model_3_2 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([.1, 10.]),
                                                            torch.tensor([3.]), torch.tensor([2.]))
nig_2p_linear_model_15_14 = normal_inverse_gamma_linear_model(torch.tensor(0.), torch.tensor([.1, 10.]),
                                                              torch.tensor([15.]), torch.tensor([14.]))
nig_re_3_2 = normal_inverse_gamma_linear_model([torch.tensor(0.), torch.tensor(0.)],
                                               [torch.tensor([.1, 10.]), torch.ones(10)],
                                               torch.tensor([3.]), torch.tensor([2.]),
                                               coef_labels=["ab", "re"])

re_guide = lambda d: LinearModelGuide(d, {"ab": 2, "re": 10})
nig_2p_guide = normal_inverse_gamma_guide((2,), mf=True)
nig_2p_ba_guide = lambda d: NormalInverseGammaGuide(d, {"w": 2})  # noqa: E731
nig_2p_ba_mf_guide = lambda d: NormalInverseGammaGuide(d, {"w": 2}, mf=True)  # noqa: E731

sigmoid_2p_model = sigmoid_model_fixed(torch.tensor([1., 10.]), torch.tensor([.25, 8.]),
                                       torch.tensor(2.))
sigmoid_re_model = sigmoid_model_fixed([torch.tensor([1.]), torch.tensor([10.])],
                                                  [torch.tensor([.25]), torch.tensor([8.])],
                                                  torch.tensor(2.), coef_labels=["coef", "loc"])
loc_2p_model = known_covariance_linear_model(torch.tensor([1., 10.]), torch.tensor([.25, 8.]), torch.tensor(2.),
                                             coef_label="w1")
logistic_2p_model = logistic_regression_model(torch.tensor([1., 10.]), torch.tensor([.25, 8.]), coef_labels="w1")
logistic_random_effects = logistic_regression_model([torch.tensor([1.]), torch.tensor([10.])],
                                                    [torch.tensor([.25]), torch.tensor([8.])],
                                                    coef_labels=["coef", "loc"])
loc_ba_guide = lambda d: LinearModelGuide(d, {"w1": 2})  # noqa: E731
logistic_guide  = lambda d: LogisticGuide(d, {"w1": 2})
logistic_random_effect_guide = lambda d: LogisticGuide(d, {"loc": 1})
logistic_response_est = lambda d: LogisticResponseEst(d, ["y"])
logistic_likelihood_est = lambda d: LogisticLikelihoodEst(d, {"coef": 1, "loc": 1}, ["y"])
sigmoid_response_est = lambda d: SigmoidResponseEst(d, ["y"])
sigmoid_likelihood_est = lambda d: SigmoidLikelihoodEst(d, {"coef": 1, "loc": 1}, ["y"])
sigmoid_low_guide = lambda d: SigmoidGuide(d, {"w": 2})  # noqa: E731
sigmoid_high_guide = lambda d: SigmoidGuide(d, {"w": 2})  # noqa: E731
sigmoid_random_effect_guide = lambda d: SigmoidGuide(d, {"coef": 1, "loc": 1})

########################################################################################
# Aux
########################################################################################

elbo = TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss

NREPS = 2

def zerofn(*args, **kwargs):
    return torch.tensor(0.)

zerofn.name = 'zero'

ba_lm_use_ae = partial(ba_eig_lm, analytic_entropy=True)
ba_lm_use_ae.__name__ = 'ba_lm_use_ae'

# Makes the plots look pretty
vi_eig_lm.name = "Variational inference"
vi_ape.name = "Variational inference"
ba_eig_lm.name = "Posterior"
ba_lm_use_ae.name = "Posterior, with analytic entropy"
ba_eig_mc.name = "Posterior"
barber_agakov_ape.name = "Posterior"
donsker_varadhan_eig.name = "Donsker-Varadhan"
linear_model_ground_truth.name = "Ground truth"
normal_inverse_gamma_ground_truth.name = "Ground truth"
naive_rainforth_eig.name = "Nested Monte Carlo"
accelerated_rainforth_eig.name = "Accelerated Rainforth"
gibbs_y_eig.name = "Marginal"
gibbs_y_re_eig.name = "Marginal with random effects"


U = namedtuple("CheckConvergenceExample", [
    "title",
    "model",
    "design",
    "observation_label",
    "target_label",
    "est1",
    "est2",
    "kwargs1",
    "kwargs2"
])

CONV_TEST_CASES = [
    U(
        "Logistic regression",
        logistic_2p_model,
        loc_4d_1n_2p,
        "y", "w1",
        barber_agakov_ape,
        None,
        {"num_steps": 800, "num_samples": 40, "optim": optim.Adam({"lr": 0.05}),
         "guide": logistic_guide((4,)), "final_num_samples": 1000},
        {}
    ),
    U(
        "High slope sigmoid -- should see major difference",
        sigmoid_2p_model,
        loc_4d_1n_2p,
        "y", "w1",
        barber_agakov_ape,
        linear_model_ground_truth,
        {"num_steps": 800, "num_samples": 40, "optim": optim.Adam({"lr": 0.05}),
         "guide": sigmoid_high_guide((4,)), "final_num_samples": 1000},
        {"eig": False}
    ),
    U(
        "Barber-Agakov on A/B test with unknown covariance",
        nig_2p_linear_model_3_2,
        AB_test_2d_10n_2p,
        "y",
        ["w", "tau"],
        barber_agakov_ape,
        None,
        {"num_steps": 800, "num_samples": 20, "optim": optim.Adam({"lr": 0.05}),
         "guide": nig_2p_ba_guide((2,)), "final_num_samples": 1000},
        {}
    ),
    U(
        "Barber-Agakov on A/B test with unknown covariance (mean-field guide)",
        nig_2p_linear_model_3_2,
        AB_test_2d_10n_2p,
        "y",
        ["w", "tau"],
        barber_agakov_ape,
        None,
        {"num_steps": 800, "num_samples": 20, "optim": optim.Adam({"lr": 0.05}),
         "guide": nig_2p_ba_mf_guide((2,)), "final_num_samples": 1000},
        {}
    ),
    U(
        "Barber-Agakov on circle",
        basic_2p_linear_model_sds_10_2pt5,
        X_circle_5d_1n_2p,
        "y",
        "w",
        barber_agakov_ape,
        linear_model_ground_truth,
        {"num_steps": 400, "num_samples": 10, "optim": optim.Adam({"lr": 0.05}),
         "guide": basic_2p_ba_guide((5,)), "final_num_samples": 1000},
        {"eig": False}
    ),
    U(
        "Barber-Agakov on small AB test",
        basic_2p_linear_model_sds_10_2pt5,
        AB_test_2d_10n_2p,
        "y",
        "w",
        barber_agakov_ape,
        linear_model_ground_truth,
        {"num_steps": 400, "num_samples": 10, "optim": optim.Adam({"lr": 0.05}),
         "guide": basic_2p_ba_guide((2,)), "final_num_samples": 1000},
        {"eig": False}
    ),
    U(
        "Donsker-Varadhan on small AB test",
        basic_2p_linear_model_sds_10_2pt5,
        AB_test_2d_10n_2p,
        "y",
        "w",
        donsker_varadhan_eig,
        linear_model_ground_truth,
        {"num_steps": 400, "num_samples": 100, "optim": optim.Adam({"lr": 0.05}),
         "T": GuideDV(basic_2p_ba_guide((2,))), "final_num_samples": 10000},
        {}
    ),
    U(
        "Donsker-Varadhan on circle",
        basic_2p_linear_model_sds_10_2pt5,
        X_circle_5d_1n_2p,
        "y",
        "w",
        donsker_varadhan_eig,
        linear_model_ground_truth,
        {"num_steps": 400, "num_samples": 400, "optim": optim.Adam({"lr": 0.05}),
         "T": GuideDV(basic_2p_ba_guide((5,))), "final_num_samples": 10000},
        {}
    ),
]


@pytest.mark.parametrize("title,model,design,observation_label,target_label,est1,est2,kwargs1,kwargs2", CONV_TEST_CASES)
def test_convergence(title, model, design, observation_label, target_label, est1, est2, kwargs1, kwargs2):
    """
    Produces a convergence plot for a Barber-Agakov or Donsker-Varadhan
    EIG estimation.
    """
    t = time.time()
    pyro.clear_param_store()
    if est2 is not None:
        truth = est2(model, design, observation_label, target_label, **kwargs2)
    else:
        truth = None
    dv, final = est1(model, design, observation_label, target_label, return_history=True, **kwargs1)
    x = np.arange(0, dv.shape[0])
    print(est1.__name__)
    if truth is not None:
        print("Final est", final, "Truth", truth, "Error", (final - truth).abs().sum())
    else:
        print("Final est", final)
    print("Time", time.time() - t)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(x, dv.detach().numpy())

        if truth is not None:
            for true, col in zip(torch.unbind(truth, 0), plt.rcParams['axes.prop_cycle'].by_key()['color']):
                plt.axhline(true.numpy(), color=col)

        plt.title(title)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design")
    parser.add_argument("--num-runs", nargs="?", default=1, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=5, type=int)
    parser.add_argument("--num-participants", nargs="?", default=10, type=int)
    parser.add_argument("--num-questions", nargs="?", default=5, type=int)
    args = parser.parse_args()
    main(args.num_runs, args.num_parallel, args.num_participants, args.num_questions)