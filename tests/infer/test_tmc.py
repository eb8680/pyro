import logging
import math
import os

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import config_enumerate
from pyro.infer.importance import vectorized_importance_weights
from pyro.infer.tmc import TensorMonteCarlo
from pyro.infer.traceenum_elbo import TraceEnum_ELBO
from tests.common import assert_equal, skipif_param


logger = logging.getLogger(__name__)


def _skip_cuda(*args):
    return skipif_param(*args,
                        condition="CUDA_TEST" in os.environ,
                        reason="https://github.com/uber/pyro/issues/1380")


@pytest.mark.parametrize("num_samples", [None, 500])
@pytest.mark.parametrize("max_plate_nesting", [0, 1])
def test_tmc_categoricals(max_plate_nesting, num_samples):
    pyro.clear_param_store()
    q1 = pyro.param("q1", torch.tensor([0.4, 0.6], requires_grad=True))
    q2 = pyro.param("q2", torch.tensor([[0.4, 0.3, 0.3],
                                        [0.25, 0.7, 0.05]],
                                       requires_grad=True))
    q3 = pyro.param("q3", torch.tensor([[0.4, 0.3, 0.2, 0.1],
                                        [0.2, 0.1, 0.6, 0.1],
                                        [0.15, 0.25, 0.5, 0.1]],
                                       requires_grad=True))
    q4 = pyro.param("q4", torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True))

    def model():
        x1 = pyro.sample("x1", dist.Categorical(pyro.param("q1")))
        x2 = pyro.sample("x2", dist.Categorical(pyro.param("q2")[..., x1, :]))
        x3 = pyro.sample("x3", dist.Categorical(pyro.param("q3")[..., x2, :]))
        pyro.sample("x4", dist.Bernoulli(pyro.param("q4")[..., x3]), obs=torch.tensor(float(1)))

    elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
    enum_model = config_enumerate(model, default="parallel", expand=False, num_samples=None)
    expected_loss = elbo.differentiable_loss(enum_model, lambda: None)
    expected_grads = grad(expected_loss, (q1, q2, q3, q4))

    tmc = TensorMonteCarlo(max_plate_nesting=max_plate_nesting)
    tmc_model = config_enumerate(model, default="parallel", expand=False, num_samples=num_samples)
    actual_loss = tmc.differentiable_loss(tmc_model, lambda: None)
    actual_grads = grad(actual_loss, (q1, q2, q3, q4))

    # TODO increase this precision, suspiciously weak
    assert_equal(actual_loss, expected_loss, prec=0.02, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_equal(actual_grad, expected_grad, prec=0.02, msg="".join([
            "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
            "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("max_plate_nesting", [1])
def test_tmc_normals(num_samples, max_plate_nesting):
    pyro.clear_param_store()
    q1 = pyro.param("q1", torch.tensor(0.5, requires_grad=True))

    def model():
        x1 = pyro.sample("x1", dist.Normal(pyro.param("q1"), 1.))
        x2 = pyro.sample("x2", dist.Normal(x1, 1.))
        x3 = pyro.sample("x3", dist.Normal(x2, 1.))
        pyro.sample("x4", dist.Normal(x3, 1.), obs=torch.tensor(float(1)))

    guide = poutine.block(model, hide_fn=lambda msg: msg["type"] == "sample" and msg["is_observed"])
    vectorized_log_weights, _, _ = vectorized_importance_weights(
        model, guide,
        max_plate_nesting=max_plate_nesting,
        num_samples=num_samples ** 3)
    assert vectorized_log_weights.shape == (num_samples ** 3,)
    expected_loss = -(vectorized_log_weights.logsumexp(dim=-1) - math.log(float(num_samples) ** 3))
    expected_grads = grad(expected_loss, (q1,))

    tmc = TensorMonteCarlo(max_plate_nesting=max_plate_nesting)
    tmc_model = config_enumerate(model, default="parallel", expand=False, num_samples=num_samples)
    actual_loss = tmc.differentiable_loss(tmc_model, lambda: None)
    actual_grads = grad(actual_loss, (q1,))

    # TODO increase this precision, suspiciously weak
    assert_equal(actual_loss, expected_loss, prec=0.1, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))

    # TODO increase this precision, suspiciously weak
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_equal(actual_grad, expected_grad, prec=0.05, msg="".join([
            "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
            "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
        ]))
