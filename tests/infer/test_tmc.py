import logging
import math
import os

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.testing import fakes
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


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("num_samples", [None, 500])
@pytest.mark.parametrize("max_plate_nesting", [0, 1])
def test_tmc_categoricals(depth, max_plate_nesting, num_samples):
    pyro.clear_param_store()
    qs = [pyro.param("q0", torch.tensor([0.4, 0.6], requires_grad=True))]
    for i in range(1, depth):
        qs.append(pyro.param(
            "q{}".format(i),
            torch.randn(2, 2).abs().detach().requires_grad_()
        ))
    qs.append(pyro.param("qy", torch.tensor([0.75, 0.25], requires_grad=True)))

    def model():
        x = pyro.sample("x0", dist.Categorical(pyro.param("q0")))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i),
                            dist.Categorical(pyro.param("q{}".format(i))[..., x, :]))
        pyro.sample("y", dist.Bernoulli(pyro.param("qy")[..., x]),
                    obs=torch.tensor(float(1)))

    elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
    enum_model = config_enumerate(model, default="parallel", expand=False, num_samples=None)
    expected_loss = elbo.differentiable_loss(enum_model, lambda: None)
    expected_grads = grad(expected_loss, qs)

    tmc = TensorMonteCarlo(max_plate_nesting=max_plate_nesting)
    tmc_model = config_enumerate(model, default="parallel", expand=False, num_samples=num_samples)
    actual_loss = tmc.differentiable_loss(tmc_model, lambda: None)
    actual_grads = grad(actual_loss, qs)

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


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("num_samples,expand", [(500, True), (500, False)])
@pytest.mark.parametrize("max_plate_nesting", [0])
@pytest.mark.parametrize("reparameterized", [True, False])
@pytest.mark.parametrize("guide_type", ["prior", "factorized", "nonfactorized"])
def test_tmc_normals_chain_iwae(depth, num_samples, max_plate_nesting,
                                reparameterized, guide_type, expand):
    # compare iwae and tmc
    pyro.clear_param_store()

    q1 = pyro.param("q1", torch.tensor(0.5, requires_grad=True))
    q2 = pyro.param("q2", torch.tensor(0.4, requires_grad=True))
    qs = (q2.unconstrained(),) if guide_type == "prior" else (q1.unconstrained(), q2.unconstrained())

    def model(reparameterized):
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
        x = pyro.sample("x0", Normal(pyro.param("q2"), 1.))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i), Normal(x, 1.))
        pyro.sample("y", Normal(x, 1.), obs=torch.tensor(float(1)))

    def factorized_guide(reparameterized):
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
        pyro.sample("x0", Normal(pyro.param("q1"), 1. / depth))
        for i in range(1, depth):
            pyro.sample("x{}".format(i), Normal(0., float(i+1 / depth)))

    def nonfactorized_guide(reparameterized):
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
        x = pyro.sample("x0", Normal(pyro.param("q1"), 1. / depth))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i), Normal(x, 1. / depth))

    guide = factorized_guide if guide_type == "factorized" else \
        nonfactorized_guide if guide_type == "nonfactorized" else \
        poutine.block(model, hide_fn=lambda msg: msg["type"] == "sample" and msg["is_observed"])
    flat_num_samples = num_samples ** min(depth, 2)  # don't use too many, expensive
    vectorized_log_weights, _, _ = vectorized_importance_weights(
        model, guide, True,
        max_plate_nesting=max_plate_nesting,
        num_samples=flat_num_samples)
    assert vectorized_log_weights.shape == (flat_num_samples,)
    expected_loss = -(vectorized_log_weights.logsumexp(dim=-1) - math.log(float(flat_num_samples)))
    expected_grads = grad(expected_loss, qs)

    tmc = TensorMonteCarlo(max_plate_nesting=max_plate_nesting)
    tmc_model = config_enumerate(
        model, default="parallel", expand=expand, num_samples=num_samples)
    tmc_guide = config_enumerate(
        guide, default="parallel", expand=expand, num_samples=num_samples)
    actual_loss = tmc.differentiable_loss(tmc_model, tmc_guide, reparameterized)
    actual_grads = grad(actual_loss, qs)

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


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("num_samples,expand", [(500, True), (1000, False)])
@pytest.mark.parametrize("max_plate_nesting", [0, 1])
@pytest.mark.parametrize("guide_type", ["prior", "factorized", "nonfactorized"])
def test_tmc_normals_chain_gradient(depth, num_samples, max_plate_nesting, expand, guide_type):
    # compare reparameterized and nonreparameterized gradient estimates
    pyro.clear_param_store()

    q1 = pyro.param("q1", torch.tensor(0.5, requires_grad=True))
    q2 = pyro.param("q2", torch.tensor(0.4, requires_grad=True))
    qs = (q2.unconstrained(),) if guide_type == "prior" else (q1.unconstrained(), q2.unconstrained())

    def model(reparameterized):
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
        x = pyro.sample("x0", Normal(pyro.param("q2"), 1.))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i), Normal(x, 1.))
        pyro.sample("y", Normal(x, 1.), obs=torch.tensor(float(1)))

    def factorized_guide(reparameterized):
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
        pyro.sample("x0", Normal(pyro.param("q1"), 1. / depth))
        for i in range(1, depth):
            pyro.sample("x{}".format(i), Normal(0., float(i+1 / depth)))

    def nonfactorized_guide(reparameterized):
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
        x = pyro.sample("x0", Normal(pyro.param("q1"), 1. / depth))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i), Normal(x, 1. / depth))

    tmc = TensorMonteCarlo(max_plate_nesting=max_plate_nesting)
    tmc_model = config_enumerate(
        model, default="parallel", expand=expand, num_samples=num_samples)
    guide = factorized_guide if guide_type == "factorized" else \
        nonfactorized_guide if guide_type == "nonfactorized" else \
        lambda *args: None
    tmc_guide = config_enumerate(
        guide, default="parallel", expand=expand, num_samples=num_samples)

    expected_loss = tmc.differentiable_loss(tmc_model, tmc_guide, True)
    expected_grads = grad(expected_loss, qs)

    actual_loss = tmc.differentiable_loss(tmc_model, tmc_guide, False)
    actual_grads = grad(actual_loss, qs)

    # TODO increase this precision, suspiciously weak
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_equal(actual_grad, expected_grad, prec=0.05, msg="".join([
            "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
            "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
        ]))
