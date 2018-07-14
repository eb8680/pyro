from __future__ import absolute_import, division, print_function

import logging

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.poutine as poutine
from pyro.poutine.enumerate_messenger import DependentEnumerateMessenger

from pyro.infer import config_enumerate
from tests.common import assert_equal

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("enumerate1,num_steps", [
    ("parallel", 2),
    ("parallel", 3),
    ("parallel", 10),
    ("parallel", 20),
    pytest.param("parallel", 30, marks=pytest.mark.skip(reason="extremely expensive")),
])
def test_dependent_hmm(enumerate1, num_steps):
    pyro.clear_param_store()
    data = torch.ones(num_steps)
    init_probs = torch.tensor([0.5, 0.5])

    @config_enumerate(default=enumerate1)
    def model(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                      constraint=constraints.simplex)
        emission_probs = pyro.param("emission_probs",
                                    torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                    constraint=constraints.simplex)

        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))
            pyro.sample("y_{}".format(i), dist.Categorical(emission_probs[x]), obs=y)

    tr = poutine.trace(
        DependentEnumerateMessenger(first_available_dim=0)(model)
    ).get_trace(data)

    i = 0
    for name, node in tr.nodes.items():
        if node["type"] == "sample":
            if not node["is_observed"]:
                assert node["value"].shape == (2,) + (1,) * i
                i += 1
