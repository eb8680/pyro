from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer.search import Search
# from pyro.infer import EmpiricalMarginal

# from tests.common import assert_equal


@pytest.mark.parametrize("enumerate1,num_steps", [
    ("sequential", 2),
    # ("sequential", 3),
    ("parallel", 2),
    # ("parallel", 3),
    # ("parallel", 10),
    # ("parallel", 20),
])
def test_hmm_model(enumerate1, num_steps):

    pyro.clear_param_store()
    data = torch.ones(num_steps)
    init_probs = torch.tensor([0.5, 0.5])

    def model(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                                      constraint=constraints.simplex)
        locs = pyro.param("obs_locs", torch.tensor([-1.0, 1.0]))
        scale = pyro.param("obs_scale", torch.tensor(1.0),
                           constraint=constraints.positive)

        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))
            pyro.sample("y_{}".format(i), dist.Normal(locs[x], scale), obs=y)

    posterior = Search(model, default=enumerate1, max_iarange_nesting=0).run(data)
    for tr in posterior.exec_traces:
        print(tr.nodes, tr.log_prob_sum())
