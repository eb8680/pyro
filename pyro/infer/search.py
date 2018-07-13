from __future__ import absolute_import, division, print_function

import warnings
from six.moves import xrange

import torch
from torch.distributions.utils import broadcast_all

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.enum import config_enumerate, iter_discrete_traces
from pyro.infer.util import Dice, is_validation_enabled
from pyro.util import check_model_guide_match, check_site_shape, check_traceenum_requirements, warn_if_nan

from .abstract_infer import TracePosterior


class Search(TracePosterior):
    """
    Partially vectorized dynamic programming inference:
    Do forward pass in parallel, then recursively split combined trace sequentially
    """

    def __init__(self,
                 model,
                 max_iarange_nesting=float('inf'),
                 default="parallel",
                 **kwargs):

        self.model = model
        self.max_iarange_nesting = max_iarange_nesting
        self.default = default
        super(Search, self).__init__(**kwargs)

    def _traces(self, *args, **kwargs):
        enum_model = poutine.enum(config_enumerate(self.model, default=self.default),
                                  first_available_dim=self.max_iarange_nesting)
        for enum_tr in iter_discrete_traces("flat", enum_model, *args, **kwargs):
            enum_tr.compute_log_prob()  # cache log_prob values
            for tr in self._split_trace(enum_tr):
                yield tr, tr.log_prob_sum()
            else:
                yield enum_tr, enum_tr.log_prob_sum()

    def _split_trace(self, enum_trace):

        def predicate(site):
            return site["type"] == "sample" and not site["is_observed"] and \
                site["infer"]["enumerate"] == "parallel"

        for site in filter(predicate, enum_trace.nodes.values()):
            # find site batch dimension
            # split along batch dimension
            # split rest of trace along that dimension
            # yield fully split
            name = site["name"]
            if len(site["value"].shape) > 0 and site["value"].shape[0] != 1:
                for i in xrange(site["value"].shape[0]):
                    tc = enum_trace.copy()
                    tc.nodes[name]["value"] = tc.nodes[name]["value"][i, ...]
                    for tcs in self._split_trace(tc):
                        print("split", name, tc.nodes[name]["value"].shape)
                        yield tcs
            else:
                print(name, enum_trace.nodes[name]["value"].shape)
                continue
