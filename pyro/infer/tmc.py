import math
import queue
import warnings

import torch

import pyro.poutine as poutine

from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace, iter_discrete_escape, iter_discrete_extend
from pyro.infer.util import is_validation_enabled
from pyro.ops import packed
from pyro.ops.contract import einsum
from pyro.poutine.enumerate_messenger import EnumerateMessenger
from pyro.util import check_traceenum_requirements, warn_if_nan


def _compute_dice_factors(model_trace, guide_trace):
    log_probs = []
    for role, trace in zip(("model", "guide"), (model_trace, guide_trace)):
        for name, site in trace.nodes.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            if role == "model" and name in guide_trace:
                continue

            log_prob = site["packed"]["score_parts"].score_function  # not scaled by subsampling
            dims = getattr(log_prob, "_pyro_dims", "")
            if not isinstance(log_prob, torch.Tensor):
                log_prob = torch.tensor(float(log_prob), device=site["value"].device)

            if site["infer"].get("enumerate") == "parallel":
                num_samples = site["infer"].get("num_samples")
                if num_samples is not None:  # site was multiply sampled
                    if not site["fn"].has_rsample and not is_identically_zero(log_prob):
                        log_prob = log_prob - log_prob.detach()
                    else:
                        log_prob = torch.zeros_like(log_prob)
                    log_prob = log_prob - math.log(num_samples)
                    log_prob._pyro_dims = dims
                    # I don't know why the following broadcast is needed, but it makes tests pass:
                    log_prob, _ = packed.broadcast_all(log_prob, site["packed"]["log_prob"])
                    log_probs.append(log_prob)
            elif site["infer"].get("enumerate") == "sequential":
                num_samples = site["infer"].get("num_samples", site["infer"]["_enum_total"])
                # XXX not correct for plates?
                log_denom = torch.tensor(math.log(num_samples),
                                         device=site["value"].device)
                log_denom._pyro_dims = dims
                log_probs.append(log_denom)
            else:  # site was singly monte carlo sampled
                if is_identically_zero(log_prob):
                    continue
                log_prob = log_prob - log_prob.detach()
                log_prob._pyro_dims = dims
                log_probs.append(log_prob)

    return log_probs


def _compute_tmc_factors(model_trace, guide_trace):
    # factors
    log_factors = []
    for name, site in guide_trace.nodes.items():
        if site["type"] != "sample" or site["is_observed"]:
            continue
        log_factors.append(packed.neg(site["packed"]["log_prob"]))
    for name, site in model_trace.nodes.items():
        if site["type"] != "sample":
            continue
        if site["name"] not in guide_trace and \
                site["infer"].get("enumerate", None) == "parallel" and \
                site["infer"].get("num_samples", -1) > 0:
            # site was sampled from the prior, include log_prob proposal term
            log_factors.append(packed.neg(site["packed"]["log_prob"]))
        log_factors.append(site["packed"]["log_prob"])

    return log_factors


def _compute_tmc_estimate(model_trace, guide_trace):

    # factors
    log_factors = _compute_tmc_factors(model_trace, guide_trace)
    log_factors += _compute_dice_factors(model_trace, guide_trace)

    # loss
    eqn = ",".join([f._pyro_dims for f in log_factors]) + "->"
    plates = "".join(frozenset().union(list(model_trace.plate_to_symbol.values()),
                                       list(guide_trace.plate_to_symbol.values())))
    tmc, = einsum(eqn, *log_factors, plates=plates,
                  backend="pyro.ops.einsum.torch_log",
                  modulo_total=False)
    return tmc


class TensorMonteCarlo(ELBO):

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, *args, **kwargs)

        if is_validation_enabled():
            check_traceenum_requirements(model_trace, guide_trace)

            has_enumerated_sites = any(site["infer"].get("enumerate")
                                       for trace in (guide_trace, model_trace)
                                       for name, site in trace.nodes.items()
                                       if site["type"] == "sample")

            if self.strict_enumeration_warning and not has_enumerated_sites:
                warnings.warn('Found no sample sites configured for enumeration. '
                              'If you want to enumerate sites, you need to @config_enumerate or set '
                              'infer={"enumerate": "sequential"} or infer={"enumerate": "parallel"}? '
                              'If you do not want to enumerate, consider using Trace_ELBO instead.')

        model_trace.compute_score_parts()
        guide_trace.pack_tensors()
        model_trace.pack_tensors(guide_trace.plate_to_symbol)
        return model_trace, guide_trace

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.max_plate_nesting == float('inf'):
            self._guess_max_plate_nesting(model, guide, *args, **kwargs)
        if self.vectorize_particles:
            guide = self._vectorized_num_particles(guide)
            model = self._vectorized_num_particles(model)

        # Enable parallel enumeration over the vectorized guide and model.
        # The model allocates enumeration dimensions after (to the left of) the guide,
        # accomplished by preserving the _ENUM_ALLOCATOR state after the guide call.
        guide_enum = EnumerateMessenger(first_available_dim=-1 - self.max_plate_nesting)
        model_enum = EnumerateMessenger()  # preserve _ENUM_ALLOCATOR state
        guide = guide_enum(guide)
        model = model_enum(model)

        q = queue.LifoQueue()
        guide = poutine.queue(guide, q,
                              escape_fn=iter_discrete_escape,
                              extend_fn=iter_discrete_extend)
        for i in range(1 if self.vectorize_particles else self.num_particles):
            q.put(poutine.Trace())
            while not q.empty():
                yield self._get_trace(model, guide, *args, **kwargs)

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        :returns: a differentiable estimate of the marginal log-likelihood
        :rtype: torch.Tensor
        :raises ValueError: if the ELBO is not differentiable (e.g. is
            identically zero)

        Computes a differentiable TMC estimate using ``num_particles`` many samples
        (particles).  The result should be infinitely differentiable (as long
        as underlying derivatives have been implemented).
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = _compute_tmc_estimate(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo = elbo + elbo_particle
        elbo = elbo / self.num_particles

        if not torch.is_tensor(elbo) or not elbo.requires_grad:
            raise ValueError('ELBO is cannot be differentiated: {}'.format(elbo))

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss
