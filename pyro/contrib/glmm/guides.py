from __future__ import absolute_import, division, print_function

import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.util import tensor_to_dict, rmv, rvv, rtril
from pyro.ops.linalg import rinverse


class LinearModelGuide(nn.Module):

    def __init__(self, d, w_sizes, tikhonov_init=-2., scale_tril_init=3.):
        """
        Guide for linear models. No amortisation happens over designs.
        Amortisation over data is taken care of by analytic formulae for
        linear models (heavy use of truth).

        :param int d: the number of designs
        :param dict w_sizes: map from variable string names to int.
        :param float tikhonov_init: initial value for `tikhonov_diag` parameter.
        :param float scale_tril_init: initial value for `scale_tril` parameter.
        """
        super(LinearModelGuide, self).__init__()
        # Represent each parameter group as independent Gaussian
        # Making a weak mean-field assumption
        # To avoid this- combine labels
        self.tikhonov_diag = nn.Parameter(
                tikhonov_init*torch.ones(sum(w_sizes.values())))
        self.scale_tril = {l: nn.Parameter(
                scale_tril_init*torch.ones(d, p, p)) for l, p in w_sizes.items()}
        # This registers the dict values in pytorch
        # Await new version to use nn.ParamterDict
        self._registered = nn.ParameterList(self.scale_tril.values())
        self.w_sizes = w_sizes
        self.softplus = nn.Softplus()

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        return self.linear_model_formula(y, design, target_labels)

    def linear_model_formula(self, y, design, target_labels):

        tikhonov_diag = torch.diag(self.softplus(self.tikhonov_diag))
        xtx = torch.matmul(design.transpose(-1, -2), design) + tikhonov_diag
        xtxi = rinverse(xtx, sym=True)
        mu = rmv(xtxi, rmv(design.transpose(-1, -2), y))

        # Extract sub-indices
        mu = tensor_to_dict(self.w_sizes, mu, subset=target_labels)
        scale_tril = {l: rtril(self.scale_tril[l]) for l in target_labels}

        return mu, scale_tril

    def forward(self, y_dict, design, observation_labels, target_labels):

        pyro.module("ba_guide", self)

        # Returns two dicts from labels -> tensors
        mu, scale_tril = self.get_params(y_dict, design, target_labels)

        for l in target_labels:
            w_dist = dist.MultivariateNormal(mu[l], scale_tril=scale_tril[l])
            pyro.sample(l, w_dist)


class SigmoidGuide(LinearModelGuide):

    def __init__(self, d, n, w_sizes, slope, scale_tril_init=.1, mu_init=0., **kwargs):
        super(SigmoidGuide, self).__init__(d, w_sizes, scale_tril_init=scale_tril_init,
                                           **kwargs)
        self.inverse_sigmoid_scale = 1./slope

        self.scale_tril0 = {l: nn.Parameter(
                scale_tril_init*torch.ones(d, p, p)) for l, p in w_sizes.items()}
        self._registered0 = nn.ParameterList(self.scale_tril0.values())

        self.scale_tril1 = {l: nn.Parameter(
                scale_tril_init*torch.ones(d, p, p)) for l, p in w_sizes.items()}
        self._registered1 = nn.ParameterList(self.scale_tril1.values())

        self.mu0 = {l: nn.Parameter(
                mu_init*torch.ones(d, p)) for l, p in w_sizes.items()}
        self._registered_mu0 = nn.ParameterList(self.mu0.values())

        self.mu1 = {l: nn.Parameter(
                mu_init*torch.ones(d, p)) for l, p in w_sizes.items()}
        self._registered_mu0 = nn.ParameterList(self.mu1.values())

    def get_params(self, y_dict, design, target_labels):

        print('0')
        print(self.mu0, self.scale_tril0)
        print('1')
        print(self.mu1, self.scale_tril1)
        print('interval')
        print(self.scale_tril)

        # For values in (0, 1), we can perfectly invert the transformation
        y = torch.cat(list(y_dict.values()), dim=-1)
        y, y1m = y.clamp(1e-35, 1), (1.-y).clamp(1e-35, 1)
        logited = y.log() - y1m.log()
        y_trans = logited * self.inverse_sigmoid_scale

        mu, scale_tril = self.linear_model_formula(y_trans, design, target_labels)
        scale_tril = {l: scale_tril[l].expand(mu[l].shape + (mu[l].shape[-1], )) for l in scale_tril}

        # Now deal with clipping- values equal to 0 or 1
        mask0 = (y < 1e-35).squeeze(-1)
        mask1 = (1.-y < 1e-35).squeeze(-1)
        print(mask0.numel())
        print(mask0.nonzero().size(0))
        print(mask1.nonzero().size(0))
        print(mask0.sum(0))
        print(mask1.sum(0))
        for l in mu.keys():
            mu[l][mask0, :] = self.mu0[l].expand(mu[l].shape)[mask0, :]
            mu[l][mask1, :] = self.mu1[l].expand(mu[l].shape)[mask1, :]
            scale_tril[l][mask0, :, :] = rtril(self.scale_tril0[l].expand(scale_tril[l].shape))[mask0, :, :]
            scale_tril[l][mask1, :, :] = rtril(self.scale_tril1[l].expand(scale_tril[l].shape))[mask1, :, :]

        return mu, scale_tril


class NormalInverseGammaGuide(LinearModelGuide):

    def __init__(self, d, w_sizes, mf=False, tau_label="tau", alpha_init=100.,
                 b0_init=100., **kwargs):
        super(NormalInverseGammaGuide, self).__init__(d, w_sizes, **kwargs)
        self.alpha = nn.Parameter(alpha_init*torch.ones(d))
        self.b0 = nn.Parameter(b0_init*torch.ones(d))
        self.mf = mf
        self.tau_label = tau_label

    def get_params(self, y_dict, design, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)

        coefficient_labels = [label for label in target_labels if label != self.tau_label]
        mu, scale_tril = self.linear_model_formula(y, design, coefficient_labels)
        mu_vec = torch.cat(list(mu.values()), dim=-1)

        yty = rvv(y, y)
        ytxmu = rvv(y, rmv(design, mu_vec))
        beta = self.b0 + .5*(yty - ytxmu)

        return mu, scale_tril, self.alpha, beta

    def forward(self, y_dict, design, observation_labels, target_labels):

        pyro.module("ba_guide", self)

        mu, scale_tril, alpha, beta = self.get_params(y_dict, design, target_labels)

        if self.tau_label in target_labels:
            tau_dist = dist.Gamma(alpha, beta)
            tau = pyro.sample(self.tau_label, tau_dist)
            obs_sd = 1./tau.sqrt().unsqueeze(-1).unsqueeze(-1)

        for label in target_labels:
            if label != self.tau_label:
                if self.mf:
                    w_dist = dist.MultivariateNormal(mu[label],
                                                     scale_tril=scale_tril[label])
                else:
                    w_dist = dist.MultivariateNormal(mu[label],
                                                     scale_tril=scale_tril[label]*obs_sd)
                pyro.sample(label, w_dist)


class GuideDV(nn.Module):
    """A Donsker-Varadhan `T` family based on a guide family via
    the relation `T = log p(theta) - log q(theta | y, d)`
    """
    def __init__(self, guide):
        super(GuideDV, self).__init__()
        self.guide = guide

    def forward(self, design, trace, observation_labels, target_labels):

        trace.compute_log_prob()
        prior_lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        conditional_guide = pyro.condition(self.guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
                y_dict, design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        posterior_lp = sum(cond_trace.nodes[l]["log_prob"] for l in target_labels)

        return posterior_lp - prior_lp
