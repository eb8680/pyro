import sys
import numpy as np

import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.poutine.block_poutine import BlockMessenger
import pyro.infer
import pyro.optim
import pyro.distributions.torch as dist


class BetaBernoulliExample(object):
    def __init__(self):
        self.alpha0 = Variable(torch.Tensor([10.0]))
        self.beta0 = Variable(torch.Tensor([10.0]))
        self.data = Variable(torch.zeros(10, 1))
        self.data[0:6, 0].data = torch.ones(6)

    def setup(self):
        self.log_alpha_q_0 = Variable(torch.Tensor([np.log(15.0)]), requires_grad=True)
        self.log_beta_q_0 = Variable(torch.Tensor([np.log(15.0)]), requires_grad=True)

    def model(self):
        f = pyro.sample("latent_fairness", dist.beta.Beta(self.alpha0, self.beta0))
        with pyro.iarange("data_iarange"):
            pyro.observe("obs", dist.bernoulli.Bernoulli(f), self.data)

    def guide(self):
        log_alpha_q = pyro.param("log_alpha_q", self.log_alpha_q_0)
        log_beta_q = pyro.param("log_beta_q", self.log_beta_q_0)
        alpha_q, beta_q = torch.exp(log_alpha_q), torch.exp(log_beta_q)
        pyro.sample("latent_fairness", dist.beta.Beta(alpha_q, beta_q))


class InferenceWrapper(object):

    def __init__(self, model_guide_pair, lam_steps):

        self.model_guide_pair = model_guide_pair
        self.lam_steps = lam_steps

    def sample(self, *args, **kwargs):

        num_steps = pyro.sample("num_steps", dist.poisson.Poisson(self.lam_steps))

        optimizer = pyro.optim.Adam({"lr": .0008, "betas": (0.93, 0.999)})
        with BlockMessenger():
            self.model_guide_pair.setup()
            pyro.clear_param_store()
            svi = pyro.infer.SVI(self.model_guide_pair.model,
                                 self.model_guide_pair.guide,
                                 optimizer, loss="ELBO")
            for i in range(int(num_steps.data[0])):
                svi.step(*args, **kwargs)

            loss = svi.evaluate_loss(*args, **kwargs)
            # hack: penalize long inference runs
            return loss / np.exp(num_steps.data[0] / 500.0)

    __call__ = sample

    def log_prob(self, x):
        return Variable(torch.zeros(1))


def model(model_guide_pair):
    lam = Variable(torch.Tensor([10.0]))
    inference_problem = InferenceWrapper(model_guide_pair, lam)
    elbo = pyro.sample("infer", inference_problem)
    pyro.sample("inner_elbo", dist.bernoulli.Bernoulli(Variable(torch.Tensor([elbo]))),
                obs=Variable(torch.ones(1)))


def guide(model_guide_pair):
    lam = Variable(torch.Tensor([10.0]))
    inference_problem = InferenceWrapper(model_guide_pair, lam)
    elbo = pyro.sample("infer", inference_problem)
    return elbo


def main():
    model_guide_pair = BetaBernoulliExample()

    meta_infer = pyro.infer.Importance(model, guide, num_samples=10)

    print(meta_infer(model_guide_pair).nodes)


if __name__ == "__main__":
    main()
