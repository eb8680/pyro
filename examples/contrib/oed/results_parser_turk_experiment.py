from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pyro.contrib.util import rmv

output_dir = "./run_outputs/"
COLOURS = [[1, .6, 0], [1, .4, .4], [.5, .5, 1.], [1., .5, .5]]
VALUE_LABELS = {"Entropy": "Posterior entropy on fixed effects",
                "L2 distance": "Expected L2 distance from posterior to truth",
                "Optimized EIG": "Maximized EIG",
                "EIG gap": "Difference between maximum and mean EIG"}


def upper_lower(array):
    centre = array.mean(1)
    upper, lower = np.percentile(array, 95, axis=1), np.percentile(array, 5, axis=1)
    return lower, centre, upper


def rlogdet(M):
    old_shape = M.shape[:-2]
    tbound = M.view(-1, M.shape[-2], M.shape[-1])
    ubound = torch.unbind(tbound, dim=0)
    logdets = [torch.logdet(m) for m in ubound]
    bound = torch.stack(logdets)
    return bound.view(old_shape)


def rtrace(M):
    old_shape = M.shape[:-2]
    tbound = M.view(-1, M.shape[-2], M.shape[-1])
    ubound = torch.unbind(tbound, dim=0)
    traces = [torch.trace(m) for m in ubound]
    bound = torch.stack(traces)
    return bound.view(old_shape)


def main(fname, findex, plot):
    make_mean = torch.cat([torch.cat([(1./3)*torch.ones(3, 3), torch.zeros(3, 3)], dim=0),
                           torch.cat([torch.zeros(3, 3), (1./3)*torch.ones(3, 3)], dim=0)], dim=1)

    if not fname:
        results_fnames = sorted(glob.glob(output_dir+"*.result_stream.pickle"))
        fname = results_fnames[findex]

    results_dict = defaultdict(lambda : defaultdict(list))
    with open(fname, 'rb') as results_file:
        try:
            while True:
                results = pickle.load(results_file)
                # Compute entropy and L2 distance to the true fixed effects
                st = results["model_fixed_effect_scale_tril"]
                covm = torch.matmul(st, st.transpose(-1, -2))
                entropy = .5*rlogdet(2*np.pi*np.e*covm).squeeze()
                centered_fixed_effects = results["model_fixed_effect_mean"] - rmv(make_mean, results["model_fixed_effect_mean"])
                trace = rtrace(covm).squeeze()
                l2d = torch.norm(centered_fixed_effects - results["true_fixed_effects"], dim=-1).squeeze()
                el2d = torch.sqrt(l2d**2 + trace)
                output = {
                    "Entropy": entropy,
                    "L2 distance": el2d
                }
                if 'estimation_surface' in results:
                    eig_star, _ = torch.max(results['estimation_surface'], dim=1)
                    eig_star = eig_star.squeeze()
                    eig_mean = results['estimation_surface'].mean(1).squeeze()
                    output['Optimized EIG'] = eig_star
                    output["EIG gap"] = eig_star - eig_mean
                # TODO deal with incorrect order of stream
                results_dict[results['typ']][results['run']].append(output)
        except EOFError:
            pass

    # Get results into better format
    # First, concat across runs
    reformed = {statistic: {
                    k: torch.stack([torch.cat([v[run][i][statistic] for run in v]) for i in range(len(v[1]))])
                    for k, v in results_dict.items() if statistic in v[1][0]}
                for statistic in results_dict['oed'][1][0].keys()}
    descript = OrderedDict([(statistic,
                    OrderedDict([(k, upper_lower(v.detach().numpy())) for k, v in sorted(sts.items())]))
                for statistic, sts in sorted(reformed.items())])

    if plot:
        for k, r in descript.items():
            value_label = VALUE_LABELS[k]
            plt.figure(figsize=(10, 5))
            for i, (lower, centre, upper) in enumerate(r.values()):
                x = np.arange(0, centre.shape[0])
                plt.plot(x, centre, linestyle='-', markersize=6, color=COLOURS[i], marker='o')
                plt.fill_between(x, upper, lower, color=COLOURS[i]+[.2])
            # plt.title(value_label, fontsize=18)
            plt.legend(r.keys(), loc=1, fontsize=16)
            plt.xlabel("Step", fontsize=18)
            plt.ylabel(value_label, fontsize=18)
            plt.xticks(fontsize=14)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.yticks(fontsize=14)
            plt.show()
    else:
        print(descript)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design results parser")
    parser.add_argument("--fname", nargs="?", default="", type=str)
    parser.add_argument("--findex", nargs="?", default=-1, type=int)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.fname, args.findex, args.plot)
