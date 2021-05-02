# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 18:00:32 2021

@author: Cosmo

Illustrates the washing out theorem.

The washing out theorem says that, for any Bayesians, if they perform an
experiment often enough and their priors in the true hypothesis are not zero,
then their posteriors in the true hypothesis will likely end up close to 1
(and therefore also close to each other). The plot_washing_out function plots
how Bayesians credences evolve, so lets you visualize washing out.
"""


import random
import itertools
import numpy as np
import matplotlib.pyplot as plt


def update(priors_matrix, likelihoods_matrix, outcome):
    """Returns an array of posteriors in a Bayesian experiment.

    Args:
        priors_matrix: n-by-k array, where column j is agent j's priors over
          the hypotheses.
        likelihoods_matrix: n-by-m array, where column j is the likelihoods
          of outcome j.
        outcome: The index of the actual outcome.

    Returns:
        n-by-k array, where column j is agent j's posteriors.
    """
    unnormalized = priors_matrix * likelihoods_matrix[:, [outcome]]
    return unnormalized / sum(unnormalized)


def random_update(priors_matrix, likelihoods_matrix, true_likelihoods):
    """Updates various agents' priors in a Bayesian experiment.

    Args:
        priors_matrix: n-by-k array, where column j is agent j's priors over
          the hypotheses.
        likelihoods_matrix: n-by-m array, where column j is the likelihoods
          of outcome j.
        true_likelihoods: Sequence of length m. The actual likelihoods. This
          may be different to any row in likelihoods_matrix.

    Returns:
        n-by-k array, where column j is agent j's posteriors.
    """
    num_outcomes = likelihoods_matrix.shape[1]
    [outcome] = random.choices(range(num_outcomes), true_likelihoods, k=1)
    return update(priors_matrix, likelihoods_matrix, outcome)


def washing_out(priors_matrix, likelihoods_matrix, true_likelihoods, reps):
    """Calculates trajectories of agents' credences in a repeated experiment.

    Args:
        priors_matrix: n-by-k array, where column j is agent j's priors over
          the hypotheses.
        likelihoods_matrix: n-by-m array, where column j is the likelihoods
          of outcome j.
        true_likelihoods: Sequence of length m. The actual likelihoods. This
          may be different to any row in likelihoods_matrix.
        reps: Number of times to repeat the experiment.

    Returns:
        (n,k,reps)-array. The trajectories of k agents' credences in the n
          hypotheses over reps-many repetitions of the experiment.
    """
    slices = [priors_matrix]

    for _ in range(reps):
        next_slice = random_update(slices[-1], likelihoods_matrix, true_likelihoods)
        slices.append(next_slice)

    return np.stack(slices)


def plot_washing_out(priors_matrix, likelihoods_matrix, true_likelihoods, reps,
                     which=None, marker_styles=None, colors=None):
    """Plots trajectories of agents' credences in a repeated experiment.

    Args:
        priors_matrix: n-by-k array, where column j is agent j's priors over
          the hypotheses.
        likelihoods_matrix: n-by-m array, where column j is the likelihoods
          of outcome j.
        true_likelihoods: Sequence of length m. The actual likelihoods. This
          may be different to any row in likelihoods_matrix.
        reps: Number of times to repeat the experiment.
        which: Optional. Tuple of two tuples. The first contains the indexes of
          which hypotheses to plot. The second contains the indexes of which
          agents' credences to plot.
        marker_styles: Optional. Sequence of matplotlib marker styles, where
          markers are used to represent agents.
        colors: Optional. Sequence of matplotlib colors, where colors are used
          to represent hypotheses.
    """
    # assumes we plot at most four people
    if marker_styles is None:
        marker_styles = ['s', 'o', '+', '1']

    # assumes we plot at most five hypotheses
    if colors is None:
        colors = ['red', 'blue', 'green', 'yellow', 'black']

    try:
        which_hyp, which_agents = which
    # plot everything
    except:
        num_hyp, num_agents = priors_matrix.shape
        which_hyp = range(num_hyp)
        which_agents = range(num_agents)

    trajectories = washing_out(priors_matrix, likelihoods_matrix,
                                true_likelihoods, reps)

    fig, ax = plt.subplots()

    for hyp, agent in itertools.product(which_hyp, which_agents):
        ax.plot(
            range(reps + 1),
            trajectories[:, hyp, agent],
            color=colors[hyp],
            marker=marker_styles[agent],
        )

    ax.set_xlim([0, reps + 0.25])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel('iteration of experiment')
    ax.set_ylabel('credence')

    return fig, ax


# EXAMPLE
# example = {
#     'priors_matrix': np.array([[8/100, 15/100], [27/100, 60/100], [65/100, 25/100]]),
#     'likelihoods_matrix': np.array([[.1, .3, .3, .2, .1], [.2, .2, .2, .2, .2], [.2, .4, .1, .1, .2]]),
#     'true_likelihoods': [.2, .2, .2, .2, .2],
#     'reps': 25,
#     'which': ((0, 1, 2), (0, 1)),
# }

# plot_washing_out(**example)