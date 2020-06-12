import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

syn_data_color = 'red'
syn_data_label = 'Synthetic data'
orig_data_color = 'blue'
orig_data_label = 'Original data'

def plot_margins(syn_df, orig_df, show=False):
    fig, axis = plt.subplots()
    red_patch = mpatches.Patch(color=syn_data_color, label=syn_data_label)
    blue_patch = mpatches.Patch(color=orig_data_color, label=orig_data_label)
    axis.legend(handles=[blue_patch, red_patch])

    syn_violin_parts = axis.violinplot(syn_df.dropna().values, onp.arange(0, syn_df.shape[1]))
    for syn_violin_part in syn_violin_parts['bodies']:
        syn_violin_part.set_color(syn_data_color)
    syn_violin_parts["cmaxes"].set_color(syn_data_color)
    syn_violin_parts["cmins"].set_color(syn_data_color)
    syn_violin_parts["cbars"].set_color(syn_data_color)

    from collections import OrderedDict as od
    orig_violin_parts = od()
    for location, name in enumerate(syn_df.columns):
        orig_violin_parts[name] = axis.violinplot(orig_df[name].dropna().values, [location])
    for orig_violin_part in orig_violin_parts.values():
        body = orig_violin_part["bodies"][0]
        body.set_color(orig_data_color)
        orig_violin_part["cmaxes"].set_color(orig_data_color)
        orig_violin_part["cmins"].set_color(orig_data_color)
        orig_violin_part["cbars"].set_color(orig_data_color)

    axis.set_xticks(onp.arange(0, syn_df.shape[1]))
    axis.set_xticklabels(list(syn_df.columns))
    if show:
        fig.show()
    return fig

def plot_covariance_heatmap(syn_df, orig_df, show=False):
    syn_cov = syn_df.dropna().cov().values
    orig_cov = orig_df.dropna().cov().values
    fig, axis = plt.subplots(nrows=1, ncols=2)
    cov_max = onp.max([orig_cov, syn_cov])
    cov_min = onp.min([orig_cov, syn_cov])
    axis[0].imshow(orig_cov, vmin=cov_min, vmax=cov_max)
    axis[0].set_xticks(onp.arange(0, syn_df.shape[1]))
    axis[0].set_yticks([])
    axis[0].set_xticklabels(list(syn_df.columns), rotation="vertical", fontdict={"fontsize":10})
    axis[0].set_title(orig_data_label)

    im = axis[1].imshow(syn_cov, vmin=cov_min, vmax=cov_max)
    axis[1].set_xticks(onp.arange(0, syn_df.shape[1]))
    axis[1].set_yticks([])
    axis[1].set_xticklabels(list(syn_df.columns), rotation="vertical", fontdict={"fontsize":10})
    axis[1].set_title(syn_data_label)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if show:
        fig.show()
    return fig

def plot_missing_values(syn_df, orig_df, show=False):
    syn_nas = syn_df.isna().mean()
    orig_nas = orig_df.isna().mean()
    bar_width = 0.25
    fig, axis = plt.subplots()
    num_features = syn_df.shape[1]
    x_ticks = onp.arange(num_features)
    axis.bar(x_ticks-.5*bar_width, syn_nas, width=bar_width, color=syn_data_color, edgecolor='white', label=syn_data_label)
    axis.bar(x_ticks+.5*bar_width, orig_nas, width=bar_width, color=orig_data_color, edgecolor='white', label=orig_data_label)
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(list(syn_nas.index))

    red_patch = mpatches.Patch(color=syn_data_color, label=syn_data_label)
    blue_patch = mpatches.Patch(color=orig_data_color, label=orig_data_label)
    axis.legend(handles=[blue_patch, red_patch])

    if show:
        fig.show()
    return fig