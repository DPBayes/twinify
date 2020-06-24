# Copyright 2020 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Visualization routines used by twinify main script.
"""

import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import OrderedDict as od

syn_data_color = 'red'
syn_data_label = 'Synthetic data'
orig_data_color = 'blue'
orig_data_label = 'Original data'

def plot_margins(syn_df, orig_df, show=False):
    fig, axis = plt.subplots()
    red_patch = mpatches.Patch(color=syn_data_color, label=syn_data_label)
    blue_patch = mpatches.Patch(color=orig_data_color, label=orig_data_label)
    axis.legend(handles=[blue_patch, red_patch])

    for location, name in enumerate(syn_df.columns):
        col = syn_df[name].dropna().values
        if col.size == 0:
            continue
        syn_violin_part = axis.violinplot(col, [location])
        syn_violin_part['bodies'][0].set_color(syn_data_color)
        syn_violin_part["cmaxes"].set_color(syn_data_color)
        syn_violin_part["cmins"].set_color(syn_data_color)
        syn_violin_part["cbars"].set_color(syn_data_color)

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
    axis.set_xticklabels(list(syn_df.columns), rotation=45)
    if show:
        fig.show()
    return fig

def plot_covariance_heatmap(syn_df, orig_df, show=False):
    syn_cov = syn_df.cov().values
    orig_cov = orig_df.cov().values
    cov_max = onp.maximum(onp.max(orig_cov[onp.isfinite(orig_cov)]), onp.max(syn_cov[onp.isfinite(syn_cov)]))
    cov_min = onp.minimum(onp.min(orig_cov[onp.isfinite(orig_cov)]), onp.min(syn_cov[onp.isfinite(syn_cov)]))

    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.15, share_all=True, cbar_location="right", cbar_mode="single", cbar_pad=0.15)

    grid[0].imshow(orig_cov, vmin=cov_min, vmax=cov_max)
    grid[0].set_xticks(onp.arange(0, syn_df.shape[1]))
    grid[0].set_yticks([])
    grid[0].set_xticklabels(list(syn_df.columns), rotation=45, fontdict={"fontsize":10})
    grid[0].set_title(orig_data_label)

    im = grid[1].imshow(syn_cov, vmin=cov_min, vmax=cov_max)
    grid[1].set_xticks(onp.arange(0, syn_df.shape[1]))
    grid[1].set_yticks(onp.arange(0, syn_df.shape[1]))
    grid[1].set_xticklabels(list(syn_df.columns), rotation=45, fontdict={"fontsize":10})
    grid[1].set_yticklabels(list(syn_df.columns), fontdict={"fontsize":10})
    grid[1].set_title(syn_data_label)

    grid[1].cax.colorbar(im)
    grid[1].cax.toggle_label(True)

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
    axis.set_xticklabels(list(syn_nas.index), rotation=45)

    red_patch = mpatches.Patch(color=syn_data_color, label=syn_data_label)
    blue_patch = mpatches.Patch(color=orig_data_color, label=orig_data_label)
    axis.legend(handles=[blue_patch, red_patch])

    if show:
        fig.show()
    return fig