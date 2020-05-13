import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def violin(syn_df, orig_df):
	plt.close()
	fig, axis = plt.subplots()
	red_patch = mpatches.Patch(color='red', label='Synthetic data')
	blue_patch = mpatches.Patch(color='blue', label='Original data')
	axis.legend(handles=[blue_patch, red_patch])

	syn_violin_parts = axis.violinplot(syn_df.values, onp.arange(0, syn_df.shape[1]))
	for syn_violin_part in syn_violin_parts['bodies']:
		syn_violin_part.set_color("red")
	syn_violin_parts["cmaxes"].set_color("red")
	syn_violin_parts["cmins"].set_color("red")
	syn_violin_parts["cbars"].set_color("red")

	from collections import OrderedDict as od
	orig_violin_parts = od()
	for location, name in enumerate(syn_df.columns):
		orig_violin_parts[name] = axis.violinplot(orig_df[name].dropna().values, [location])
	for orig_violin_part in orig_violin_parts.values():
		body = orig_violin_part["bodies"][0]
		body.set_color("blue")
		orig_violin_part["cmaxes"].set_color("blue")
		orig_violin_part["cmins"].set_color("blue")
		orig_violin_part["cbars"].set_color("blue")

	axis.set_xticks(onp.arange(0, syn_df.shape[1]))
	axis.set_xticklabels(list(syn_df.columns))
	return fig

def covariance_heatmap(syn_df, orig_df, path=None, show=False):
	plt.close()
	syn_cov = syn_df.cov().values
	orig_cov = orig_df.cov().values
	fig, axis = plt.subplots(nrows=1, ncols=2)
	cov_max = onp.max([orig_cov, syn_cov])
	cov_min = onp.min([orig_cov, syn_cov])
	axis[0].imshow(orig_cov, vmin=cov_min, vmax=cov_max)
	axis[0].set_xticks(onp.arange(0, syn_df.shape[1]))
	axis[0].set_yticks([])
	axis[0].set_xticklabels(list(syn_df.columns), rotation="vertical", fontdict={"fontsize":10})
	axis[0].set_title("Original data")

	im = axis[1].imshow(syn_cov, vmin=cov_min, vmax=cov_max)
	axis[1].set_xticks(onp.arange(0, syn_df.shape[1]))
	axis[1].set_yticks([])
	axis[1].set_xticklabels(list(syn_df.columns), rotation="vertical", fontdict={"fontsize":10})
	axis[1].set_title("Synthetic data")

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	return fig

