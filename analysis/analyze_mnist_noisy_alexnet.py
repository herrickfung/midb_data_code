"""
Standalone MNIST analysis: Human vs AlexNet vs Noisy AlexNet.

Independent of analyze_main.py / util/plotting.py - noisy_alexnet is a
one-off model variant (MNIST-only) that doesn't fit the existing
multi-model, multi-dataset pipeline, so this script owns its own data
loading and plotting rather than extending the shared ones.
"""
import argparse
import pathlib
import tarfile
from itertools import combinations

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from indimap import IndiMap
from indimap.util import stat_func
import util.dataset as dataset

# Editable text (not outlined) when the PDF is opened in Illustrator.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

HUMAN_COLOR = (0.8941176470588236, 0.10196078431372549, 0.10980392156862745)
ALEXNET_COLOR = (0.30196078431372547, 0.6862745098039216, 0.2901960784313726)
GROUP_LABELS = ['Human', 'AlexNet', 'Noisy AlexNet']
GROUP_COLORS = [HUMAN_COLOR, ALEXNET_COLOR, ALEXNET_COLOR]
GROUP_ALPHAS = [1.0, 1.0, 0.4]  # noisy AlexNet = lower-alpha version of the AlexNet color
METRIC_LABELS = ['Accuracy', 'Confidence']


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def get_noisy_alexnet_on_mnist(variant: str = 'standard'):
    """ Mirrors util.dataset.get_alexnet_on_mnist for the noisy_alexnet model. """
    human_data = dataset.get_human_on_mnist()
    model_data = pd.read_csv(f'dataset/mnist/{variant}/noisy_alexnet.csv')
    model_data['mnist_index'] = model_data.minst_index + 1
    model_data['cond'] = [0 if x in [0, 1] else 1 for x in model_data.cond]
    model_data['conf'] = model_data['top2diff_conf']
    if variant == 'standard':
        model_data.loc[model_data['resp'].isin([0, 9]), 'resp'] = np.nan

    config = {
        'task_name': 'mnist',
        'model_name': 'noisy_alexnet',
        'subj_data': human_data,
        'inst_data': model_data,
        'map_variables': ['acc', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'output_path': f'IndiMap_results/{variant}/mnist_noisy_alexnet',
        'graph_path': f'IndiMap_plots/{variant}/mnist_noisy_alexnet',
    }
    return {**dataset.default_config, **config}


def manage_path():
    current_path = pathlib.Path(__file__).parent.absolute()
    graph_path = current_path / 'graphs' / 'mnist_noisy_alexnet'
    graph_path.mkdir(parents=True, exist_ok=True)
    return current_path, graph_path


def download_and_extract_data(current_path):
    OSF_DATA_URL = "https://files.osf.io/v1/resources/n6m7b/providers/osfstorage/69741b25a94a8b2b80bb82dc"
    tar_path = current_path / 'midb_data.tar.gz'

    data_path_for_check = [current_path / 'dataset', current_path / 'IndiMap_results']
    if all(p.exists() for p in data_path_for_check):
        return

    print("Downloading data from OSF, please wait ...")
    response = requests.get(OSF_DATA_URL, stream=True)
    with open(tar_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=current_path)

    tar_path.unlink()


def init_map(variant: str = 'standard'):
    """ maps[0] = AlexNet-on-MNIST (also carries the Human data used for the
    Human-Human bar), maps[1] = Noisy AlexNet-on-MNIST. """
    return [
        IndiMap(dataset.get_alexnet_on_mnist(variant)),
        IndiMap(get_noisy_alexnet_on_mnist(variant)),
    ]


def compute(maps, load: bool = False):
    """ Only CorrMap/RankMap are needed for these plots - skip Top/Dims/Pred. """
    for obj in maps:
        obj.compute_corr(load_exists=load)
        obj.compute_rank(load_exists=load)


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

def _save_pdf(path, dpi=384):
    fig = plt.gcf()
    for ax in fig.axes:
        for artist in ax.get_children():
            artist.set_clip_on(False)
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close(fig)


def _style_boxplot(box, color, alpha, linewidth=2.5):
    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(alpha * 0.5)
        patch.set_linewidth(0)
    for part in ('whiskers', 'caps', 'medians'):
        for artist in box[part]:
            artist.set_color(color)
            artist.set_linewidth(linewidth)
            artist.set_alpha(alpha)


def _annotate_bracket(x1, x2, y, text, alpha=1.0, fontsize=9, xytext=(0, 1), y_text_offset=0.0):
    ax = plt.gca()
    ax.plot([x1, x2], [y, y], color='black', linewidth=1.5, alpha=alpha)
    ax.annotate(text, ((x1 + x2) / 2, y + y_text_offset), textcoords="offset points",
                xytext=xytext, ha='center', size=fontsize, alpha=alpha)


def _format_pval(p_val, threshold=1e-3):
    if p_val <= 0:
        p_val = np.finfo(float).tiny
    if p_val < threshold:
        power = int(np.floor(np.log10(p_val)))
        coefficient = p_val / (10 ** power)
        return f"p = {coefficient:.2f} x 10^{power}"
    return f"p = {p_val:.3f}"


def _format_pval_simple(p_val, threshold=0.001):
    if p_val < threshold:
        return f'p < {threshold}'
    return f'p = {p_val:.3f}'


def _stars_for_pval(p_val):
    if p_val < 1e-3:
        return '***', 1
    elif p_val < 0.01:
        return '**', 1
    elif p_val < 0.05:
        return '*', 1
    return 'n.s.', 0.5


def _legend():
    """ Placed fully outside the axes (to the right) so it never collides
    with the significance brackets drawn above the boxes. """
    handles = [Patch(facecolor=GROUP_COLORS[g], alpha=GROUP_ALPHAS[g] * 0.5, label=GROUP_LABELS[g])
               for g in range(3)]
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=9, frameon=False, borderaxespad=0)


def _plot_grouped_boxes(data, group_spacing=0.8, metric_gap=3.0, scatter=True, scatter_alpha_scale=0.75):
    """ data: (n_groups, n_metrics, n_samples) array - one box (+ optional
    per-sample scatter) per (group, metric) cell, colored/alpha'd by group.
    Returns (x_positions[n_groups, n_metrics], tick_centers[n_metrics]). """
    n_groups, n_metrics, _ = data.shape
    x_positions = np.empty((n_groups, n_metrics))
    for met in range(n_metrics):
        for g in range(n_groups):
            x_pos = met * metric_gap + g * group_spacing
            x_positions[g, met] = x_pos
            vals = data[g, met]
            vals = vals[~np.isnan(vals)]
            box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, GROUP_COLORS[g], GROUP_ALPHAS[g])
            if scatter:
                for val in vals:
                    plt.scatter(x_pos - 0.3, val, color=GROUP_COLORS[g],
                                alpha=scatter_alpha_scale * GROUP_ALPHAS[g], s=10)
    tick_centers = [met * metric_gap + (n_groups - 1) * group_spacing / 2 for met in range(n_metrics)]
    return x_positions, tick_centers


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_alignment_average(maps, path):
    n_groups = 3
    n_metrics = len(maps[0].map_var)
    n_subjs = maps[0].n_subjs

    avg_per_subj = np.full((n_groups, n_metrics, n_subjs), np.nan)
    for g in range(n_groups):
        if g == 0:
            # get_corr_map('subj', 'subj') already excludes the trivial
            # self-match at the source (shape (..., subj, subj - 1)), unlike
            # the manually diagonal-padded matrix used in the standard
            # pipeline's plot_raw_matrix - no further masking needed here.
            mat = maps[0].get_corr_map('subj', 'subj').mat  # (boot, split, met, subj, subj-1)
        else:
            mat = maps[g - 1].get_corr_map('subj', 'inst').mat
        mat = np.nanmean(mat, axis=(0, 1))
        avg_per_subj[g] = np.nanmean(mat, axis=-1)

    plt.figure(figsize=(4.5, 4))
    x_positions, tick_centers = _plot_grouped_boxes(avg_per_subj)

    stat_data = stat_func.r2z(avg_per_subj, metric='pearson')
    data_min = np.nanmin(avg_per_subj)
    data_max = np.nanmax(avg_per_subj)
    data_range = data_max - data_min
    max_bracket = data_max

    for met in range(n_metrics):
        for k, (i, j) in enumerate(combinations(range(n_groups), 2)):
            _, p_val = stats.ttest_rel(stat_data[i, met], stat_data[j, met], nan_policy='omit')
            y_max = data_max + data_range * (0.12 + 0.1 * k)
            max_bracket = max(max_bracket, y_max)
            _annotate_bracket(x_positions[i, met], x_positions[j, met], y_max, _format_pval(p_val),
                               y_text_offset=data_range * 0.01)

    plt.xticks(tick_centers, METRIC_LABELS[:n_metrics], fontsize=12)
    plt.xlim(x_positions.min() - 1, x_positions.max() + 1)
    plt.ylim(data_min - 0.1 * data_range, max_bracket + 0.15 * data_range)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Correlation coefficient', fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _legend()
    plt.tight_layout()
    _save_pdf(path / 'align_avg_mnist.pdf')


def plot_alignment_variance(maps, path):
    n_groups = 3
    n_boots, n_splits, n_metrics, n_subjs, _ = maps[0].get_corr_map('subj', 'subj').mat.shape

    std_per_subj = np.full((n_groups, n_boots * n_splits, n_metrics, n_subjs), np.nan)
    for g in range(n_groups):
        if g == 0:
            mat = maps[0].get_corr_map('subj', 'subj').mat
        else:
            mat = maps[g - 1].get_corr_map('subj', 'inst').mat
        mat = stat_func.r2z(mat, metric='pearson')
        mat = np.std(mat, axis=-1)  # std across the "to" axis, per boot/split
        std_per_subj[g] = mat.reshape(n_boots * n_splits, n_metrics, n_subjs)

    avg_std = np.mean(std_per_subj, axis=1)  # (n_groups, n_metrics, n_subjs)

    plt.figure(figsize=(4.5, 4))
    x_positions, tick_centers = _plot_grouped_boxes(avg_std)

    data_min = np.nanmin(avg_std)
    data_max = np.nanmax(avg_std)
    data_range = data_max - data_min
    max_bracket = data_max

    for met in range(n_metrics):
        for k, (i, j) in enumerate(combinations(range(n_groups), 2)):
            # Matches the standard pipeline's plot_alignment_variance: the
            # proportion test runs over the flattened (boot, subj) array,
            # not on a per-subject-averaged difference.
            diff = std_per_subj[i, :, met] - std_per_subj[j, :, met]  # (n_boots * n_splits, n_subjs)
            p_val = 2 * min(np.mean(diff < 0), np.mean(diff > 0))
            y_max = data_max + data_range * (0.12 + 0.1 * k)
            max_bracket = max(max_bracket, y_max)
            anno = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.3f}'
            _annotate_bracket(x_positions[i, met], x_positions[j, met], y_max, anno,
                               y_text_offset=data_range * 0.01)

    plt.xticks(tick_centers, METRIC_LABELS[:n_metrics], fontsize=12)
    plt.xlim(x_positions.min() - 1, x_positions.max() + 1)
    plt.ylim(max(0, data_min - 0.1 * data_range), max_bracket + 0.15 * data_range)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Standard deviation', fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _legend()
    plt.tight_layout()
    _save_pdf(path / 'align_var_mnist.pdf')


def _corr_consistency_plot(maps, path, split_by, btw, xtick_labels, title_suffix, filename):
    n_groups = 3
    n_boots, n_metrics, n_subjs = maps[0].get_corr_results('subj', 'inst', 'subj', btw, split_by).mat.shape

    plot_data = np.full((2, n_groups, n_metrics, n_subjs), np.nan)
    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for g in range(n_groups):
            if g == 0:
                mat = maps[0].get_corr_results('subj', 'subj', map_type, btw, split_by).mat
            else:
                mat = maps[g - 1].get_corr_results('subj', 'inst', map_type, btw, split_by).mat
            mat = stat_func.r2z(mat, metric='pearson')
            plot_data[type_idx, g] = np.nanmean(mat, axis=0)

    plot_data = -np.diff(plot_data, axis=0)
    plot_data = np.squeeze(plot_data, axis=0)  # (n_groups, n_metrics, n_subjs), z-space
    plot_data_r = stat_func.z2r(plot_data, metric='pearson')

    figsize = (4.5, 4) if n_metrics > 1 else (3.2, 4)
    plt.figure(figsize=figsize)
    x_positions, tick_centers = _plot_grouped_boxes(plot_data_r)

    data_min = np.nanmin(plot_data_r)
    data_max = np.nanmax(plot_data_r)
    data_range = data_max - data_min
    vs_zero_y = data_min - 0.15 * data_range
    max_bracket = data_max

    for met in range(n_metrics):
        for k, (i, j) in enumerate(combinations(range(n_groups), 2)):
            _, p_val = stats.ttest_ind(plot_data[i, met], plot_data[j, met], equal_var=False, nan_policy='omit')
            y_max = data_max + data_range * (0.1 + 0.08 * k)
            max_bracket = max(max_bracket, y_max)
            _annotate_bracket(x_positions[i, met], x_positions[j, met], y_max, _format_pval(p_val),
                               y_text_offset=data_range * 0.02)
        for g in range(n_groups):
            _, p_val = stats.ttest_1samp(plot_data[g, met], 0, nan_policy='omit')
            anno, alpha = _stars_for_pval(p_val)
            plt.annotate(anno, (x_positions[g, met], vs_zero_y), ha='center', size=8, alpha=alpha, fontweight='bold')

    plt.xticks(tick_centers, xtick_labels, fontsize=12)
    plt.xlim(x_positions.min() - 1, x_positions.max() + 1)
    plt.ylim(vs_zero_y - 0.08 * data_range, max_bracket + 0.15 * data_range)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics' if btw == 'split' else 'Pairs of behavioral metrics',
                fontsize=13, fontweight='bold')
    plt.ylabel('r(same subject) - r(other subjects)', fontsize=12, fontweight='bold')
    plt.title(f'Correlation consistency{title_suffix}', fontsize=15, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _legend()
    plt.tight_layout()
    _save_pdf(path / filename)


def plot_corr_within_metric_consistency(maps, path, split_by='rand'):
    _corr_consistency_plot(maps, path, split_by, btw='split', xtick_labels=METRIC_LABELS,
                            title_suffix='', filename=f'corr_btw_bs_mnist_split_{split_by}.pdf')


def plot_corr_across_metric_consistency(maps, path, split_by='rand'):
    pair_labels = [f'{maps[0].map_var[i].capitalize()}-{maps[0].map_var[j].capitalize()}'
                   for i, j in combinations(range(len(maps[0].map_var)), 2)]
    _corr_consistency_plot(maps, path, split_by, btw='var', xtick_labels=pair_labels,
                            title_suffix=' (across metrics)', filename=f'corr_btw_var_mnist_split_{split_by}.pdf')


def _rank_consistency_plot(maps, path, split_by, btw, xtick_labels, title_suffix, filename):
    n_groups = 3
    n_boots, n_metrics, n_subjs = maps[0].get_corr_results('subj', 'inst', 'subj', btw, split_by).mat.shape

    plot_data = np.full((n_groups, n_boots, n_metrics), np.nan)
    for g in range(n_groups):
        if g == 0:
            plot_data[g] = maps[0].get_rank_results('subj', 'subj', btw, split_by).mat
        else:
            plot_data[g] = maps[g - 1].get_rank_results('subj', 'inst', btw, split_by).mat

    data_for_box = np.transpose(plot_data, (0, 2, 1))  # (n_groups, n_metrics, n_boots)

    figsize = (4.5, 4) if n_metrics > 1 else (3.2, 4)
    plt.figure(figsize=figsize)
    x_positions, tick_centers = _plot_grouped_boxes(data_for_box, scatter=False)

    data_min = np.nanmin(plot_data)
    data_max = np.nanmax(plot_data)
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    for met in range(n_metrics):
        for k, (i, j) in enumerate(combinations(range(n_groups), 2)):
            diff = plot_data[i, :, met] - plot_data[j, :, met]
            p_val = 2 * min(np.mean(diff >= 0), np.mean(diff < 0))
            y_max = data_max + data_range * (0.1 + 0.08 * k)
            max_bracket = max(max_bracket, y_max)
            _annotate_bracket(x_positions[i, met], x_positions[j, met], y_max, _format_pval_simple(p_val),
                               fontsize=9, xytext=(0, 3))

    plt.xticks(tick_centers, xtick_labels, fontsize=12)
    plt.xlim(x_positions.min() - 1, x_positions.max() + 1)
    plt.ylim(min_annot - 0.08 * data_range, max_bracket + 0.15 * data_range)
    plt.xlabel('Behavioral metrics' if btw == 'split' else 'Pairs of behavioral metrics',
                fontsize=13, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title(f'Rank consistency{title_suffix}', fontsize=15, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _legend()
    plt.tight_layout()
    _save_pdf(path / filename)


def plot_rank_within_metric_consistency(maps, path, split_by='rand'):
    _rank_consistency_plot(maps, path, split_by, btw='split', xtick_labels=METRIC_LABELS,
                            title_suffix='', filename=f'rank_btw_bs_mnist_split_{split_by}.pdf')


def plot_rank_across_metric_consistency(maps, path, split_by='rand'):
    pair_labels = [f'{maps[0].map_var[i].capitalize()}-{maps[0].map_var[j].capitalize()}'
                   for i, j in combinations(range(len(maps[0].map_var)), 2)]
    _rank_consistency_plot(maps, path, split_by, btw='var', xtick_labels=pair_labels,
                            title_suffix=' (across metrics)', filename=f'rank_btw_var_mnist_split_{split_by}.pdf')


def graph(maps, path, split_by='rand'):
    plot_alignment_average(maps, path)
    plot_alignment_variance(maps, path)
    plot_corr_within_metric_consistency(maps, path, split_by)
    plot_rank_within_metric_consistency(maps, path, split_by)
    plot_corr_across_metric_consistency(maps, path, split_by)
    plot_rank_across_metric_consistency(maps, path, split_by)


def main():
    parser = argparse.ArgumentParser(
        description="MNIST alignment analysis: Human vs AlexNet vs Noisy AlexNet"
    )
    parser.add_argument('--recompute', default=False, action='store_true',
                        help='Recompute all results instead of loading')
    args = parser.parse_args()
    load = not args.recompute

    current_path, graph_path = manage_path()
    download_and_extract_data(current_path)
    maps = init_map(variant='standard')
    compute(maps, load=load)
    graph(maps, graph_path)


if __name__ == "__main__":
    main()
