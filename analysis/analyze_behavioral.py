""" behavioral heatmap analysis and plot script.

Plots a single subject/instance x category heatmap grid per dataset (MNIST,
ecoset10): columns are Human / RTNet / AlexNet / ResNet18, rows are the
behavioral metrics (accuracy, confidence, RT - where available). Rows are
subjects (Human column) or network instances (ANN columns), columns within
each panel are the behavioral categories (digits for MNIST, classes for
ecoset10).

Column banding within a panel (certain categories consistently darker/lighter
across almost every row) would indicate a shared category-difficulty signal;
a checkerboard-like pattern with no consistent column structure would
indicate idiosyncrasy dominates instead.

This reads the raw per-trial dataframes directly from util/dataset.py's
existing getters (grouped by category via variant='category' for humans, and
via the *_category_map getters' 'inst_data' for the ANNs) and pivots them
into subject/instance x category matrices - no IndiMap object is needed.

Also plots the MNIST accuracy-control counterpart: one combined heatmap per
sd in {-2, -1, 0, +1, +2} (accuracy control is MNIST-only), reading directly
from the accuracy-control CSVs for sd != 0. The Human panel is identical
across all five since only the ANNs' accuracy is shifted.

Also plots a "genericness" analysis testing whether a subject/instance's
similarity to the group-average (category-level) profile is a stable trait
or a noisy, category-contingent artifact - using only single-subject-level
computations (leave-one-out group averages, per-subject correlations), no
pairwise similarity matrix or subject/instance mapping:
1. consensus profile check - leave-one-out group-average category profile,
   computed independently within two item-partition split halves.
2. genericness score - each subject/instance's own profile correlated
   against the leave-one-out consensus, within each split half.
3. genericness reliability under a random item split.
4. genericness reliability under a category split (first half of categories
   vs second half).
5. residualized split-half reliability - subtract the consensus profile from
   the raw profile in each split half, then check whether the residual
   ("idiosyncratic") profile is more reliable (random split) than the raw
   profile, which would confirm the consensus/generic component is what
   drags down the raw split-half reliability.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from indimap.util import stat_func

import util.dataset as dataset


METRIC_LABELS = {
    'acc': 'Accuracy',
    'conf': 'Confidence',
    'rt': 'RT',
}

GROUP_LABELS = {
    'human': 'Human',
    'rtnet': 'RTNet',
    'alexnet': 'AlexNet',
    'resnet18': 'ResNet18',
}

INDEX_COL = {
    'human': 'subj',
    'rtnet': 'inst',
    'alexnet': 'inst',
    'resnet18': 'inst',
}

GROUPS = ['human', 'rtnet', 'alexnet', 'resnet18']
METRICS = ['acc', 'conf', 'rt']

ACCURACY_CONTROL_SDS = [-2, -1, 0, 1, 2]
ACCURACY_CONTROL_CATEGORY_MAP_GETTERS = {
    'rtnet': dataset.get_rtnet_on_mnist_category_map,
    'alexnet': dataset.get_alexnet_on_mnist_category_map,
    'resnet18': dataset.get_resnet18_on_mnist_category_map,
}

EXPTS = ['mnist', 'ecoset10']
ITEM_COL = {'mnist': 'mnist_index', 'ecoset10': 'image_index'}
GENERICNESS_SEED = 0


def manage_path():
    current_path = pathlib.Path(__file__).parent.absolute()
    graph_path = current_path / 'graphs' / 'behavioral'
    graph_path.mkdir(parents=True, exist_ok=True)
    return current_path, graph_path


def _pivot(data, index_col, metric):
    mat = data.pivot_table(index=index_col, columns='stim', values=metric, aggfunc='mean')
    return mat.sort_index().sort_index(axis=1)


def plot_combined_heatmap(expt, data_by_group, path):
    """ Grid of subject/instance x category heatmaps: columns = groups
    (Human, RTNet, AlexNet, ResNet18), rows = metrics (accuracy, confidence,
    RT - where available). Color scale is shared within a row so groups are
    directly comparable for that metric. Each panel has its own color scale
    (and its own colorbar) since raw scales differ wildly across groups
    (e.g. RTNet's confidence metric is on a very different scale than the
    others), so a shared per-row scale would wash out everything else. """
    matrices = {}
    for metric in METRICS:
        for group in GROUPS:
            df = data_by_group[group]
            if metric in df.columns:
                matrices[(metric, group)] = _pivot(df, INDEX_COL[group], metric)

    n_rows, n_cols = len(METRICS), len(GROUPS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows))

    for i, metric in enumerate(METRICS):
        for j, group in enumerate(GROUPS):
            ax = axes[i, j]
            key = (metric, group)
            if key not in matrices:
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10, alpha=0.5,
                        transform=ax.transAxes)
                continue

            mat = matrices[key]
            im = ax.imshow(mat.values, aspect='auto', cmap='rainbow')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels(mat.columns, fontsize=7)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(GROUP_LABELS[group], fontsize=13, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{METRIC_LABELS[metric]}\n({INDEX_COL[group]}, n={mat.shape[0]})',
                               fontsize=10, fontweight='bold')
            if i == n_rows - 1:
                ax.set_xlabel('Category', fontsize=9)

    fig.suptitle(f'{expt} - behavioral heatmaps', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_name = path / f'{expt}_combined_heatmap.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()


def _sd_label(sd):
    return f'+{sd}sd' if sd > 0 else f'{sd}sd'


def _load_accuracy_control_model_data(model_name, sd):
    """ MNIST accuracy-control model data, grouped by category ('stim').

    sd == 0 is the standard (untouched) trained network, same data used by
    the main mnist combined heatmap; sd in {-2, -1, 1, 2} reads the
    accuracy-control CSVs directly, which already come with per-trial 'stim'
    and 'acc'/'conf'(/'rt' for rtnet) columns - pivot_table's aggfunc='mean'
    takes care of averaging over the remaining trial-level columns.
    """
    if sd == 0:
        return ACCURACY_CONTROL_CATEGORY_MAP_GETTERS[model_name]()['inst_data']

    df = pd.read_csv(f'dataset/mnist/accuracy_control/{model_name}/mean{sd}sd.csv')
    df = df.rename(columns={'instance': 'inst'})
    if 'conf' not in df.columns and 'top2diff_conf' in df.columns:
        df['conf'] = df['top2diff_conf']
    return df


def graph_accuracy_control(path):
    """ MNIST accuracy-control combined heatmaps, one figure per sd in
    {-2, -1, 0, +1, +2}. The Human panel is identical across all five since
    only the ANNs' accuracy is shifted. """
    human_data = dataset.get_human_on_mnist('category')
    for sd in ACCURACY_CONTROL_SDS:
        data_by_group = {
            'human': human_data,
            'rtnet': _load_accuracy_control_model_data('rtnet', sd),
            'alexnet': _load_accuracy_control_model_data('alexnet', sd),
            'resnet18': _load_accuracy_control_model_data('resnet18', sd),
        }
        plot_combined_heatmap(f'mnist_accuracy_control_{_sd_label(sd)}', data_by_group, path)


def graph(path):
    mnist_data = {
        'human': dataset.get_human_on_mnist('category'),
        'rtnet': dataset.get_rtnet_on_mnist_category_map()['inst_data'],
        'alexnet': dataset.get_alexnet_on_mnist_category_map()['inst_data'],
        'resnet18': dataset.get_resnet18_on_mnist_category_map()['inst_data'],
    }
    plot_combined_heatmap('mnist', mnist_data, path)

    ecoset10_data = {
        'human': dataset.get_human_on_ecoset10('category'),
        'rtnet': dataset.get_rtnet_on_ecoset10_category_map()['inst_data'],
        'alexnet': dataset.get_alexnet_on_ecoset10_category_map()['inst_data'],
        'resnet18': dataset.get_resnet18_on_ecoset10_category_map()['inst_data'],
    }
    plot_combined_heatmap('ecoset10', ecoset10_data, path)

    graph_accuracy_control(path)


def _group_item_level_data(expt, group):
    """ Item-level (per subject/instance x item x category) raw dataframe for
    a given (expt, group), preserving both item id and 'stim' (category). """
    getters = {
        ('mnist', 'human'): lambda: dataset.get_human_on_mnist('standard'),
        ('mnist', 'rtnet'): lambda: dataset.get_rtnet_on_mnist('standard')['inst_data'],
        ('mnist', 'alexnet'): lambda: dataset.get_alexnet_on_mnist('standard')['inst_data'],
        ('mnist', 'resnet18'): lambda: dataset.get_resnet18_on_mnist('standard')['inst_data'],
        ('ecoset10', 'human'): lambda: dataset.get_human_on_ecoset10('standard'),
        ('ecoset10', 'rtnet'): lambda: dataset.get_rtnet_on_ecoset10('standard')['inst_data'],
        ('ecoset10', 'alexnet'): lambda: dataset.get_alexnet_on_ecoset10('standard')['inst_data'],
        ('ecoset10', 'resnet18'): lambda: dataset.get_resnet18_on_ecoset10('standard')['inst_data'],
    }
    return getters[(expt, group)]()


def _category_halves(cats):
    cats = sorted(cats)
    half = len(cats) // 2
    return cats[:half], cats[half:]


def _stratified_item_split(df, item_col, seed=GENERICNESS_SEED):
    """ Random 50/50 split of items into two halves, stratified per category
    so both halves cover every category (needed since genericness/residual
    profiles are compared category-by-category across split halves). """
    rng = np.random.default_rng(seed)
    item_to_stim = df[[item_col, 'stim']].drop_duplicates().set_index(item_col)['stim']
    split_a, split_b = [], []
    for cat in sorted(item_to_stim.unique()):
        items = item_to_stim[item_to_stim == cat].index.to_numpy().copy()
        rng.shuffle(items)
        half = len(items) // 2
        split_a.extend(items[:half].tolist())
        split_b.extend(items[half:].tolist())
    return set(split_a), set(split_b)


def _category_profile(df, index_col, item_col, metric, items=None, cats=None):
    """ subject/instance x category matrix of mean(metric), optionally
    restricted to a subset of items and/or categories before aggregating. """
    sub = df
    if items is not None:
        sub = sub[sub[item_col].isin(items)]
    if cats is not None:
        sub = sub[sub['stim'].isin(cats)]
    mat = sub.pivot_table(index=index_col, columns='stim', values=metric, aggfunc='mean')
    return mat.sort_index().sort_index(axis=1)


def _leave_one_out_consensus(mat):
    """ Same-shape matrix where row i is the mean of all other rows
    (excluding i) - the leave-one-out group-average profile for subject i. """
    total = mat.sum(axis=0, skipna=True)
    count = mat.notna().sum(axis=0)
    present = mat.notna()
    numer = total.to_numpy()[None, :] - mat.fillna(0).to_numpy()
    denom = count.to_numpy()[None, :] - present.to_numpy().astype(int)
    with np.errstate(invalid='ignore', divide='ignore'):
        loo = numer / denom
    loo[denom == 0] = np.nan
    return pd.DataFrame(loo, index=mat.index, columns=mat.columns)


def _row_corr(mat_a, mat_b):
    """ Per-row (subject/instance) Pearson correlation across the shared
    columns of mat_a and mat_b. """
    common_idx = mat_a.index.intersection(mat_b.index)
    common_cols = mat_a.columns.intersection(mat_b.columns)
    out = {}
    for idx in common_idx:
        a = mat_a.loc[idx, common_cols].to_numpy(dtype=float)
        b = mat_b.loc[idx, common_cols].to_numpy(dtype=float)
        valid = ~np.isnan(a) & ~np.isnan(b)
        if valid.sum() < 2 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
            out[idx] = np.nan
        else:
            out[idx] = np.corrcoef(a[valid], b[valid])[0, 1]
    return pd.Series(out)


def _paired_corr(series_a, series_b):
    """ Single Pearson correlation between two subject/instance-indexed
    Series (e.g. genericness scores from two split halves). """
    common = series_a.index.intersection(series_b.index)
    a = series_a.loc[common].to_numpy(dtype=float)
    b = series_b.loc[common].to_numpy(dtype=float)
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 2:
        return np.nan
    return np.corrcoef(a[valid], b[valid])[0, 1]


def _z_avg(series):
    """ Fisher-z-average a Series of per-subject correlations back to r. """
    vals = series.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan
    return stat_func.z2r(np.nanmean(stat_func.r2z(vals, metric='pearson')), metric='pearson')


def compute_genericness_results(expt, seed=GENERICNESS_SEED):
    """ For each group/metric, compute everything needed for the 5 checks:
    consensus profiles, genericness scores (random split and category split),
    and raw vs. residualized split-half reliability (random split). """
    item_col = ITEM_COL[expt]
    raw_data = {group: _group_item_level_data(expt, group) for group in GROUPS}

    reference_df = raw_data['human']
    cats_sorted = sorted(reference_df['stim'].unique())
    cat_a, cat_b = _category_halves(cats_sorted)
    split_a_items, split_b_items = _stratified_item_split(reference_df, item_col, seed=seed)

    results = {}
    for group in GROUPS:
        df = raw_data[group]
        index_col = INDEX_COL[group]
        results[group] = {}

        for metric in METRICS:
            if metric not in df.columns:
                continue

            profile_a = _category_profile(df, index_col, item_col, metric, items=split_a_items)
            profile_b = _category_profile(df, index_col, item_col, metric, items=split_b_items)
            consensus_a = _leave_one_out_consensus(profile_a)
            consensus_b = _leave_one_out_consensus(profile_b)
            genericness_a = _row_corr(profile_a, consensus_a)
            genericness_b = _row_corr(profile_b, consensus_b)

            profile_cat_a = _category_profile(df, index_col, item_col, metric, cats=cat_a)
            profile_cat_b = _category_profile(df, index_col, item_col, metric, cats=cat_b)
            consensus_cat_a = _leave_one_out_consensus(profile_cat_a)
            consensus_cat_b = _leave_one_out_consensus(profile_cat_b)
            genericness_cat_a = _row_corr(profile_cat_a, consensus_cat_a)
            genericness_cat_b = _row_corr(profile_cat_b, consensus_cat_b)

            # Note: profile_cat_a and profile_cat_b live over disjoint category
            # axes (1st half vs 2nd half of categories), so there is no shared
            # column to correlate a subject's own profile between them - only
            # the *within-half* genericness scores (above) are meaningful for
            # the category split. The residualization test (step 5) therefore
            # only uses the random (item-partition) split, which shares the
            # full category axis across both halves.
            residual_a = profile_a - consensus_a
            residual_b = profile_b - consensus_b
            raw_r_rand = _row_corr(profile_a, profile_b)
            resid_r_rand = _row_corr(residual_a, residual_b)

            results[group][metric] = {
                'mean_profile_a': profile_a.mean(axis=0, skipna=True),
                'mean_profile_b': profile_b.mean(axis=0, skipna=True),
                'genericness_a': genericness_a,
                'genericness_b': genericness_b,
                'genericness_cat_a': genericness_cat_a,
                'genericness_cat_b': genericness_cat_b,
                'raw_r_rand': raw_r_rand,
                'resid_r_rand': resid_r_rand,
            }

            print(f"[{expt}/{group}/{metric}] genericness reliability - random split: "
                  f"r={_paired_corr(genericness_a, genericness_b):.3f}, "
                  f"category split: r={_paired_corr(genericness_cat_a, genericness_cat_b):.3f} | "
                  f"split-half reliability (random split) - raw: r={_z_avg(raw_r_rand):.3f}, "
                  f"residual: r={_z_avg(resid_r_rand):.3f}")

    return results


def _genericness_grid_figure(expt, results, panel_fn, suptitle, filename, path):
    n_rows, n_cols = len(METRICS), len(GROUPS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows))

    for i, metric in enumerate(METRICS):
        for j, group in enumerate(GROUPS):
            ax = axes[i, j]
            result = results.get(group, {}).get(metric)
            if result is None:
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10, alpha=0.5,
                        transform=ax.transAxes)
                continue

            panel_fn(ax, result)
            if i == 0:
                # Use an annotation (not set_title) so it doesn't clobber a
                # panel-specific title (e.g. the r-value) set by panel_fn.
                ax.annotate(GROUP_LABELS[group], xy=(0.5, 1.22), xycoords='axes fraction',
                            ha='center', fontsize=13, fontweight='bold')
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=10, fontweight='bold')

    fig.suptitle(f'{expt} - {suptitle}', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path / filename, dpi=384, transparent=True)
    plt.close()


def _panel_consensus(ax, result):
    profile_a = result['mean_profile_a']
    profile_b = result['mean_profile_b']
    ax.plot(profile_a.index.astype(str), profile_a.to_numpy(), marker='o', label='Split A', color='tab:blue')
    ax.plot(profile_b.index.astype(str), profile_b.to_numpy(), marker='o', label='Split B', color='tab:orange')
    ax.set_xlabel('Category', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def _panel_genericness_dist(ax, result):
    a = result['genericness_a'].dropna().to_numpy()
    b = result['genericness_b'].dropna().to_numpy()
    box = ax.boxplot([a, b], tick_labels=['Split A', 'Split B'], widths=0.5, showfliers=False, patch_artist=True)
    for patch, color in zip(box['boxes'], ['tab:blue', 'tab:orange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0, color='black', linestyle='dotted', linewidth=1, alpha=0.6)


def _panel_reliability_scatter(ax, series_a, series_b):
    common = series_a.index.intersection(series_b.index)
    x = series_a.loc[common].to_numpy(dtype=float)
    y = series_b.loc[common].to_numpy(dtype=float)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    ax.scatter(x, y, s=14, color='tab:purple', alpha=0.7)
    if len(x) > 1:
        r = np.corrcoef(x, y)[0, 1]
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color='grey', linestyle='dotted', linewidth=1)
        ax.set_title(f'r = {r:.2f}', fontsize=10)
    ax.set_xlabel('Split A', fontsize=8)


def _panel_reliability_bars(ax, result):
    labels = ['Raw', 'Residual']
    values = [_z_avg(result['raw_r_rand']), _z_avg(result['resid_r_rand'])]
    colors = ['tab:gray', 'tab:red']
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color='black', linewidth=1, alpha=0.6)
    ax.tick_params(axis='x', labelsize=8)


def plot_consensus_profile_check(expt, results, path):
    _genericness_grid_figure(expt, results, _panel_consensus,
                              'consensus profile check (split A vs split B)',
                              f'{expt}_consensus_profile_check.png', path)


def plot_genericness_distribution(expt, results, path):
    _genericness_grid_figure(expt, results, _panel_genericness_dist,
                              'genericness score distribution',
                              f'{expt}_genericness_distribution.png', path)


def plot_genericness_reliability_random(expt, results, path):
    _genericness_grid_figure(
        expt, results,
        lambda ax, result: _panel_reliability_scatter(ax, result['genericness_a'], result['genericness_b']),
        'genericness reliability - random split',
        f'{expt}_genericness_reliability_random.png', path)


def plot_genericness_reliability_category(expt, results, path):
    _genericness_grid_figure(
        expt, results,
        lambda ax, result: _panel_reliability_scatter(ax, result['genericness_cat_a'], result['genericness_cat_b']),
        'genericness reliability - category split (1st half vs 2nd half)',
        f'{expt}_genericness_reliability_category.png', path)


def plot_residual_reliability(expt, results, path):
    _genericness_grid_figure(expt, results, _panel_reliability_bars,
                              'raw vs. residualized split-half reliability (random split)',
                              f'{expt}_residual_reliability.png', path)


def graph_genericness(path):
    genericness_path = path / 'genericness'
    genericness_path.mkdir(parents=True, exist_ok=True)
    for expt in EXPTS:
        results = compute_genericness_results(expt)
        plot_consensus_profile_check(expt, results, genericness_path)
        plot_genericness_distribution(expt, results, genericness_path)
        plot_genericness_reliability_random(expt, results, genericness_path)
        plot_genericness_reliability_category(expt, results, genericness_path)
        plot_residual_reliability(expt, results, genericness_path)


def main():
    _, graph_path = manage_path()
    graph(graph_path)
    graph_genericness(graph_path)


if __name__ == "__main__":
    main()
