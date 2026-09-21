from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from itertools import combinations, permutations
from scipy.stats import sem, wasserstein_distance
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pingouin as pg

from indimap.util import map_func, stat_func

# Editable text (not outlined) when the PDF is opened in Illustrator.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

_COLORS = plt.cm.get_cmap('Set1', 8)


def _save_pdf(path, dpi=384):
    """ Save the current figure as an Illustrator-editable PDF: no per-artist
    clip paths (avoids clipping-mask groups) and closes the figure after. """
    fig = plt.gcf()
    for ax in fig.axes:
        for artist in ax.get_children():
            artist.set_clip_on(False)
    fig.savefig(path, dpi=dpi, transparent=True)
    plt.close(fig)


def _style_boxplot(box, color, linewidth=2.5):
    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_linewidth(0)
    for whisker in box['whiskers']:
        whisker.set_color(color)
        whisker.set_linewidth(linewidth)
        whisker.set_alpha(1)
    for cap in box['caps']:
        cap.set_color(color)
        cap.set_linewidth(linewidth)
        cap.set_alpha(1)
    for median in box['medians']:
        median.set_color(color)
        median.set_linewidth(linewidth)
        median.set_alpha(1)


# Plain-text (non-mathtext) scientific notation for stats annotations.
# Mathtext ($...$) fonts don't survive pdf.fonttype=42 embedding correctly in
# Illustrator (glyphs render broken/missing), so annotations are built as
# ordinary text instead of LaTeX-style math strings. Unicode superscript
# digits (e.g. "10⁻⁴") are avoided too - they're missing from this project's
# font stack and would render as empty boxes; plain caret notation has no
# font-coverage risk.
def _sci_notation(coefficient, power):
    return f"{coefficient:.2f} × 10^{power}"


def _format_pval(p_val, threshold=1e-3):
    """ 'p = 1.23 × 10^-4'-style label, or fixed-point above threshold. """
    if p_val <= 0:
        p_val = np.finfo(float).tiny  # guard against log10(0) on an underflowed p-value
    if p_val < threshold:
        power = int(np.floor(np.log10(p_val)))
        coefficient = p_val / (10 ** power)
        return f"p = {_sci_notation(coefficient, power)}"
    return f"p = {p_val:.3f}"


def _format_pval_simple(p_val, threshold=0.001):
    if p_val < threshold:
        return f'p < {threshold}'
    return f'p = {p_val:.3f}'


def _format_pval_print(p_val, decimals=4, threshold=1e-4):
    """ Fixed-point at `decimals`, or the exact value in scientific notation below `threshold`. """
    if p_val < threshold:
        return f"{p_val:.6e}"
    return f"{p_val:.{decimals}f}"


def _stars_for_pval(p_val):
    """ Significance stars + a de-emphasis alpha for non-significant results. """
    if p_val < 1e-3:
        return '***', 1
    elif p_val < 0.01:
        return '**', 1
    elif p_val < 0.05:
        return '*', 1
    return 'n.s.', 0.5


def _bayes_factors(a, b, paired):
    try:
        bf10 = float(pg.ttest(a, b, paired=paired)['BF10'].values[0])
    except Exception:
        bf10 = np.nan
    return bf10, 1 / bf10


def _annotate_bracket(x1, x2, y, text, ax=None, alpha=1.0, fontsize=10, xytext=(0, 1), y_text_offset=0.0):
    if ax is None:
        ax = plt.gca()
    ax.plot([x1, x2], [y, y], color='black', linewidth=1.5, alpha=alpha)
    ax.annotate(text, ((x1 + x2) / 2, y + y_text_offset), textcoords="offset points", xytext=xytext,
                ha='center', size=fontsize, alpha=alpha)


def _fit_lim_to_text(ax=None, axis='x', pad=0.3, iterations=4):
    """ Widen the given axis' limit (from its current value) so no
    Text/Annotation artist - e.g. a stats bracket p-value label - is cut off
    at the edges. Widening the range changes the data-units-per-pixel scale,
    so a single measurement under-corrects; a few iterations converge to a
    stable, tight fit (no more padding than the text actually needs). """
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    get_lim, set_lim = (ax.get_xlim, ax.set_xlim) if axis == 'x' else (ax.get_ylim, ax.set_ylim)
    lo_attr, hi_attr = ('x0', 'x1') if axis == 'x' else ('y0', 'y1')
    for _ in range(iterations):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lo, hi = get_lim()
        for txt in ax.texts:
            bbox = txt.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
            lo = min(lo, getattr(bbox, lo_attr))
            hi = max(hi, getattr(bbox, hi_attr))
        set_lim(lo - pad, hi + pad)


def plot_wasserstein(data, name, path):
    n_maps, n_metrics, n_subjs, n_trials = data.shape
    n_pairs = n_subjs * (n_subjs - 1) // 2
    dists = np.empty((n_maps, n_metrics, n_pairs))

    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            if met_idx > 0:
                vec = data[map_idx, met_idx]
                vec = vec / np.mean(vec)
                data[map_idx, met_idx] = vec

    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            specs = []
            for i, j in combinations(range(n_subjs), 2):
                dist = wasserstein_distance(
                    data[map_idx, met_idx, i],
                    data[map_idx, met_idx, j]
                )
                specs.append(dist)
            dists[map_idx, met_idx] = specs

    print('Wasserstein distances:')
    print(dists.mean(axis=2))
    print(dists.std(axis=2))

    plt.figure(figsize=(5, 3))

    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            x_pos = met_idx * 4 + map_idx * 0.8
            box = plt.boxplot(dists[map_idx, met_idx], positions=[x_pos], widths=0.6, patch_artist=True,
                        showfliers=False)
            _style_boxplot(box, _COLORS(map_idx))

    plt.xticks([1.2, 5.2, 8.4],
               ['Accuracy', 'Confidence', 'RT'],
               fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Wasserstein distance', fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc='upper left', fontsize=12, frameon=False)
    plt.tight_layout()
    _save_pdf(path / f"wasserstein_{name}.pdf")


def plot_raincloud(data, name, path):
    n_maps = len(data) + 1
    n_conds, n_metrics, n_subjs, n_imgs = data[0].dims_map.human_arr.shape
    data_arr = np.empty(shape=(n_maps, n_metrics, n_subjs, n_imgs*n_conds))
    data_arr.fill(np.nan)

    for map_idx in range(n_maps):
        # avg across conditions
        if map_idx == 0:
            data_arr[0] = data[map_idx].dims_map.human_arr.transpose(1, 2, 0, 3).reshape(n_metrics, n_subjs, n_imgs*n_conds if n_conds > 1 else n_imgs)
        else:
            try:
                data_arr[map_idx] = data[map_idx-1].dims_map.model_arr.transpose(1, 2, 0, 3).reshape(n_metrics, n_subjs, n_imgs*n_conds if n_conds > 1 else n_imgs)
            except ValueError:
                data_arr[map_idx, :2] = data[map_idx-1].dims_map.model_arr.transpose(1, 2, 0, 3).reshape(2, n_subjs, n_imgs*n_conds if n_conds > 1 else n_imgs)

    plot_wasserstein(data_arr, name, path)

    data_arr = np.nanmean(data_arr, axis = 3) # average across trials
    # normalized confidence and rt by average before plotting
    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            if met_idx > 0:
                vec = data_arr[map_idx, met_idx]
                vec = vec / np.mean(vec)
                data_arr[map_idx, met_idx] = vec

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']

    figure, ax = plt.subplots(1, n_metrics, figsize=(8, 3))
    met_titles = ['Accuracy', 'Confidence', 'RT']
    met_labels = ['Accuracy', 'Normalized confidence', 'Normalized RT']
    colors = _COLORS

    for met_idx in range(n_metrics):
        plot_data = []
        plot_labels = []
        for map_idx in range(n_maps):
            if met_idx == 2 and map_idx > 1:
                continue
            plot_data.extend(data_arr[map_idx, met_idx])
            plot_labels.extend([model_labels[map_idx]] * len(data_arr[map_idx, met_idx]))
        df_plot = pd.DataFrame({'value': plot_data, 'model': plot_labels})

        # Raincloud plot: violin + strip
        sns.violinplot(
            y='model',
            x='value',
            data=df_plot,
            ax=ax[met_idx],
            inner=None,
            palette=list(colors.colors) if hasattr(colors, 'colors') else colors.colors,
            alpha=0.5,
            linewidth=0
        )

        for collection in ax[met_idx].collections:
            _paths = collection.get_paths()[0]
            vertices = _paths.vertices
            mean_y = vertices[:, 1].mean()
            vertices[vertices[:, 1] > mean_y, 1] = mean_y  # clip below mean y

        for map_idx in range(n_maps):
            if met_idx == 2 and map_idx > 1:
                continue
            y_buffer = map_idx + (np.random.rand(len(data_arr[map_idx, met_idx])) - 0.5) * 0.15
            ax[met_idx].scatter(
                data_arr[map_idx, met_idx],
                y_buffer + 0.15,
                s = 1,
                color = colors(map_idx),
                alpha=0.75,
                edgecolor=None
            )

        sns.pointplot(
            y='model',
            x='value',
            data=df_plot,
            ax=ax[met_idx],
            join=False,              # Don't connect the points
            ci='sd',                 # Show standard deviation as error bars
            color='black',           # Color of points and bars
            errwidth=1.5,            # Width of error bars
            markers='D',              # Marker style 'D' for diamond
            markersize=2
        )

        ax[met_idx].set_ylabel("", fontsize=6)
        ax[met_idx].set_title(met_titles[met_idx], fontsize=10, fontweight='bold')
        ax[met_idx].tick_params(axis='y', labelsize=6)
        ax[met_idx].tick_params(axis='x', labelsize=6)
        ax[met_idx].spines['top'].set_visible(False)
        ax[met_idx].spines['right'].set_visible(False)
        ax[met_idx].spines['left'].set_visible(False)
        ax[met_idx].set_xlabel(met_labels[met_idx], fontsize=8, fontweight='bold')

        if met_idx < 2:
            ax[met_idx].set_yticks([x for x in range(n_maps)])
            ax[met_idx].set_yticklabels(model_labels, fontsize=8)
        else:
            ax[met_idx].set_yticks([x for x in range(2)])
            ax[met_idx].set_yticklabels(model_labels[:2], fontsize=8)

    plt.tight_layout()
    _save_pdf(path / f"raincloud_{name}.pdf")


def mapping_matrix(arr1, arr2):

    same = np.array_equal(arr1, arr2)

    if same:
        output = np.zeros((
                           arr1.shape[0], arr1.shape[1], 
                           arr1.shape[2], arr1.shape[2] - 1
                           ))
    else:
        output = np.zeros((
                           arr1.shape[0], arr1.shape[1], 
                           arr1.shape[2], arr1.shape[2]
                           ))

    for k in range(arr1.shape[0]):
        for l in range(arr1.shape[1]):
            result = map_func.compute_full_corr_matrix(
                arr1[k, l, :, :],
                arr2[k, l, :, :],
                )
            if same:
                np.fill_diagonal(result, np.nan)
                result = result[~np.isnan(result)]
                result = result.reshape(arr1.shape[2], arr1.shape[2]-1)
            output[k,l,:,:] = result

    output = stat_func.r2z(output, 'pearson')
    output = np.mean(output, axis=0)
    output = stat_func.z2r(output, 'pearson')
    return output


def plot_raw_matrix(data, name, path):
    # compute raw matrix
    n_maps = len(data) + 1
    n_conds, n_metrics, n_subjs, n_imgs = data[0].dims_map.human_arr.shape
    data_list = []
    for map_idx in range(n_maps):
        if map_idx == 0:
            arr = data[map_idx].dims_map.human_arr
        else:
            arr = data[map_idx-1].dims_map.model_arr
        data_list.append(arr)

    results = np.empty(shape=(n_maps, n_metrics, n_subjs, n_subjs))
    results.fill(np.nan)
    for map_idx in range(n_maps):
        if map_idx < 2:
            arr1 = data_list[0]
            arr2 = data_list[map_idx]
        else:
            arr1 = data_list[0][:, [0,1], :, :] # for ecoset10, refactor later
            arr2 = data_list[map_idx]
        result = mapping_matrix(arr1, arr2)

        if map_idx == 0:  # Subject add diagnoal
            mat_with_diag = np.zeros((n_metrics, n_subjs, n_subjs))
            for met_idx in range(n_metrics):
                mat_with_diag[met_idx, np.arange(n_subjs), np.arange(n_subjs)] = 1.0
                for row in range(n_subjs):
                    mat_with_diag[met_idx, row, :row] = result[met_idx, row, :row]
                    mat_with_diag[met_idx, row, row+1:] = result[met_idx, row, row:]
            result = mat_with_diag

        try:
            results[map_idx] = result
        except ValueError:
            results[map_idx, :2] = result

    xlabels = ['Subjects', 'RTNet', 'AlexNet', 'ResNet18']
    metrics = ['acc', 'conf', 'rt']
    all_counts = np.zeros((n_metrics, n_maps, n_subjs))

    for metric in range(n_metrics):
        fig, ax = plt.subplots(2, n_maps, figsize=(20, 7),
                                gridspec_kw={'height_ratios': [4, 1]}
                            )
        plt.subplots_adjust(hspace=0.001, wspace=0.01)
        counts = np.zeros((n_maps, n_subjs))

        for map_idx in range(n_maps):
            im = ax[0, map_idx].imshow(
                results[map_idx, metric],
                cmap='seismic',
                vmin=-1, vmax=1,
            )
            for row in range(n_subjs):
                if map_idx == 0:
                    sorted_idx = np.argsort(results[map_idx, metric, row, :])
                    best_index = sorted_idx[-2]
                else:
                    best_index = np.argmax(results[map_idx, metric, row, :])
                ax[0, map_idx].scatter(best_index, row, facecolors='none',
                                   edgecolor='black', s=5, linewidth=1.5
                                   )
                counts[map_idx, best_index] += 1

            ax[0, map_idx].set_xticks([], [])
            ax[0, map_idx].set_yticks([], [])
            ax[0, map_idx].spines['top'].set_visible(False)
            ax[0, map_idx].spines['right'].set_visible(False)
            ax[0, map_idx].spines['left'].set_visible(False)
            ax[0, map_idx].spines['bottom'].set_visible(False)

            ax[1, map_idx].bar(np.arange(n_subjs), 
                               counts[map_idx], 
                               color='#A30E12', 
                               alpha=0.75)
            ax[1, map_idx].set_xlim(-0.5, n_subjs - 0.5)
            ax[1, map_idx].spines['top'].set_visible(False)
            ax[1, map_idx].spines['right'].set_visible(False)

            ax[0, map_idx].set_ylabel('Subjects', fontsize=28, fontweight='bold')

            if map_idx == 0:
                ax[0, map_idx].set_title('Human subjects', fontsize=28, fontweight='bold')
                ax[1, map_idx].set_xlabel(xlabels[map_idx], fontsize=28, fontweight='bold')
            else:
                ax[0, map_idx].set_title(xlabels[map_idx], fontsize=28, fontweight='bold')
                ax[1, map_idx].set_xlabel(xlabels[map_idx] + ' instances', fontsize=28, fontweight='bold')

            ax[1, map_idx].set_ylabel('Count', fontsize=28, fontweight='bold')
            ax[1, map_idx].set_yticks([], [])
            ax[1, map_idx].set_xticks([], [])

        all_counts[metric] = counts

        plt.tight_layout()
        _save_pdf(path / f'raw_sim_map_{metrics[metric]}_{name}.pdf')
    return results, all_counts


def plot_raw_matrix_colorbar(path):
    """ Standalone colorbar for the 'seismic', vmin=-1/vmax=1 imshow panels
    drawn in plot_raw_matrix, for compositing in Illustrator. """
    fig, ax = plt.subplots(figsize=(1.2, 6))
    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap='seismic')
    cbar = fig.colorbar(mappable, cax=ax)
    cbar.set_label('Correlation coefficient', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    _save_pdf(path / 'raw_sim_map_colorbar.pdf')


def _best_match_mean_counts(data):
    """
    Best-match count, averaged across the 1000 random image-split
    iterations x 2 halves = 2000 subsets (i.e. mean of IndiMap's get_top_ct
    over its bootstrap and split axes) for each map (Subject, RTNet, AlexNet,
    ResNet18) and behavioral metric.

    Returns: (results, n_maps, n_metrics, n_subjs), results shape
    (n_maps, n_metrics, n_subjs), NaN where a map/metric has no data (e.g.
    the human map's self-match slot, or an architecture with no RT metric).
    """
    n_maps = len(data) + 1
    _, _, n_metrics, n_subjs = data[0].get_top_ct('subj', 'inst').mat.shape
    results = np.empty(shape=(n_maps, n_metrics, n_subjs))
    results.fill(np.nan)

    for map_idx in range(n_maps):
        if map_idx == 0:
            results[map_idx, :, :n_subjs - 1] = data[map_idx].get_top_ct('subj', 'subj').mat.mean(axis=(0, 1))
        else:
            try:
                results[map_idx, :, :] = data[map_idx - 1].get_top_ct('subj', 'inst').mat.mean(axis=(0, 1))
            except ValueError:
                results[map_idx, :2, :] = data[map_idx - 1].get_top_ct('subj', 'inst').mat.mean(axis=(0, 1))

    return results, n_maps, n_metrics, n_subjs


def plot_best_match_freq_raw(data, name, path):
    """ Best-match frequency distribution: mean best-match count across the
    1000 random image-split iterations x 2 halves = 2000 subsets, sorted in
    descending order per map/metric. Subject/RTNet/AlexNet/ResNet18 shown as
    separate colored lines, one subplot per behavioral metric. """
    results, n_maps, n_metrics, n_subjs = _best_match_mean_counts(data)
    met_labels = ['Accuracy', 'Confidence', 'RT']
    map_labels = ['Subject', 'RTNet', 'AlexNet', 'ResNet18']
    colors = _COLORS

    fig, ax = plt.subplots(1, n_metrics, figsize=(9, 2.5))
    for map_idx in range(n_maps):
        for met in range(n_metrics):
            row = results[map_idx, met, :]
            if np.all(np.isnan(row)):
                continue
            sorted_data = np.sort(row)[::-1]
            sorted_data = sorted_data[~np.isnan(sorted_data)]
            denom = (n_subjs - 1) if map_idx == 0 else n_subjs
            x_pos = np.arange(1, len(sorted_data) + 1)

            ax[met].plot(x_pos, sorted_data / denom,
                         color=colors(map_idx),
                         label=map_labels[map_idx] if met == 0 else None,
                         alpha=1, lw=3, zorder=99 if map_idx == 0 else 1,
                         )

            ax[met].set_xlim(-0.5, n_subjs + 1.5)
            ax[met].set_xticks([], [])
            ax[met].tick_params(axis='y', labelsize=8)
            ax[met].set_xlabel('Subjects/Instances', fontsize=10, fontweight='bold')
            ax[met].set_ylabel('Best-matched frequency', fontsize=10)
            ax[met].spines['top'].set_visible(False)
            ax[met].spines['right'].set_visible(False)
            ax[met].set_title(met_labels[met], fontsize=12, fontweight='bold')

    fig.legend(*ax[0].get_legend_handles_labels(), loc='upper right', fontsize=8, frameon=False)
    plt.tight_layout()
    _save_pdf(path / f'best_match_freq_raw_{name}.pdf')


def plot_best_match_freq_fit(data, name, path):
    """ Exponential decay fit, f(x) = A * e^(-lambda * x), of the best-match
    frequency distribution shown in plot_best_match_freq_raw (a single fit
    per map/metric on the mean-across-2000-subsets, sorted curve), with the
    decay rate (lambda) annotated per line. """
    results, n_maps, n_metrics, n_subjs = _best_match_mean_counts(data)
    met_labels = ['Accuracy', 'Confidence', 'RT']
    map_labels = ['Subject', 'RTNet', 'AlexNet', 'ResNet18']
    colors = _COLORS

    fig, ax = plt.subplots(1, n_metrics, figsize=(9, 2.5))
    for map_idx in range(n_maps):
        for met in range(n_metrics):
            row = results[map_idx, met, :]
            if np.all(np.isnan(row)):
                continue
            sorted_data = np.sort(row)[::-1]
            sorted_data = sorted_data[~np.isnan(sorted_data)]
            denom = (n_subjs - 1) if map_idx == 0 else n_subjs
            freq = sorted_data / denom

            a, b = stat_func.fit_expo(freq)
            if a is None:
                continue
            x_pos = np.arange(1, len(freq) + 1)
            fit_curve = stat_func.exponential_func(x_pos - 1, a, b)
            lam = -b

            ax[met].plot(x_pos, fit_curve,
                         color=colors(map_idx),
                         label=map_labels[map_idx] if met == 0 else None,
                         alpha=1, lw=3, zorder=99 if map_idx == 0 else 1,
                         )
            ax[met].annotate(f'λ={lam:.3f}',
                             xy=(x_pos[0], fit_curve[0]),
                             xytext=(8, -10 * map_idx),
                             textcoords='offset points',
                             color=colors(map_idx), fontsize=8, fontweight='bold',
                             )

            ax[met].set_xlim(-0.5, n_subjs + 1.5)
            ax[met].set_xticks([], [])
            ax[met].tick_params(axis='y', labelsize=8)
            ax[met].set_xlabel('Subjects/Instances', fontsize=10, fontweight='bold')
            ax[met].set_ylabel('Best-matched frequency', fontsize=10)
            ax[met].spines['top'].set_visible(False)
            ax[met].spines['right'].set_visible(False)
            ax[met].set_title(met_labels[met], fontsize=12, fontweight='bold')

    fig.legend(*ax[0].get_legend_handles_labels(), loc='upper right', fontsize=8, frameon=False)
    plt.tight_layout()
    _save_pdf(path / f'best_match_freq_fit_{name}.pdf')


def report_best_match_decay_significance(data, name):
    """
    Non-parametric bootstrap test comparing best-match frequency decay rates
    (lambda) between humans and each ANN architecture.

    Reuses the split-half exponential fits IndiMap already computes for each
    model (1000 random image-split iterations x 2 halves = 2000 subsets, via
    get_top_expo). IndiMap builds 'subj_to_subj' and 'subj_to_inst' with the
    same bootstrap seed, so they are generated from the identical sequence of
    image splits: the human lambda and the ANN lambda at a given repetition
    are paired, not independent draws. Each of the 2000 subsets is therefore
    itself one bootstrap sample, with statistic diff = lambda_human -
    lambda_ANN for that subset. The two-sided p-value is the proportion of
    these 2000 paired differences that fall on the opposite side of zero from
    the observed mean, and the 95% CI is their [2.5, 97.5] percentile.
    """
    met_labels = ['Accuracy', 'Confidence', 'RT']
    map_labels = ['RTNet', 'AlexNet', 'ResNet18']

    human_lambda = -data[0].get_top_expo('subj', 'subj').mat[..., 1]

    for map_idx, map_label in enumerate(map_labels):
        ann_lambda = -data[map_idx].get_top_expo('subj', 'inst').mat[..., 1]
        n_metrics = min(human_lambda.shape[1], ann_lambda.shape[1])

        for met in range(n_metrics):
            diffs = human_lambda[:, met] - ann_lambda[:, met]
            diffs = diffs[~np.isnan(diffs)]
            if len(diffs) == 0:
                continue

            observed_diff = np.mean(diffs)
            ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
            n_cross = min(np.sum(diffs <= 0), np.sum(diffs >= 0))
            p_val = min(2 * n_cross / len(diffs), 1.0)
            # A count of 0 doesn't mean p==0, only that no subset crossed zero at this
            # resolution - report the detection floor (2/n) as an upper bound instead.
            p_str = f"< {2 / len(diffs):.4g}" if n_cross == 0 else _format_pval_print(p_val, 4)

            print(f"[{name}] Decay rate (lambda) Human vs {map_label}, {met_labels[met]}: "
                  f"mean diff={observed_diff:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}], "
                  f"p={p_str}")


def plot_alignment_average(data, name, path):
    plot_avg_data = data.copy()
    if name == 'mnist':
        plt.figure(figsize=(5, 4))
    else:
        plt.figure(figsize=(6, 4))
    colors = _COLORS
    n_maps, n_metrics, n_subjs, _ = plot_avg_data.shape

    map_labels = ['Human-Human',
                'Human-RTNet',
                'Human-AlexNet',
                'Human-ResNet18',
                ]
    plt.ylim(-0.05, 0.9)
    plt.xlim(-1.5, 10.5)
    # y_poss = [0.15, 0.15, 0.15]
    y_poss = [0.18, 0.20, 0.15]

    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 4 + map * 0.8
            if map == 0:
                plot_avg_data[map, met][plot_avg_data[map, met] == 1] = np.nan
            if map > 1 and met == 2:
                continue

            avg_data = np.nanmean(plot_avg_data[map, met, :, :], axis=1)
            print(map, met)
            print(np.nanmean(avg_data))

            box = plt.boxplot(avg_data, positions=[x_pos], widths=0.4, patch_artist=True,
                              showfliers=False)
            _style_boxplot(box, colors(map))

            for subj in range(n_subjs):
                plt.scatter(x_pos-0.3, avg_data[subj], color=colors(map), alpha=0.75, s = 10)

    stat_data = stat_func.r2z(plot_avg_data, metric='pearson')
    for map in range(n_maps):
        for met in range(n_metrics):
            if map == 0:
                stat_data[map, met][stat_data[map, met] == 10] = np.nan

    stat_data = np.nanmean(stat_data, axis = -1)
    for i, j in combinations(range(n_maps), 2):
        for met in range(n_metrics):
            scipy_results = stats.ttest_rel(stat_data[i, met], stat_data[j, met])
            p_val = scipy_results.pvalue

            if i == 0:
                if met == 2 and not (j < 2):
                    plot_avg_data[map, met] = np.nan
                    continue
                y_max = np.nanmean(plot_avg_data[:, met]) + y_poss[met] + 0.06 * j
                _annotate_bracket(met * 4, met * 4 + j * 0.8, y_max, _format_pval(p_val), y_text_offset=0.01)

    plt.xticks([1.2, 5.2, 8.4], 
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
                )
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Correlation coefficient', fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc='upper right', fontsize=10, frameon=False)
    plt.tight_layout()
    _save_pdf(path / f'align_avg_{name}.pdf')


def plot_alignment_variance(data, name, path):
    n_maps = len(data) + 1
    n_boots, n_splits, n_metrics, n_subjs, _ = data[0].get_corr_map('subj', 'subj').mat.shape
    results = np.empty(shape=(n_maps, n_boots, n_splits, n_metrics, n_subjs))
    results.fill(np.nan)

    for i in range(n_maps):
        if i == 0:
            result = data[i].get_corr_map('subj', 'subj').mat
        else:
            result = data[i-1].get_corr_map('subj', 'inst').mat
        result = stat_func.r2z(result, metric='pearson')
        result = np.std(result, axis = (4))
        try:
            results[i] = result
        except ValueError:
            results[i, :, :, :2] = result
    
    results = results.reshape(n_maps, n_boots * n_splits, n_metrics, n_subjs)

    if name == 'mnist':
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        plt.ylim(0, 0.3)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.ylim(0, 0.25)
    colors = _COLORS

    map_labels = ['Human-Human',
                'Human-RTNet',
                'Human-AlexNet',
                'Human-ResNet18'
                ]

    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 4 + map * 0.8
            avg = np.mean(results[map, :,  met], axis=0)
            box = plt.boxplot(avg, positions=[x_pos], widths=0.4, patch_artist=True,
                                showfliers=False)
            _style_boxplot(box, colors(map))

            for subj in range(n_subjs):
                plt.scatter(x_pos-0.3, avg[subj], color=colors(map), alpha=0.75, s = 10)

    var_p_values = np.empty((n_metrics, n_maps, n_maps))
    for met in range(n_metrics):
        met_data = results[:, :, met]
        for i, j in combinations(range(n_maps), 2):
            diff = met_data[i] - met_data[j]
            p_val = 2 * min(np.mean(diff < 0), np.mean(diff > 0))
            var_p_values[met, i, j] = p_val
            ci_lower = np.percentile(diff, 2.5)
            ci_upper = np.percentile(diff, 97.5)

            if (map_labels[i] == 'Human-Human'):
                if met == 2 and not (j == 1):
                    continue
                print(f'{map_labels[i]} vs {map_labels[j]} - p-value: {_format_pval_print(p_val, 4)}, CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
                y_max = np.nanmax(np.nanmean(results[:, :, met], axis=1)) + 0.02 * j
                anno = 'p < 0.0005' if p_val < 0.001 else 'p = {:.3f}'.format(p_val)
                _annotate_bracket(met * 4, met * 4 + j * 0.8, y_max, anno, y_text_offset=0.001)

    plt.xticks([1.2, 5.2, 8.4], 
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
                )
    plt.xlim(-1.5, 10.5)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Standard deviation', fontsize=14)
    # plt.title('Variance', fontsize=16, fontweight='bold')  
    if name == 'mnist':
        plt.legend(loc='upper right', fontsize=10, frameon=False)
    else:
        plt.legend(loc='upper left', fontsize=10, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'align_var_{name}.pdf')


def plot_across_metric_illustration(data, name, path):
    if name == 'mnist':
        indicate_best_match = True
    else:
        indicate_best_match = False
    
    n_maps, n_metrics, n_subj, _ = data.shape
    corr_consistency = np.full((n_maps, n_metrics, n_subj), np.nan)
    stats_results = np.full((n_maps, n_metrics), np.nan)
    stat_data = stat_func.r2z(data, metric='pearson')
    colors = _COLORS

    model_labels = ['Human-Human', 'Human-RTNet', 'Human-AlexNet', 'Human-ResNet18']
    met_labels = ['Accuracy\nConfidence', 'Accuracy\nRT', 'RT\nConfidence']
    metric_labels = ['Accuracy alignment', 'Confidence alignment', 'RT alignment']
    fig, ax = plt.subplots(3, 4, figsize=(8, 5))

    for i in range(n_maps):
        for j, (met1, met2) in enumerate(combinations(range(n_metrics), 2)):
            for k in range(n_subj):
                met1_data = stat_data[i, met1, k]
                met2_data = stat_data[i, met2, k]
                mask = np.isfinite(met1_data) & np.isfinite(met2_data)
                corr_consistency[i, j, k] = np.corrcoef(met1_data[mask], met2_data[mask])[0, 1]

    best_match = np.argmax(corr_consistency[1, 0])  # Best match for RTNet in the first metric (Accuracy)

    for i in range(n_maps):
        for j, (met1, met2) in enumerate(combinations(range(n_metrics), 2)):
            if i > 1 and j > 0:
                ax[j, i].set_visible(False)
                continue
            met1_data = stat_data[i, met1, best_match]
            met2_data = stat_data[i, met2, best_match]
            mask = np.isfinite(met1_data) & np.isfinite(met2_data)
            ax[j, i].scatter(met1_data[mask], met2_data[mask], 
                        color=colors(i), alpha=0.5, s=20, label=model_labels[i] if j == 0 else None,
                            edgecolor='none'
                        )
            slope, intercept, r_value, p_value, std_err = stats.linregress(met1_data[mask], met2_data[mask])
            x_fit = np.linspace(np.min(met1_data[mask]), np.max(met1_data[mask]), 100)
            y_fit = slope * x_fit + intercept
            ax[j, i].plot(x_fit, y_fit, color=colors(i), linewidth=1.5, alpha=0.75)
            ax[j, i].fill_between(x_fit, y_fit - std_err, y_fit + std_err, color=colors(i), alpha=0.1, edgecolor='none')
            ax[j, i].set_xlabel(metric_labels[met1], fontsize=8, fontweight='bold')
            ax[j, i].set_ylabel(metric_labels[met2], fontsize=8, fontweight='bold')
            ax[j, i].tick_params(axis='x', labelsize=6)
            ax[j, i].tick_params(axis='y', labelsize=6)
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            if j == 0:
                ax[j, i].set_title(f'{model_labels[i]}', fontsize=10, fontweight='bold')

            if p_value < 0.001:
                power = int(np.floor(np.log10(p_value)))
                coefficient = p_value / (10 ** power)
                p_value_str = _sci_notation(coefficient, power)
            else:
                p_value_str = f"{p_value:.3f}"
            ax[j, i].set_ylim(np.min(met2_data[mask]-0.05), np.max(met2_data[mask]+0.18))
            # Annotate correlation and p-value in upper left corner
            ax[j, i].annotate(f'r = {r_value:.2f};  p = {p_value_str}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=7, ha='left', va='top',
            color=colors(i))
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_var_illustration_{name}.pdf')

    if name == 'mnist':
        plt.figure(figsize=(4.5, 3))
    else:
        plt.figure(figsize=(4, 4))

    for i in range(n_maps):
        for j, (met1, met2) in enumerate(combinations(range(n_metrics), 2)):
            if i > 1 and j > 0:
                continue
            if j < 2:
                x_pos = j * 4 + i * 0.8
            else:
                x_pos = j * 3.2 + i * 0.8

            box = plt.boxplot(corr_consistency[i, j], positions=[x_pos], widths=0.4, patch_artist=True,
                                showfliers=False)
            _style_boxplot(box, colors(i))

            for subj in range(n_subj):
                if subj == best_match and indicate_best_match:
                    color='black'
                    plt.scatter(x_pos-0.3, corr_consistency[i, j, subj], color=color, alpha=1, s = 50, zorder=2, marker='*',
                                label='Illustrated subject' if i == 0 and j == 0 else None)
                else:
                    color = colors(i)
                    plt.scatter(x_pos-0.3, corr_consistency[i, j, subj], color=color, alpha=0.75, s = 10,
                                edgecolor='none')

    consistency_stat_data = stat_func.r2z(corr_consistency, metric='pearson')  # Convert to z-scores
    for i, j in combinations(range(n_maps), 2):
        for k in range(n_metrics):
            if i != 0:
                continue
            if j > 1 and k > 0:
                continue

            if name == 'mnist':
                stat_results = stats.ttest_rel(consistency_stat_data[i, k], consistency_stat_data[j, k])
                bf10, bf01 = _bayes_factors(consistency_stat_data[i, k], consistency_stat_data[j, k], paired=True)
                print(f"Model: {model_labels[i]}, Comparison: {model_labels[j]}, Metric: {k}, BF10: {bf10}, BF01: {bf01}")
                print("T-test results:", stat_results)
                p_val = stat_results.pvalue
                if k < 2:
                    plot_x_pos = [k * 4, k * 4 + j * 0.8]
                else:
                    plot_x_pos = [k * 3.2, k * 3.2 + j * 0.8]
                y_max = np.nanmax(corr_consistency[:, k]) + 0.2 * j
                _annotate_bracket(plot_x_pos[0], plot_x_pos[1], y_max, _format_pval(p_val), y_text_offset=0.02)
                plt.ylim(-0.4, 1.95)

            if name == 'ecoset10':
                one_sample_stats = stats.ttest_1samp(consistency_stat_data[j, k], 0)
                print()
                print(f"Model: {model_labels[j]}, Metric: {k}, average r: {np.nanmean(corr_consistency[j, k]):.3f}")
                print("One-sample t-test results:", one_sample_stats)
                print()
                if ((i == 0)):
                    if k != 0 and not (j == 1):
                        continue
                p_val = one_sample_stats.pvalue

                if k < 2:
                    x_pos = (k * 4 + j * 0.8)
                else:
                    x_pos = (k * 3.2 + j * 0.8)

                y_max = -0.325
                anno, alpha = _stars_for_pval(p_val)
                plt.annotate(anno, (x_pos, y_max), textcoords="offset points", xytext=(0, 1), ha='center', size=10, alpha=alpha, fontweight='bold')
                plt.ylim(-0.35, 0.9)

    plt.xticks([1.2, 4.4, 6.8], 
                ['Acc-Conf', 'Acc-RT', 'Conf-RT'],
                fontsize=12
                )
    plt.xlim(-1, 8.5)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Across-metric alignment', fontsize=14, fontweight='bold')
    plt.ylabel('Correlation coefficient', fontsize=14)
    plt.title('Across-metric alignment', fontsize=14, fontweight='bold')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.legend(loc='upper right', fontsize=8, frameon=False)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_var_correlation_{name}.pdf')


def plot_best_count_distribution(data, name, path):
    n_maps = len(data) + 1
    _, _, n_metrics, n_subjs = data[0].get_top_ct('subj', 'inst').mat.shape
    results = np.empty(shape=(n_maps, n_metrics, n_subjs))
    results.fill(np.nan)

    for map_idx in range(n_maps):
        if map_idx == 0:
            results[map_idx, :, :n_subjs - 1] = data[map_idx].get_top_ct('subj', 'subj').mat.mean(axis = (0, 1))
        else:
            try:
                results[map_idx, :, :] = data[map_idx-1].get_top_ct('subj', 'inst').mat.mean(axis = (0, 1))
            except ValueError:
                results[map_idx, :2, :] = data[map_idx-1].get_top_ct('subj', 'inst').mat.mean(axis = (0, 1))

    fig, ax = plt.subplots(1, n_metrics, figsize=(9, 2.5))
    colors = _COLORS
    met_labels = ['Accuracy', 'Confidence', 'RT']
    map_labels = ['Subject', 'RTNet', 'AlexNet', 'ResNet18']

    for map_idx in range(n_maps):
        for met in range(n_metrics):
            sorted_indices = np.argsort(results[map_idx, met, :])[::-1]
            sorted_data = results[map_idx, met, :][sorted_indices]
            x_pos = np.array([x for x in range(n_subjs)])
            lw = 3

            if map_idx == 0:
                ax[met].plot(x_pos + 1,
                sorted_data / (n_subjs - 1),
                color=colors(map_idx),
                label=map_labels[map_idx],
                alpha=1, lw=lw, zorder=99,
                             )

            else:
                ax[met].plot(x_pos + 0.25*map_idx,
                sorted_data / n_subjs,
                color=colors(map_idx),
                label=map_labels[map_idx],
                alpha=1, lw=lw
                             )

            ax[met].set_xlim(-0.5, n_subjs + 1.5)
            ax[met].set_xticks([], [])
            ax[met].tick_params(axis='y', labelsize=8)
            ax[met].set_xlabel('Subjects/Instances', fontsize=10, fontweight='bold')
            ax[met].set_ylabel('Best-matched frequency', fontsize=10)
            ax[met].spines['top'].set_visible(False)
            ax[met].spines['right'].set_visible(False)
            ax[met].set_title(met_labels[met], fontsize=12, fontweight='bold')
            ax[met].legend(loc='upper right', fontsize=8, frameon=False)

    plt.tight_layout()
    _save_pdf(path / f'top_count_dist_{name}.pdf')


def plot_top_identifiability(data, name, path):
    n_maps = len(data) + 1
    _, n_metrics, n_subjs = data[0].get_top_iden('subj', 'inst', 'pair').mat.shape

    plot_data = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)

    for type_idx, map_type in enumerate(['pair', 'gp']):
        for map_idx in range(n_maps):
            for met_idx in range(n_metrics):
                if map_idx == 0:
                    map_data = data[map_idx].get_top_iden('subj', 'subj', map_type).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    plot_data[type_idx, map_idx] = map_data
                else:
                    map_data = data[map_idx-1].get_top_iden('subj', 'inst', map_type).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    try:
                        plot_data[type_idx, map_idx] = map_data
                    except ValueError:
                        plot_data[type_idx, map_idx, :2] = map_data
    
    plot_data = -np.diff(plot_data, axis = 0)
    plot_data = np.squeeze(plot_data, axis = 0)

    plt.figure(figsize=(6, 4))

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS

    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson')
    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 4 + map * 0.8
            box = plt.boxplot(plot_data_in_r[map, met, :], positions=[x_pos], widths=0.4, patch_artist=True,
                              showfliers=False,
                        )
            _style_boxplot(box, colors(map))

            for k in range(n_subjs):
                plt.scatter(x_pos - 0.3,
                            plot_data_in_r[map, met, k],
                            color=colors(map), s=5
                            )

    data_min = np.nanmin(plot_data_in_r)
    data_max = np.nanmax(plot_data_in_r)
    data_range = data_max - data_min
    vs_zero_y = data_min - 0.15 * data_range
    max_bracket = data_max
    bracket_step = 0.09 * data_range  # compact stacking so brackets don't balloon the y-range

    for met in range(n_metrics):
        sub_data = plot_data[:, met, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):

                if name in ['mnist', 'ecoset10']:
                    t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                    bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                    mean_diff = np.nanmean(sub_data[i]) - np.nanmean(sub_data[j])
                    pooled_std = np.sqrt((np.nanvar(sub_data[i], ddof=1) + np.nanvar(sub_data[j], ddof=1)) / 2)
                    cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
                    # print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]}, "
                    #     f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                    #     f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 8)}, "
                    #     f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                    #     f"Cohen's d: {cohen_d:.4f}")

                    # Check for significance
                    if ((map_labels[i] == 'Human-Human')):
                        if met == 2 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        y_max = np.nanmax(plot_data_in_r[:, met, :]) + bracket_step * abs(j - i)
                        max_bracket = max(max_bracket, y_max)
                        alpha = 0.5 if p_val < 0.05 else 1
                        _annotate_bracket(met * 4 + i * 0.8, met * 4 + j * 0.8, y_max, _format_pval(p_val),
                                           alpha=alpha, fontsize=8, y_text_offset=0.006)
                        # plt.legend(loc='upper right', fontsize=8, frameon=False)

                if name in ['mnist', 'ecoset10']:
                    t_stat, p_val = stats.ttest_1samp(sub_data[j], 0, nan_policy='omit')
                    bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                    cohen_d = t_stat / np.sqrt(n_subjs) if n_subjs != 0 else np.nan
                    print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 8)}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # compare to 0 signfiicance
                    if ((map_labels[i] == 'Human-Human')):
                        if met == 2 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        x_pos = (met * 4 + j * 0.8) + 0.15
                        anno, alpha = _stars_for_pval(p_val)
                        plt.annotate(anno, (x_pos, vs_zero_y), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha, fontweight='bold')
                        # plt.legend(loc='upper left', fontsize=8, frameon=False)

    plt.xticks([1.2, 5.2, 8.4],
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
            )
    plt.xlim(-1, 10)
    plt.ylim(vs_zero_y - 0.05 * data_range, max_bracket + 0.05 * data_range)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('r(best pair) − r(other pairs)', fontsize=12, fontweight='bold')
    plt.title('Best-pair advantage', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _fit_lim_to_text(axis='x', pad=0.3)
    _fit_lim_to_text(axis='y', pad=0.015)
    plt.tight_layout()
    _save_pdf(path / f'top_btw_bs_{name}.pdf')


def plot_top_identifiability_raw(data, name, path):
    n_maps = len(data) + 1
    _, n_metrics, n_subjs = data[0].get_top_iden('subj', 'inst', 'pair').mat.shape

    plot_data = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)

    for type_idx, map_type in enumerate(['pair', 'gp']):
        for map_idx in range(n_maps):
            for met_idx in range(n_metrics):
                if map_idx == 0:
                    map_data = data[map_idx].get_top_iden('subj', 'subj', map_type).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    plot_data[type_idx, map_idx] = map_data
                else:
                    map_data = data[map_idx-1].get_top_iden('subj', 'inst', map_type).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    try:
                        plot_data[type_idx, map_idx] = map_data
                    except ValueError:
                        plot_data[type_idx, map_idx, :2] = map_data

    plot_data = stat_func.z2r(plot_data, metric='pearson')

    fig, ax = plt.subplots(1, n_metrics, figsize=(10.5, 3.5))
    colors = _COLORS
    model_label = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    pair_labels = ['Best pair', 'Other pairs']
    titles = ['Accuracy', 'Confidence', 'RT']

    for map_idx in range(n_maps):
        for met in range(n_metrics):
            for type_idx in range(2):
                x_pos = type_idx + map_idx * 3
                if not np.isnan(plot_data[type_idx, map_idx, met, :]).all():
                    ax[met].scatter(x_pos,
                        np.nanmean(plot_data[type_idx, map_idx, met, :]),
                        color=colors(map_idx), label=pair_labels[type_idx] if map_idx == 0 else None,
                        alpha=1, marker='s' if type_idx == 0 else 'D', s=50
                        )
                for k in range(n_subjs):
                    ax[met].scatter(x_pos,
                                plot_data[type_idx, map_idx, met, k],
                                color=colors(map_idx), s=1, alpha=0.5
                                )
                    if type_idx == 0:
                        ax[met].plot([x_pos, x_pos + 1],
                                    [plot_data[type_idx, map_idx, met, k], plot_data[type_idx + 1, map_idx, met, k]],
                                    color=colors(map_idx), alpha=0.2, lw=0.5
                                    )

            ax[met].legend(loc='upper right', fontsize=8, frameon=False)
            ax[met].set_ylabel('Identifiability (r)', fontsize=10, fontweight='bold')
            if met < 2:
                ax[met].set_xlim(-1, 11)
                ax[met].set_xticks([0.5, 3.5, 6.5, 9.5], model_label)
            else:
                ax[met].set_xlim(-1, 5)
                ax[met].set_xticks([0.5, 3.5], model_label[:2])
            ax[met].set_title(f'{titles[met]}', fontsize=14, fontweight='bold')
            ax[met].spines['top'].set_visible(False)
            ax[met].spines['right'].set_visible(False)

    for met in range(n_metrics):
        for map_ in range(n_maps):
            a = stat_func.r2z(plot_data[0, map_, met, :], 'pearson')
            b = stat_func.r2z(plot_data[1, map_, met, :], 'pearson')
            results = stats.ttest_rel(a, b, nan_policy='omit')  # best pair vs other pairs
            p_val = results.pvalue
            print(f'Difference: {np.nanmean(plot_data[0, map_, met, :]) - np.nanmean(plot_data[1, map_, met, :]):.4f}')
            print(f"Metric: {titles[met]}, Map: {model_label[map_]}, t-value: {results.statistic:.4f}, p-value: {p_val}")

    plt.suptitle('Identifiability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_pdf(path / f'top_iden_raw_{name}.pdf')
    return plot_data


def plot_corr_within_metric_consistency(data, name, path, split_by):
       # plot corr consistency
    n_maps = len(data) + 1
    n_boots, n_metrics, n_subjs = data[0].get_corr_results('subj', 'inst', 'subj', 'split', split_by=split_by).mat.shape

    plot_data = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            for met_idx in range(n_metrics):
                if map_idx == 0:
                    map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'split', split_by=split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    plot_data[type_idx, map_idx] = map_data
                else:
                    map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'split', split_by=split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    try:
                        plot_data[type_idx, map_idx] = map_data
                    except ValueError:
                        plot_data[type_idx, map_idx, :2] = map_data

    plot_data = -np.diff(plot_data, axis = 0)
    plot_data = np.squeeze(plot_data, axis = 0)

    plt.figure(figsize=(5, 4))

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS

    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson')
    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 4 + map * 0.8
            box = plt.boxplot(plot_data_in_r[map, met, :], positions=[x_pos], widths=0.4, patch_artist=True,
                              showfliers=False,
                        )
            _style_boxplot(box, colors(map))

            for k in range(n_subjs):
                plt.scatter(x_pos - 0.3,
                            plot_data_in_r[map, met, k],
                            color=colors(map), s=5
                            )

    data_min = np.nanmin(plot_data_in_r)
    data_max = np.nanmax(plot_data_in_r)
    data_range = data_max - data_min
    vs_zero_y = data_min - 0.15 * data_range
    max_bracket = data_max

    for met in range(n_metrics):
        sub_data = plot_data[:, met, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):

                if name in ['mnist', 'ecoset10']:
                    t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                    bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                    mean_diff = np.nanmean(sub_data[i]) - np.nanmean(sub_data[j])
                    pooled_std = np.sqrt((np.nanvar(sub_data[i], ddof=1) + np.nanvar(sub_data[j], ddof=1)) / 2)
                    cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
                    print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 8)}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # Check for significance
                    if ((map_labels[i] == 'Human-Human')):
                        if met == 2 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        y_max = max(np.nanmax(plot_data_in_r[:, met, :]), np.nanmax(plot_data_in_r[:, met, :])) + 0.1 * abs(j - i)
                        max_bracket = max(max_bracket, y_max)
                        alpha = 0.5 if p_val < 0.05 else 1
                        _annotate_bracket(met * 4 + i * 0.8, met * 4 + j * 0.8, y_max, _format_pval(p_val),
                                           alpha=alpha, y_text_offset=0.02)
                        plt.legend(loc='upper right', fontsize=8, frameon=False)

                if name in ['mnist', 'ecoset10']:
                    t_stat, p_val = stats.ttest_1samp(sub_data[j], 0, nan_policy='omit')
                    bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                    cohen_d = t_stat / np.sqrt(n_subjs) if n_subjs != 0 else np.nan
                    print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 8)}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # compare to 0 signfiicance
                    if ((map_labels[i] == 'Human-Human')):
                        if met == 2 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        x_pos = (met * 4 + j * 0.8) + 0.15
                        anno, alpha = _stars_for_pval(p_val)
                        plt.annotate(anno, (x_pos, vs_zero_y), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha, fontweight='bold')
                        plt.legend(loc='upper left', fontsize=8, frameon=False)

    plt.xticks([1.2, 5.2, 8.4],
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
            )
    plt.xlim(-1, 10)
    plt.ylim(vs_zero_y - 0.08 * data_range, max_bracket + 0.15 * data_range)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('r(same subject) − r(other subjects)', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_bs_{name}_split_{split_by}.pdf')


def plot_rank_within_metric_consistency(data, name, path, split_by):
    n_maps = len(data) + 1
    n_boots, n_metrics, n_subjs = data[0].get_corr_results('subj', 'inst', 'subj', 'split', split_by).mat.shape

    # plot rank consistency
    plot_data = np.empty(shape=(n_maps, n_boots, n_metrics))
    plot_data.fill(np.nan)
    for map_idx in range(n_maps):
        if map_idx == 0:
            map_data = data[map_idx].get_rank_results('subj', 'subj', 'split', split_by).mat
            plot_data[map_idx] = map_data
        else:
            map_data = data[map_idx-1].get_rank_results('subj', 'inst', 'split', split_by).mat
            try:
                plot_data[map_idx] = map_data
            except ValueError:
                plot_data[map_idx, :, :2] = map_data

    plt.figure(figsize=(5, 4))
    colors = _COLORS

    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 4 + map * 0.8
            if map > 1 and met > 1:
                continue
            box = plt.boxplot(plot_data[map, :, met], positions=[x_pos], widths=0.4, patch_artist=True,
                              showfliers=False,
                        )
            _style_boxplot(box, colors(map))

    data_min = np.nanmin(plot_data)
    data_max = np.nanmax(plot_data)
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    # compute stats using bootstrapping test
    if name == 'mnist':
        for map in range(1, n_maps):
            diff = plot_data[0] - plot_data[map]
            for met in range(n_metrics):
                for_proportion = diff[:, met]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map-1]} - "
                    f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]} - "
                    f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met == 2 and not (map == 1):
                    continue
                y_max = np.nanmax(plot_data[:, :, met]) + 0.08 * data_range * map
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(met * 4, met * 4 + map * 0.8, y_max, _format_pval_simple(p_val),
                                   alpha=alpha, fontsize=9, xytext=(0, 3))

    if name == 'ecoset10':
        for _map in range(1, n_maps):
            for met in range(n_metrics):
                for_proportion = plot_data[_map, :, met]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][_map-1]} - "
                    f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]} - "
                    f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met == 2 and not (_map == 1):
                    continue
                x_pos = (met * 4 + _map * 0.8)
                y_max = np.nanpercentile(plot_data[_map, :, met], 0) - 0.08 * data_range
                min_annot = min(min_annot, y_max)
                anno, alpha = _stars_for_pval(p_val)
                plt.annotate(anno, (x_pos, y_max), ha='center', size=9, alpha=alpha, fontweight='bold')


    plt.xticks([1.2, 5.2, 8.4],
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
            )
    plt.ylim(min_annot - 0.08 * data_range, max_bracket + 0.15 * data_range)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title('Rank consistency', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=8, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'rank_btw_bs_{name}_split_{split_by}.pdf')


def plot_corr_across_metric_consistency(data, name, path, split_by):
    # plot corr consistency
    n_maps = len(data) + 1
    n_boots, n_metrics, n_subjs = data[0].get_corr_results('subj', 'inst', 'subj', 'var', split_by).mat.shape

    plot_data = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'var', split_by).mat
                map_data = stat_func.r2z(map_data, metric='pearson')
                map_data = np.mean(map_data, axis=0)
                plot_data[type_idx, map_idx] = map_data
            else:
                map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'var', split_by).mat
                map_data = stat_func.r2z(map_data, metric='pearson')
                map_data = np.mean(map_data, axis=0)
                if map_idx > 1 and n_metrics > 1:
                    plot_data[type_idx, map_idx, 0] = map_data
                else:
                    plot_data[type_idx, map_idx] = map_data

    plot_data = -np.diff(plot_data, axis = 0)
    plot_data = np.squeeze(plot_data, axis = 0)

    plt.figure(figsize=(4, 4))

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS

    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson')
    for map in range(n_maps):
        for met in range(n_metrics):
            if met < 2:
                x_pos = met * 4 + map * 0.8
            else:
                x_pos = met * 3.2 + map * 0.8

            box = plt.boxplot(plot_data_in_r[map, met, :], positions=[x_pos], widths=0.4, patch_artist=True,
                        showfliers=False,
                        )
            _style_boxplot(box, colors(map))

            for k in range(n_subjs):
                plt.scatter(x_pos - 0.3,
                            plot_data_in_r[map, met, k],
                            color=colors(map), s=5
                            )

    data_min = np.nanmin(plot_data_in_r)
    data_max = np.nanmax(plot_data_in_r)
    data_range = data_max - data_min
    vs_zero_y = data_min - 0.15 * data_range
    max_bracket = data_max

    for met in range(n_metrics):
        sub_data = plot_data[:, met, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):
                if name in ['mnist', 'ecoset10']:
                    t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                    bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                    mean_diff = np.nanmean(sub_data[i]) - np.nanmean(sub_data[j])
                    pooled_std = np.sqrt((np.nanvar(sub_data[i], ddof=1) + np.nanvar(sub_data[j], ddof=1)) / 2)
                    cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
                    print(f"Metric: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 6)}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # Check for significance
                    if ((map_labels[i] == 'Human-Human')):
                        if met != 0 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        y_max = max(np.nanmax(plot_data_in_r[:, met, :]), np.nanmax(plot_data_in_r[:, met, :])) + 0.1 * abs(j - i)
                        max_bracket = max(max_bracket, y_max)
                        if met < 2:
                            plot_x_pos = [met * 4, met * 4 + j * 0.8]
                        else:
                            plot_x_pos = [met * 3.2, met * 3.2 + j * 0.8]

                        alpha = 0.5 if p_val < 0.05 else 1
                        _annotate_bracket(plot_x_pos[0], plot_x_pos[1], y_max, _format_pval(p_val),
                                           alpha=alpha, y_text_offset=0.02)

                if name in ['mnist', 'ecoset10']:
                    t_stat, p_val = stats.ttest_1samp(sub_data[j], 0, nan_policy='omit')
                    bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                    mean_diff = np.nanmean(sub_data[i]) - np.nanmean(sub_data[j])
                    pooled_std = np.sqrt((np.nanvar(sub_data[i], ddof=1) + np.nanvar(sub_data[j], ddof=1)) / 2)
                    # cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
                    cohen_d = t_stat / np.sqrt(n_subjs) if n_subjs != 0 else np.nan
                    print(f"Metric: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 6)}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # compare to 0 signfiicance
                    if ((map_labels[i] == 'Human-Human')):
                        if met != 0 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        if met < 2:
                            x_pos = met * 4 + j * 0.8 + 0.15
                        else:
                            x_pos = met * 3.2 + j * 0.8 + 0.15
                        anno, alpha = _stars_for_pval(p_val)
                        plt.annotate(anno, (x_pos, vs_zero_y), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha, fontweight='bold')

    plt.xticks([1.2, 4.4, 6.8],
                ['Acc-Conf', 'Acc-RT', 'Conf-RT'],
                fontsize=12
            )
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlim(-1, 8.5)
    plt.ylim(vs_zero_y - 0.08 * data_range, max_bracket + 0.15 * data_range)

    plt.xlabel('Pairs of behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel('r(same subject) − r(other subjects)', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=8, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_var_{name}_split_{split_by}.pdf')


def plot_rank_across_metric_consistency(data, name, path, split_by):
    n_maps = len(data) + 1
    n_boots, n_metrics, n_subjs = data[0].get_corr_results('subj', 'inst', 'subj', 'var', split_by).mat.shape

    # plot rank consistency
    plot_data = np.empty(shape=(n_maps, n_boots, n_metrics))
    plot_data.fill(np.nan)
    for map_idx in range(n_maps):
        if map_idx == 0:
            map_data = data[map_idx].get_rank_results('subj', 'subj', 'var', split_by).mat
            plot_data[map_idx] = map_data
        else:
            map_data = data[map_idx-1].get_rank_results('subj', 'inst', 'var', split_by).mat
            if map_idx > 1 and n_metrics > 1:
                plot_data[map_idx, :] = map_data
            else:
                plot_data[map_idx] = map_data

    plt.figure(figsize=(4, 4))
    colors = _COLORS

    for map in range(n_maps):
        for met in range(n_metrics):
            if met != 0 and not (map <= 1):
                continue
            if met < 2:
                x_pos = met * 4 + map * 0.8
            else:
                x_pos = met * 3.2 + map * 0.8

            box = plt.boxplot(plot_data[map, :, met], positions=[x_pos], widths=0.4, patch_artist=True,
                        showfliers=False,
                        )
            _style_boxplot(box, colors(map))


    # only consider the values actually rendered above (met==0 for any map, or
    # met>0 restricted to map<=1) - other entries may hold broadcast junk
    render_mask = np.zeros_like(plot_data, dtype=bool)
    for map in range(n_maps):
        for met in range(n_metrics):
            if met != 0 and not (map <= 1):
                continue
            render_mask[map, :, met] = True
    data_min = np.nanmin(plot_data[render_mask])
    data_max = np.nanmax(plot_data[render_mask])
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    if name == 'mnist':
        # compute stats using bootstrapping test
        for map in range(1, n_maps):
            diff = plot_data[0] - plot_data[map]
            for met in range(n_metrics):
                for_proportion = diff[:, met]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map-1]} - "
                    f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]} - "
                    f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met != 0 and not (map == 1):
                    continue
                if met < 2:
                    plot_x_pos = [met * 4, met * 4 + map * 0.8]
                else:
                    plot_x_pos = [met * 3.2, met * 3.2 + map * 0.8]
                y_max = np.nanmax(plot_data[:, :, met]) + 0.08 * data_range * map
                max_bracket = max(max_bracket, y_max)
                _annotate_bracket(plot_x_pos[0], plot_x_pos[1], y_max, _format_pval_simple(p_val), fontsize=9, xytext=(0, 3))

    if name == 'ecoset10':
        for _map in range(1, n_maps):
            for met in range(n_metrics):
                for_proportion = plot_data[_map, :, met]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][_map-1]} - "
                    f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]} - "
                    f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met != 0 and not (_map == 1):
                    continue
                if met < 2:
                    x_pos = (met * 4 + _map * 0.8)
                else:
                    x_pos = (met * 3.2 + _map * 0.8)
                y_max = min_annot - 0.08 * data_range
                min_annot = min(min_annot, y_max)
                anno, alpha = _stars_for_pval(p_val)
                plt.annotate(anno, (x_pos, y_max), ha='center', size=9, alpha=alpha, fontweight='bold')

    plt.xticks([1.2, 4.4, 6.8],
                ['Acc-Conf', 'Acc-RT', 'Conf-RT'],
                fontsize=12
            )
    plt.xlim(-1, 8.5)
    plt.ylim(min_annot - 0.08 * data_range, max_bracket + 0.15 * data_range)

    plt.xlabel('Pairs of behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title('Rank consistency', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=8, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'rank_btw_var_{name}_split_{split_by}.pdf')


def plot_pca_shuffle_comparison(data, name, path):
    n_maps = len(data)
    n_bs, n_conds, n_met, n_comps = data[0].dims_map.split_half_pca_results['proj_model_var'].shape

    # unpack results
    proj_same_human_var = data[0].dims_map.split_half_pca_results['proj_same_human_var']
    proj_diff_human_var = data[0].dims_map.split_half_pca_results['proj_diff_human_var']
    proj_scrm_human_var = data[0].dims_map.split_half_pca_results['proj_scrm_human_var']
    proj_model_var = np.empty((n_maps, n_conds, n_met, n_comps))
    proj_model_var.fill(np.nan)
    for i in range(n_maps):
        if i == 0:
            proj_model_var[i] = data[i].dims_map.split_half_pca_results['proj_model_var'].mean(axis = 0)
        else:
            proj_model_var[i, :, :2, :] = data[i].dims_map.split_half_pca_results['proj_model_var'].mean(axis = 0)[:, :2, :]

    mean_scrm_var = proj_scrm_human_var.mean(axis=0)
    stack_comparison = np.stack([proj_same_human_var, proj_diff_human_var] + [proj_model_var[i] for i in range(n_maps)], axis=0)
    diff_explained = stack_comparison - mean_scrm_var
    perm_test = np.zeros((stack_comparison.shape[0],
                          n_conds, n_met, n_comps,
                          ))

    for i in range(stack_comparison.shape[0]):
        for cond in range(n_conds):
            for met in range(n_met):
                for comp in range(n_comps):
                    test_val = stack_comparison[i, cond, met, comp]
                    null_dist = proj_scrm_human_var[:, cond, met, comp]
                    perm_test[i, cond, met, comp] = np.sum(test_val <= null_dist) / len(null_dist)


    metric_label = ['Accuracy', 'Confidence', 'RT']
    yticks = ['Same\nhumans', 'Held-out\nhumans', 'RTNet', 'AlexNet', 'ResNet18']
    plt.clf()
    fig, ax = plt.subplots(2, n_met, figsize=(12, 5))
    for i in range(n_conds):
        for j in range(n_met):
            if j == 2:
                ax[i,j].imshow(
                    diff_explained[:3,i,j],
                    aspect='auto',
                    cmap='copper_r',
                )
            else:
                ax[i,j].imshow(
                    diff_explained[:,i,j],
                    aspect='auto',
                    cmap='copper_r',
                )
    
            if j == 2:
                ax[i,j].set_yticks([0,1,2], yticks[:3], fontsize=10)
                ax[i,j].set_xticks(np.arange(10), np.arange(10) + 1, fontsize=8)
                ax[i,j].set_xlabel('PCA components', fontsize=10)
                ax[i,j].set_title(f'{metric_label[j]}', fontsize=12, fontweight='bold')
            else:
                ax[i,j].set_yticks(np.arange(len(yticks)), yticks, fontsize=10)
                ax[i,j].set_xticks(np.arange(10), np.arange(10) + 1, fontsize=8)
                ax[i,j].set_xlabel('PCA components', fontsize=10)
                ax[i,j].set_title(f'{metric_label[j]}', fontsize=12, fontweight='bold')
            
            if np.any(perm_test[:, i, j] < 0.05):
                for map_ in range(diff_explained.shape[0]):
                    for comp in range(diff_explained.shape[3]):
                        color = 'white' if diff_explained[map_, i, j, comp] > 0.05 else 'black'
                        anno_text = "***" if perm_test[map_, i, j, comp] < 0.001 else "**" if perm_test[map_, i, j, comp] < 0.01 else "*" if perm_test[map_, i, j, comp] < 0.05 else ""
                        ax[i,j].annotate(anno_text, xy=(comp, map_), color=color, fontsize=10, ha='center', va='center')

    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.9])  # [left, bottom, width, height]
    cbar = fig.colorbar(ax[0, 0].images[0], cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout(rect = [0, 0, 0.92, 1])
    plt.savefig(path / f'pca_shuffle_comparison_{name}.png', dpi=384, transparent=True)
    plt.close()


def plot_pca_cumulative_evidence_lineplot(data, name, path):
    n_maps = len(data)
    n_bs, n_conds, n_met, n_comps = data[0].dims_map.split_half_pca_results['proj_model_var'].shape

    # unpack results
    proj_same_human_var = data[0].dims_map.split_half_pca_results['proj_same_human_var']
    proj_diff_human_var = data[0].dims_map.split_half_pca_results['proj_diff_human_var']
    proj_scrm_human_var = data[0].dims_map.split_half_pca_results['proj_scrm_human_var'].mean(axis = 0)
    proj_model_var = np.empty((n_maps, n_conds, n_met, n_comps))
    proj_model_var.fill(np.nan)
    for i in range(n_maps):
        if i == 0:
            proj_model_var[i] = data[i].dims_map.split_half_pca_results['proj_model_var'].mean(axis = 0)
        else:
            proj_model_var[i, :, :2, :] = data[i].dims_map.split_half_pca_results['proj_model_var'].mean(axis = 0)[:, :2, :]
    plot_data = np.stack([proj_diff_human_var, proj_scrm_human_var] + [proj_model_var[i] for i in range(n_maps)], axis=0)

    plt.clf()
    figure, ax = plt.subplots(2, n_met, figsize=(7, 3.5))
    # figure, ax = plt.subplots(2, n_met, figsize=(7, 5))
    colors = plt.cm.get_cmap('Set1', 8)
    colors = [colors(0), 'grey', colors(1), colors(2), colors(3)]
    zorder = np.arange(len(colors))[::-1]
    x_axis = np.arange(0, 11)
    met_labels = ['Accuracy', 'Confidence', 'RT']
    model_labels = ['Held-out Human', 'Scrambled Human', 'RTNet', 'AlexNet', 'ResNet18']
    conds_labels = ['Accuracy Focus', 'Speed Focus']

    for map_idx in range(plot_data.shape[0]):
        for conds_idx in range(n_conds):
            for metric in range(n_met):
                print(f"{model_labels[map_idx]}, {conds_labels[conds_idx]}, {met_labels[metric]}:")
                print(np.cumsum(plot_data[map_idx, conds_idx, metric])[-1]*100)
                print()
                ax[conds_idx, metric].plot(
                    range(1, n_comps + 1),
                    np.cumsum(plot_data[map_idx, conds_idx, metric]),
                    color=colors[map_idx],
                    linestyle='solid',
                    lw=2.5, zorder=zorder[map_idx], alpha=0.75
                )
                ax[conds_idx, metric].set_xlim(0.1, 11)
                # ax[conds_idx, metric].set_ylim(-0.05, 0.65)
                ax[conds_idx, metric].set_xticks(x_axis[1:], x_axis[1:], fontsize=6)
                ax[conds_idx, metric].tick_params(axis='y', labelsize=6)
                ax[conds_idx, metric].set_xlabel('PCA components', fontsize=6, fontweight='bold')
                ax[conds_idx, metric].set_ylabel('Cumulative Explained Variance', fontsize=6, fontweight='bold')
                ax[conds_idx, metric].set_title(f'{met_labels[metric]}', fontsize=8, fontweight='bold')
                ax[conds_idx, metric].spines['top'].set_visible(False)
                ax[conds_idx, metric].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(path / f'pca_cumulative_explained_variance_{name}.png', dpi=384, transparent=True)
    plt.close()


def plot_pca_total_variance_comparison_split_graph(data, name, path):
    n_maps = len(data)
    n_bs, n_cond, n_met, n_comps = data[0].dims_map.split_half_pca_results['proj_model_var'].shape

    proj_human_var = data[0].dims_map.split_half_pca_results['proj_diff_human_var'].sum(axis = 2)
    scrm_human_var = data[0].dims_map.split_half_pca_results['proj_scrm_human_var'].sum(axis = 3)
    proj_model_var = np.empty((n_maps, n_bs, n_cond, n_met))
    proj_model_var.fill(np.nan)
    for i in range(n_maps):
        if i == 0:
            proj_model_var[i] = data[i].dims_map.split_half_pca_results['proj_model_var'].sum(axis=3)
        else:
            proj_model_var[i, :, :, :2] = data[i].dims_map.split_half_pca_results['proj_model_var'].sum(axis=3)

    conds_labels = ['accuracy_focus', 'speed_focus']
    for cond in range(n_cond):
        for met in range(n_met):
            plt.clf()
            plt.figure(figsize=(3, 2))
            colors = plt.cm.get_cmap('Set1', 8)
            model_label = ['Held-out human', 'RTNet', 'AlexNet', 'ResNet18', 'Shuffled human']
            metric_label = ['Accuracy', 'Confidence', 'RT']

            for map_ in range(n_maps + 2):
                x_pos = map_ * 0.8
                met_human_arr = proj_human_var[cond, met]
                scrm_human_arr = scrm_human_var[:, cond, met]
                if map_ == 0:
                    plt.bar(x_pos, met_human_arr, color=colors(map_), 
                            label=model_label[map_] if met == 0 else None, alpha=0.75)
                elif map_ == n_maps + 1:
                    if met == 2:
                        x_pos -= 1.6
                    plt.bar(x_pos, np.nanmean(scrm_human_arr), 
                            yerr=np.nanstd(scrm_human_arr),
                            color='grey',
                            label=model_label[map_] if met == 0 else None, alpha=0.75)
                else:
                    met_model_arr = proj_model_var[map_-1, :, cond, met] if map_ > 0 else None
                    plt.bar(x_pos, np.nanmean(met_model_arr), 
                            yerr=np.nanstd(met_model_arr),
                            color=colors(map_), 
                            label=model_label[map_] if met == 0 else None, 
                            alpha=0.5)

                    # annotation
                    diff = met_model_arr - met_human_arr
                    p_val = 2 * min(
                        len(diff[diff >= 0]) / len(diff),
                        len(diff[diff <= 0]) / len(diff)
                    )
                    ci_lower = np.percentile(diff, 2.5)
                    ci_upper = np.percentile(diff, 97.5)
                    x_mid = (map_ * 0.8) / 2  # Midpoint between bars
                    y_max = np.nanmax(np.nanmean(proj_model_var[:, :, cond, met], axis = 1)) + 0.02 * map_ + 0.02

                    alpha = 1
                    if p_val < 0.001:
                        anno = f'$p$ < 0.001'
                    else:
                        anno = f'$p$ = {p_val:.3f}'
                    if map_ > 1 and met > 1:
                        continue
                    plt.plot([0, map_ * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                    plt.annotate(anno, (x_mid, y_max+0.002), textcoords="offset points", 
                                 xytext=(0, 1), ha='center', size=8, alpha=alpha)

                    print(f"Condition: {conds_labels[cond]}, Map: {model_label[map_]}, Metric: {metric_label[met]}, p-value: {_format_pval_print(p_val, 3)}")
                    print(f"CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                    print(f"Human: {met_human_arr}")
                    print(f"Mean: {np.nanmean(met_model_arr):.5f}, Std: {np.nanstd(met_model_arr):.5f}")
            
            plt.xticks([], [])
            # plt.ylim(0, 0.25)
            plt.ylabel('Total explained variance', fontsize=10, fontweight='bold')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(path / f'dims_split_half_pca_{conds_labels[cond]}_{metric_label[met]}_{name}.png', dpi=384, transparent=True)
            plt.close()


def plot_pca_total_variance_comparison(data, name, path):
    n_maps = len(data)
    n_bs, n_cond, n_met, n_comps = data[0].dims_map.split_half_pca_results['proj_model_var'].shape

    proj_human_var = data[0].dims_map.split_half_pca_results['proj_diff_human_var'].sum(axis = 2)
    scrm_human_var = data[0].dims_map.split_half_pca_results['proj_scrm_human_var'].sum(axis = 3)
    proj_model_var = np.empty((n_maps, n_bs, n_cond, n_met))
    proj_model_var.fill(np.nan)
    for i in range(n_maps):
        if i == 0:
            proj_model_var[i] = data[i].dims_map.split_half_pca_results['proj_model_var'].sum(axis=3)
        else:
            proj_model_var[i, :, :, :2] = data[i].dims_map.split_half_pca_results['proj_model_var'].sum(axis=3)

    conds_labels = ['accuracy_focus', 'speed_focus']
    for cond in range(n_cond):
        plt.clf()
        plt.figure(figsize=(6, 3))
        colors = plt.cm.get_cmap('Set1', 8)
        model_label = ['Held-out human', 'RTNet', 'AlexNet', 'ResNet18', 'Shuffled human']

        if name == 'mnist' or name == 'ecoset10':
            plt.xticks([1.6, 6.6, 10.8], 
                        ['Accuracy', 'Confidence', 'RT'],
                        fontsize=10
                        )

        for map_ in range(n_maps + 2):
            for met in range(n_met):
                x_pos = met * 5 + map_ * 0.8
                met_human_arr = proj_human_var[cond, met]
                scrm_human_arr = scrm_human_var[:, cond, met]
                if map_ == 0:
                    plt.bar(x_pos, met_human_arr, color=colors(map_), 
                            label=model_label[map_] if met == 0 else None, alpha=0.75)
                elif map_ == n_maps + 1:
                    if met == 2:
                        x_pos -= 1.6
                    plt.bar(x_pos, np.nanmean(scrm_human_arr), 
                            yerr=np.nanstd(scrm_human_arr),
                            color='grey',
                            label=model_label[map_] if met == 0 else None, alpha=0.75)
                else:
                    met_model_arr = proj_model_var[map_-1, :, cond, met] if map_ > 0 else None
                    plt.bar(x_pos, np.nanmean(met_model_arr), 
                            yerr=np.nanstd(met_model_arr),
                            color=colors(map_), 
                            label=model_label[map_] if met == 0 else None, 
                            alpha=0.5)

                    # annotation
                    diff = met_model_arr - met_human_arr
                    p_val = 2 * min(
                        len(diff[diff >= 0]) / len(diff),
                        len(diff[diff <= 0]) / len(diff)
                    )
                    ci_lower = np.percentile(diff, 2.5)
                    ci_upper = np.percentile(diff, 97.5)
                    x_mid = (met * 5 + met * 5 + map_ * 0.8) / 2  # Midpoint between bars
                    y_max = np.nanmax(np.nanmean(proj_model_var[:, :, cond, met], axis = 1)) + 0.02 * map_ + 0.02

                    alpha = 1
                    if p_val < 0.001:
                        anno = f'$p$ < 0.001'
                    else:
                        anno = f'$p$ = {p_val:.3f}'
                    if map_ > 1 and met > 1:
                        continue
                    if p_val > 0.05:
                        alpha=0.5
                        plt.plot([met * 5, met * 5 + map_ * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                        plt.annotate(anno, (x_mid, y_max+0.002), textcoords="offset points", 
                                    xytext=(0, 1), ha='center', size=8, alpha=alpha)

                    print("=======================================")
                    print("Annotation for model vs. Held-out human")
                    print("=======================================")
                    print(f"Map: {map_}, Metric: {met}, p-value: {_format_pval_print(p_val, 3)}")
                    print(f"CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                    print(f"Human: {met_human_arr}")
                    print(f"Mean: {np.nanmean(met_model_arr):.5f}, Std: {np.nanstd(met_model_arr):.5f}")

                    # annotate for shuffled
                    shuffle_x_pos = 4 if met < 2 else 2
                    diff = (met_model_arr[:, None] - scrm_human_arr[None, :]).ravel()
                    p_val = 2 * min(
                        len(diff[diff >= 0]) / len(diff),
                        len(diff[diff <= 0]) / len(diff)
                    )
                    ci_lower = np.percentile(diff, 2.5)
                    ci_upper = np.percentile(diff, 97.5)
                    x_mid = (met * 5 + shuffle_x_pos * 0.8 + met * 5 + map_ * 0.8) / 2  # Midpoint between bars
                    y_max = np.nanmax(np.nanmean(proj_model_var[:, :, cond, met], axis = 1)) + 0.02 * (4-map_)

                    alpha = 1
                    if p_val < 1e-6:
                        anno = r'$p < 10^{-6}$'
                    elif p_val < 0.001:
                        power = int(np.floor(np.log10(p_val)))
                        coefficient = p_val / (10 ** power)
                        anno = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
                    else:
                        anno = r"$p = {:.3f}$".format(p_val)
                    if map_ > 1 and met > 1:
                        continue
                    plt.plot([met * 5 + shuffle_x_pos * 0.8, met * 5 + map_ * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                    plt.annotate(anno, (x_mid, y_max+0.002), textcoords="offset points", 
                                 xytext=(0, 1), ha='center', size=8, alpha=alpha)

                    print("=======================================")
                    print("Annotation for model vs. shuffled human")
                    print("=======================================")
                    print(f"Map: {map_}, Metric: {met}, p-value: {_format_pval_print(p_val, 6)}")
                    print(f"CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                    print(f"Human: {np.nanmean(scrm_human_arr):.5f}, Std: {np.nanstd(scrm_human_arr):.5f}")
                    print(f"Mean: {np.nanmean(met_model_arr):.5f}, Std: {np.nanstd(met_model_arr):.5f}")


        plt.ylim(0, 0.2)
        plt.xlabel('Behavioral metrics', fontsize=10, fontweight='bold')
        plt.ylabel('Total explained variance', fontsize=10)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.legend(loc='upper left', fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(path / f'dims_split_half_pca_{conds_labels[cond]}_{name}.png', dpi=384, transparent=True)
        plt.close()


def plot_within_metric_prediction_raw(data, name, path):
    n_maps = len(data) + 1
    n_boots, n_metrics, n_subjs = data[0].get_pred_results('subj', 'avg', True).mat.shape
    methods = ['avg', 'corr']
    plot_data = np.empty((len(methods), n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)

    for i, method in enumerate(methods):
        for map_idx in range(n_maps):
            if map_idx == 0:
                result = data[map_idx].get_pred_results('subj', method, True).mat
            else:
                result = data[map_idx-1].get_pred_results('inst', method, True).mat
            result = stat_func.r2z(result, metric='pearson')
            result = np.nanmean(result, axis = 0)
            try:
                plot_data[i, map_idx] = result
            except ValueError:
                plot_data[i, map_idx, :2] = result
    plot_data = stat_func.z2r(plot_data, metric='pearson')

    fig, ax = plt.subplots(1, n_metrics, figsize=(10.5, 3.5))
    colors = _COLORS
    model_label = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = ['Unweighted', 'Weighted']
    titles = ['Accuracy', 'Confidence', 'RT']

    for map_idx in range(n_maps):
        for met in range(n_metrics):
            for meth in range(len(methods)):
                x_pos = meth + map_idx * 3
                if not np.isnan(plot_data[meth, map_idx, met, :]).all():
                    ax[met].scatter(x_pos,
                        np.mean(plot_data[meth, map_idx, met, :]),
                        color=colors(map_idx), label=map_labels[meth] if map_idx == 0 else None,
                        alpha = 1, marker='s' if meth == 0 else 'D', s=50
                        )
                for k in range(n_subjs):
                    ax[met].scatter(x_pos,
                                plot_data[meth, map_idx, met, k],
                                color=colors(map_idx), s=1, alpha=0.5
                                )
                    if meth == 0:
                        ax[met].plot([x_pos, x_pos + 1],
                                    [plot_data[meth, map_idx, met, k], plot_data[meth + 1, map_idx, met, k]],
                                    color=colors(map_idx), alpha=0.2, lw=0.5
                                    )

            ax[met].legend(loc='upper right', fontsize=8, frameon=False)
            ax[met].set_ylabel('Predictive accuracy (ρ)', fontsize=10, fontweight='bold')
            if met < 2:
                ax[met].set_xlim(-1, 11)
                ax[met].set_xticks([0.5, 3.5, 6.5, 9.5], model_label)
            else:
                ax[met].set_xlim(-1, 5)
                ax[met].set_xticks([0.5, 3.5], model_label[:2])
            ax[met].set_title(f'{titles[met]}', fontsize=14, fontweight='bold')
            ax[met].spines['top'].set_visible(False)
            ax[met].spines['right'].set_visible(False)

    for met in range(n_metrics):
        for map_ in range(n_maps):
            a = stat_func.r2z(plot_data[0, map_, met, :], 'pearson') 
            b = stat_func.r2z(plot_data[1, map_, met, :], 'pearson')
            results = stats.ttest_rel(a, b)  # average vs corr weight
            p_val = results.pvalue
            print(f'Difference: {np.mean(plot_data[0, map_, met, :]) - np.mean(plot_data[1, map_, met, :]):.4f}')
            print(f"Metric: {titles[met]}, Map: {model_label[map_]}, Method: {methods[1]}, t-value: {results.statistic:.4f}, p-value: {p_val}")

    plt.suptitle('Within-metric prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_pdf(path / f'pred_wn_var_{name}.pdf')
    return plot_data


def plot_within_metric_prediction_diff(data, name, path):
    plot_data = np.diff(data, axis = 0).squeeze()
    n_maps, n_metrics, n_subjs = plot_data.shape

    if name == 'mnist':
        plt.figure(figsize=(6, 3))
        colors = _COLORS
        titles = ['Accuracy', 'Confidence', 'RT']
        model_label = ['Subject', 'RTNet', 'AlexNet', 'ResNet18']

        for map_ in range(n_maps):
            for met in range(n_metrics):
                x_pos = met * 5 + map_ * 0.8
                box = plt.boxplot(plot_data[map_, met, :], positions=[x_pos], widths=0.6, patch_artist=True,
                            showfliers=False,
                            )
                _style_boxplot(box, colors(map_))

        for met in range(n_metrics):
            sub_data = plot_data[:, met, :]
            for i, j in combinations(range(n_maps), 2):
                if i != 0:
                    continue
                if met == 2 and j > 1:
                    continue

                data_i = stat_func.r2z(sub_data[i], 'pearson')
                data_j = stat_func.r2z(sub_data[j], 'pearson')
                t_stat, p_val = stats.ttest_rel(data_i, data_j)  # average vs corr weight
                print(f"Metric: {titles[met]}, Map1: {model_label[i]}, Map2: {model_label[j]}, t-value: {t_stat:.4f}, p-value: {p_val}")

                alpha = 1 if p_val < 0.05 else 0.5
                y_max = np.nanpercentile(plot_data[:, met, :], 97.5) + 0.015 * abs(j-i)
                _annotate_bracket(met * 5 + i * 0.8, met * 5 + j * 0.8, y_max, _format_pval(p_val),
                                   alpha=alpha, fontsize=8, xytext=(0, 2.5))

        plt.xticks([1.2, 6.2, 10.4],
                        ['Accuracy', 'Confidence', 'RT'],
                        fontsize=12
                        )
        plt.xlabel('Behavioral metrics', fontsize=12, fontweight='bold')
        plt.ylabel('r(weighted) − r(unweighted)', fontsize=12)
        plt.axhline(0, color='black', lw=1, ls='dotted', alpha=0.8)
        plt.xlim(-1, 15.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()
        _save_pdf(path / f'pred_wn_var_diff_{name}.pdf')

    else:
        fig, ax = plt.subplots(1, n_metrics, figsize=(7, 3))
        colors = _COLORS
        titles = ['Accuracy', 'Confidence', 'RT']
        model_label = ['Subject', 'RTNet', 'AlexNet', 'ResNet18']

        for map_ in range(n_maps):
            for met in range(n_metrics):
                x_pos = map_ * 0.8
                box = ax[met].boxplot(plot_data[map_, met, :], positions=[x_pos], widths=0.6, patch_artist=True,
                            showfliers=False,
                            )
                _style_boxplot(box, colors(map_))

        for met in range(n_metrics):
            sub_data = plot_data[:, met, :]
            for i, j in combinations(range(n_maps), 2):
                if i != 0:
                    continue
                if met == 2 and j > 1:
                    continue

                data_i = stat_func.r2z(sub_data[i], 'pearson')
                data_j = stat_func.r2z(sub_data[j], 'pearson')
                t_stat, p_val = stats.ttest_rel(data_i, data_j)  # average vs corr weight
                print(f"Metric: {titles[met]}, Map1: {model_label[i]}, Map2: {model_label[j]}, t-value: {t_stat:.4f}, p-value: {p_val}")

                alpha = 1 if p_val < 0.05 else 0.5
                y_max = np.nanpercentile(plot_data[:, met, :], 95) + 0.0005 * abs(j-i)
                _annotate_bracket(i * 0.8, j * 0.8, y_max, _format_pval(p_val), ax=ax[met],
                                   alpha=alpha, fontsize=8, xytext=(0, 2.5))
                ax[met].set_title(f'{titles[met]}', fontsize=10)
                ax[met].spines['top'].set_visible(False)
                ax[met].spines['right'].set_visible(False)
                ax[met].axhline(0, color='black', lw=1, ls='dotted', alpha=0.8)
                ax[met].tick_params(axis='y', labelsize=6)
                ax[met].set_ylabel('r(weighted) − r(unweighted)', fontsize=12) if met == 0 else None
                if met < 2:
                    ax[met].set_xticks([0, 0.8, 1.6, 2.4], model_label, fontsize=6)
                    ax[met].set_xlim(-0.8, 3.2)
                else:
                    ax[met].set_xticks([0, 0.8], model_label[:2], fontsize=6)
                    ax[met].set_xlim(-0.8, 3.2)

        plt.tight_layout()
        _save_pdf(path / f'pred_wn_var_diff_{name}.pdf')


def plot_within_subject_consistency_in_human(expt, path):
    from util import dataset
    if expt == 'mnist':
        data = dataset.get_human_on_mnist('repeat')
        image_index = 'mnist_index'
    elif expt == 'ecoset10':
        data = dataset.get_human_on_ecoset10('repeat')
        image_index = 'image_index'

    results = []
    for subj, grp in data.groupby('subj'):
        r0 = grp[grp['reps'] == 0][[image_index, 'resp', 'acc', 'rt', 'conf']]
        r1 = grp[grp['reps'] == 1][[image_index, 'resp', 'acc', 'rt', 'conf']]
        merged = pd.merge(r0, r1, on=image_index, suffixes=('_0', '_1'))
        row = {'subject': subj}
        for var in ['resp', 'acc', 'rt', 'conf']:
            row[f'r_{var}'] = merged[f'{var}_0'].corr(merged[f'{var}_1'])
        results.append(row)

    out = pd.DataFrame(results)
    metrics = ['acc', 'conf', 'rt', 'resp']
    metric_labels = ['Accuracy', 'Confidence', 'RT', 'Response']
    dists = [out[f'r_{m}'].dropna().values for m in metrics]
    colors = _COLORS

    for met_label, vals in zip(metric_labels, dists):
        print(f'{expt} within-subject consistency ({met_label}): mean = {np.mean(vals):.3f}, sem = {sem(vals):.3f}')

    plt.figure(figsize=(3, 3.5))

    for met_idx, vals in enumerate(dists):
        x_pos = met_idx
        box = plt.boxplot(vals, positions=[x_pos * 0.8], widths=0.4, patch_artist=True, showfliers=False)
        _style_boxplot(box, colors(0))

        for subj_val in vals:
            plt.scatter(x_pos * 0.8 - 0.3, subj_val, color=colors(0), alpha=0.75, s=10)

    plt.xticks([i * 0.8 for i in range(len(metrics))], metric_labels, fontsize=8.5)
    plt.xlabel('Behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel(r'r', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(f'{expt.upper()}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    _save_pdf(path / f'human_within_subject_consistency_{expt}.pdf')


def plot_across_metric_correlation_in_human(expt, path):
    from util import dataset
    if expt == 'mnist':
        data = dataset.get_human_on_mnist()
        image_index = 'mnist_index'
    elif expt == 'ecoset10':
        data = dataset.get_human_on_ecoset10()
        image_index = 'image_index'

    avg = data.groupby(['subj', image_index])[['acc', 'rt', 'conf']].mean().reset_index()

    # For each subject, correlate averaged metrics image-by-image
    results = []
    for subj, grp in avg.groupby('subj'):
        row = {'subj': subj}
        row['r_acc_conf'] = grp['acc'].corr(grp['conf'])
        row['r_acc_rt']   = -grp['acc'].corr(grp['rt'])
        row['r_conf_rt']  = -grp['conf'].corr(grp['rt'])
        results.append(row)

    out = pd.DataFrame(results)

    metrics = ['r_acc_conf', 'r_acc_rt', 'r_conf_rt']
    metric_labels = ['Acc-Conf', 'Acc-RT', 'Conf-RT']
    dists = [out[m].dropna().values for m in metrics]
    colors = _COLORS

    for met_label, vals in zip(metric_labels, dists):
        print(f'{expt} across-metric correlation ({met_label}): mean = {np.mean(vals):.3f}, sem = {sem(vals):.3f}')

    plt.figure(figsize=(3, 3.5))

    for met_idx, vals in enumerate(dists):
        box = plt.boxplot(vals, positions=[met_idx * 0.8], widths=0.4, patch_artist=True, showfliers=False)
        _style_boxplot(box, colors(0))

        for subj_val in vals:
            plt.scatter(met_idx * 0.8 - 0.3, subj_val, color=colors(0), alpha=0.75, s=10)

    plt.xticks([i * 0.8 for i in range(len(metrics))], metric_labels, fontsize=10)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1)
    plt.xlabel('Pairs of behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel(r'r', fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(f'{expt.upper()}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    _save_pdf(path / f'human_across_metric_correlation_{expt}.pdf')


def _style_control_boxplot(box, color):
    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_linewidth(0)
    for whisker in box['whiskers']:
        whisker.set_color(color)
        whisker.set_linewidth(2.5)
        whisker.set_alpha(1)
    for cap in box['caps']:
        cap.set_color(color)
        cap.set_linewidth(2.5)
        cap.set_alpha(1)
    for median in box['medians']:
        median.set_color(color)
        median.set_linewidth(2.5)
        median.set_alpha(1)


def plot_corr_within_metric_consistency_accuracy_control(all_maps, sds, path, split_by):
    """ Accuracy-control counterpart of plot_corr_within_metric_consistency (MNIST only).

    all_maps: list indexed by sd position, each entry a [rtnet, alexnet, resnet18]
    list of IndiMap objects for that sd. One figure is produced per behavioral
    metric, with sd on the x-axis and the four Human-X comparisons grouped
    (and colored) within each sd position.
    """
    n_sds = len(all_maps)
    n_maps = len(all_maps[0]) + 1
    n_boots, n_metrics, n_subjs = all_maps[0][0].get_corr_results('subj', 'inst', 'subj', 'split', split_by=split_by).mat.shape

    plot_data = np.empty(shape=(2, n_sds, n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for sd_idx, data in enumerate(all_maps):
            for map_idx in range(n_maps):
                if map_idx == 0:
                    map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'split', split_by=split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    plot_data[type_idx, sd_idx, map_idx] = map_data
                else:
                    map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'split', split_by=split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    try:
                        plot_data[type_idx, sd_idx, map_idx] = map_data
                    except ValueError:
                        plot_data[type_idx, sd_idx, map_idx, :2] = map_data

    plot_data = -np.diff(plot_data, axis=0)
    plot_data = np.squeeze(plot_data, axis=0)  # (n_sds, n_maps, n_metrics, n_subjs)
    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson')

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS
    sd_labels = [str(sd) if sd != 0 else '0' for sd in sds]
    met_titles = ['Accuracy', 'Confidence', 'Reaction time']
    met_fnames = ['acc', 'conf', 'rt']

    for met_idx in range(n_metrics):
        plt.figure(figsize=(7, 4))

        for sd_idx in range(n_sds):
            for map_idx in range(n_maps):
                vals = plot_data_in_r[sd_idx, map_idx, met_idx, :]
                if np.all(np.isnan(vals)):
                    continue
                x_pos = sd_idx * 4 + map_idx * 0.8
                box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
                _style_boxplot(box, colors(map_idx))
                for k in range(n_subjs):
                    plt.scatter(x_pos - 0.3, vals[k], color=colors(map_idx), s=5)

        for sd_idx in range(n_sds):
            sub_data = plot_data[sd_idx, :, met_idx, :]
            for map_idx in range(1, n_maps):
                if np.all(np.isnan(sub_data[map_idx])):
                    continue
                t_stat, p_val = stats.ttest_ind(sub_data[0], sub_data[map_idx], equal_var=False, nan_policy='omit')
                bayes10, bayes01 = _bayes_factors(sub_data[0], sub_data[map_idx], paired=False)
                print(f"Metric: {met_titles[met_idx]}, SD: {sd_labels[sd_idx]}, "
                      f"Comparison: {map_labels[0]} vs {map_labels[map_idx]} - "
                      f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 6)}, BF10: {bayes10:.4f}, BF01: {bayes01:.4f}")

                y_max = np.nanmax(plot_data_in_r[sd_idx, :, met_idx, :]) + 0.08 * map_idx
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(sd_idx * 4, sd_idx * 4 + map_idx * 0.8, y_max, _format_pval(p_val),
                                   alpha=alpha, fontsize=6)

                t_stat0, p_val0 = stats.ttest_1samp(sub_data[map_idx], 0, nan_policy='omit')
                star_x = sd_idx * 4 + map_idx * 0.8
                star_y = np.nanmin(plot_data_in_r[sd_idx, :, met_idx, :]) - 0.05
                star, star_alpha = _stars_for_pval(p_val0)
                plt.annotate(star, (star_x, star_y), ha='center', size=8, fontweight='bold', alpha=star_alpha)

        plt.xticks([sd_idx * 4 + 1.2 for sd_idx in range(n_sds)], sd_labels, fontsize=12)
        plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
        plt.xlabel('Accuracy shift (SD)', fontsize=14, fontweight='bold')
        plt.ylabel('r(same subject) − r(other subjects)', fontsize=12, fontweight='bold')
        plt.title(f'Correlation consistency - {met_titles[met_idx]}', fontsize=15, fontweight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        _save_pdf(path / f'corr_btw_bs_accuracy_control_{met_fnames[met_idx]}_split_{split_by}.pdf')


def plot_rank_within_metric_consistency_accuracy_control(all_maps, sds, path):
    """ Accuracy-control counterpart of plot_rank_within_metric_consistency (MNIST only). """
    n_sds = len(all_maps)
    n_maps = len(all_maps[0]) + 1
    n_boots, n_metrics, n_subjs = all_maps[0][0].get_corr_results('subj', 'inst', 'subj', 'split').mat.shape

    plot_data = np.empty(shape=(n_sds, n_maps, n_boots, n_metrics))
    plot_data.fill(np.nan)
    for sd_idx, data in enumerate(all_maps):
        for map_idx in range(n_maps):
            if map_idx == 0:
                map_data = data[map_idx].get_rank_results('subj', 'subj', 'split').mat
                plot_data[sd_idx, map_idx] = map_data
            else:
                map_data = data[map_idx-1].get_rank_results('subj', 'inst', 'split').mat
                try:
                    plot_data[sd_idx, map_idx] = map_data
                except ValueError:
                    plot_data[sd_idx, map_idx, :, :2] = map_data

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS
    sd_labels = [str(sd) if sd != 0 else '0' for sd in sds]
    met_titles = ['Accuracy', 'Confidence', 'Reaction time']
    met_fnames = ['acc', 'conf', 'rt']

    for met_idx in range(n_metrics):
        plt.figure(figsize=(7, 4))

        for sd_idx in range(n_sds):
            for map_idx in range(n_maps):
                vals = plot_data[sd_idx, map_idx, :, met_idx]
                if np.all(np.isnan(vals)):
                    continue
                x_pos = sd_idx * 4 + map_idx * 0.8
                box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
                _style_boxplot(box, colors(map_idx))

        for sd_idx in range(n_sds):
            for map_idx in range(1, n_maps):
                if np.all(np.isnan(plot_data[sd_idx, map_idx, :, met_idx])):
                    continue
                diff = plot_data[sd_idx, 0, :, met_idx] - plot_data[sd_idx, map_idx, :, met_idx]
                p_val = 2 * min(np.mean(diff >= 0), np.mean(diff < 0))
                ci_lower = np.percentile(diff, 2.5)
                ci_upper = np.percentile(diff, 97.5)
                print(f"Metric: {met_titles[met_idx]}, SD: {sd_labels[sd_idx]}, "
                      f"Comparison: {map_labels[0]} vs {map_labels[map_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

                y_max = np.nanmax(plot_data[sd_idx, :, :, met_idx]) + 15 * map_idx
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(sd_idx * 4, sd_idx * 4 + map_idx * 0.8, y_max, _format_pval_simple(p_val),
                                   alpha=alpha, fontsize=6)

        plt.xticks([sd_idx * 4 + 1.2 for sd_idx in range(n_sds)], sd_labels, fontsize=12)
        plt.xlabel('Accuracy shift (SD)', fontsize=14, fontweight='bold')
        plt.ylabel('Rank consistency metric', fontsize=12)
        plt.title(f'Rank consistency - {met_titles[met_idx]}', fontsize=15, fontweight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        _save_pdf(path / f'rank_btw_bs_accuracy_control_{met_fnames[met_idx]}.pdf')


def plot_corr_across_metric_consistency_accuracy_control(all_maps, sds, path, split_by):
    """ Accuracy-control counterpart of plot_corr_across_metric_consistency (MNIST only).

    One figure is produced per metric-pair (Acc-Conf, Acc-RT, RT-Conf). AlexNet
    and ResNet18 lack RT, so their boxes/stats are only ever populated for the
    Acc-Conf pair (index 0) - handled the same way as the original function.
    """
    n_sds = len(all_maps)
    n_maps = len(all_maps[0]) + 1
    n_boots, n_pairs, n_subjs = all_maps[0][0].get_corr_results('subj', 'inst', 'subj', 'var', split_by).mat.shape

    plot_data = np.empty(shape=(2, n_sds, n_maps, n_pairs, n_subjs))
    plot_data.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for sd_idx, data in enumerate(all_maps):
            for map_idx in range(n_maps):
                if map_idx == 0:
                    map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'var', split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.mean(map_data, axis=0)
                    plot_data[type_idx, sd_idx, map_idx] = map_data
                else:
                    map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'var', split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.mean(map_data, axis=0)
                    if map_idx > 1 and n_pairs > 1:
                        plot_data[type_idx, sd_idx, map_idx, 0] = map_data
                    else:
                        plot_data[type_idx, sd_idx, map_idx] = map_data

    plot_data = -np.diff(plot_data, axis=0)
    plot_data = np.squeeze(plot_data, axis=0)  # (n_sds, n_maps, n_pairs, n_subjs)
    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson')

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS
    sd_labels = [str(sd) if sd != 0 else '0' for sd in sds]
    pair_titles = ['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence']
    pair_fnames = ['acc_conf', 'acc_rt', 'rt_conf']

    for pair_idx in range(n_pairs):
        plt.figure(figsize=(7, 4))

        for sd_idx in range(n_sds):
            for map_idx in range(n_maps):
                vals = plot_data_in_r[sd_idx, map_idx, pair_idx, :]
                if np.all(np.isnan(vals)):
                    continue
                x_pos = sd_idx * 4 + map_idx * 0.8
                box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
                _style_boxplot(box, colors(map_idx))
                for k in range(n_subjs):
                    plt.scatter(x_pos - 0.3, vals[k], color=colors(map_idx), s=5)

        for sd_idx in range(n_sds):
            sub_data = plot_data[sd_idx, :, pair_idx, :]
            for map_idx in range(1, n_maps):
                if np.all(np.isnan(sub_data[map_idx])):
                    continue
                t_stat, p_val = stats.ttest_ind(sub_data[0], sub_data[map_idx], equal_var=False, nan_policy='omit')
                bayes10, bayes01 = _bayes_factors(sub_data[0], sub_data[map_idx], paired=False)
                print(f"Pair: {pair_titles[pair_idx]}, SD: {sd_labels[sd_idx]}, "
                      f"Comparison: {map_labels[0]} vs {map_labels[map_idx]} - "
                      f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 6)}, BF10: {bayes10:.4f}, BF01: {bayes01:.4f}")

                y_max = np.nanmax(plot_data_in_r[sd_idx, :, pair_idx, :]) + 0.08 * map_idx
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(sd_idx * 4, sd_idx * 4 + map_idx * 0.8, y_max, _format_pval(p_val),
                                   alpha=alpha, fontsize=6)

                t_stat0, p_val0 = stats.ttest_1samp(sub_data[map_idx], 0, nan_policy='omit')
                star_x = sd_idx * 4 + map_idx * 0.8
                star_y = np.nanmin(plot_data_in_r[sd_idx, :, pair_idx, :]) - 0.05
                star, star_alpha = _stars_for_pval(p_val0)
                plt.annotate(star, (star_x, star_y), ha='center', size=8, fontweight='bold', alpha=star_alpha)

        plt.xticks([sd_idx * 4 + 1.2 for sd_idx in range(n_sds)], sd_labels, fontsize=12)
        plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
        plt.xlabel('Accuracy shift (SD)', fontsize=14, fontweight='bold')
        plt.ylabel('r(same subject) − r(other subjects)', fontsize=12, fontweight='bold')
        plt.title(f'Correlation consistency - {pair_titles[pair_idx]}', fontsize=14, fontweight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        _save_pdf(path / f'corr_btw_var_accuracy_control_{pair_fnames[pair_idx]}_split_{split_by}.pdf')


def plot_rank_across_metric_consistency_accuracy_control(all_maps, sds, path):
    """ Accuracy-control counterpart of plot_rank_across_metric_consistency (MNIST only).

    Note: get_rank_results('subj', 'inst', 'var') broadcasts a single-pair
    model's value across all pair slots (same behavior as the original
    function relies on), so only the Acc-Conf pair (index 0) - or map_idx <= 1
    (Human-Human/Human-RTNet) for the other pairs - is ever displayed/tested,
    mirroring the original's guard conditions.
    """
    n_sds = len(all_maps)
    n_maps = len(all_maps[0]) + 1
    n_boots, n_pairs, n_subjs = all_maps[0][0].get_corr_results('subj', 'inst', 'subj', 'var').mat.shape

    plot_data = np.empty(shape=(n_sds, n_maps, n_boots, n_pairs))
    plot_data.fill(np.nan)
    for sd_idx, data in enumerate(all_maps):
        for map_idx in range(n_maps):
            if map_idx == 0:
                map_data = data[map_idx].get_rank_results('subj', 'subj', 'var').mat
                plot_data[sd_idx, map_idx] = map_data
            else:
                map_data = data[map_idx-1].get_rank_results('subj', 'inst', 'var').mat
                plot_data[sd_idx, map_idx] = map_data

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS
    sd_labels = [str(sd) if sd != 0 else 'Std' for sd in sds]
    pair_titles = ['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence']
    pair_fnames = ['acc_conf', 'acc_rt', 'rt_conf']

    for pair_idx in range(n_pairs):
        plt.figure(figsize=(7, 4))

        for sd_idx in range(n_sds):
            for map_idx in range(n_maps):
                if pair_idx != 0 and not (map_idx <= 1):
                    continue
                x_pos = sd_idx * 4 + map_idx * 0.8
                box = plt.boxplot(plot_data[sd_idx, map_idx, :, pair_idx], positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
                _style_boxplot(box, colors(map_idx))

        for sd_idx in range(n_sds):
            for map_idx in range(1, n_maps):
                if pair_idx != 0 and not (map_idx == 1):
                    continue
                diff = plot_data[sd_idx, 0, :, pair_idx] - plot_data[sd_idx, map_idx, :, pair_idx]
                p_val = 2 * min(np.mean(diff >= 0), np.mean(diff < 0))
                ci_lower = np.percentile(diff, 2.5)
                ci_upper = np.percentile(diff, 97.5)
                print(f"Pair: {pair_titles[pair_idx]}, SD: {sd_labels[sd_idx]}, "
                      f"Comparison: {map_labels[0]} vs {map_labels[map_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

                y_max = np.nanmax(plot_data[sd_idx, :, :, pair_idx]) + 10 * map_idx
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(sd_idx * 4, sd_idx * 4 + map_idx * 0.8, y_max, _format_pval_simple(p_val),
                                   alpha=alpha, fontsize=6)

        plt.xticks([sd_idx * 4 + 1.2 for sd_idx in range(n_sds)], sd_labels, fontsize=12)
        plt.xlabel('Accuracy shift (SD)', fontsize=14, fontweight='bold')
        plt.ylabel('Rank consistency metric', fontsize=12)
        plt.title(f'Rank consistency - {pair_titles[pair_idx]}', fontsize=14, fontweight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        _save_pdf(path / f'rank_btw_var_accuracy_control_{pair_fnames[pair_idx]}.pdf')


def _plot_grouped_sd_by_arch(ax, data, sds, baseline_idx, colors, arch_labels, gap=2, marker_size=8,
                              test_zero=False):
    """ Draw one metric's panel for the grouped accuracy-control figures: a
    mean marker with SEM error bar + connecting line per architecture across
    the five sd levels, a single dotted red line for human consistency, and
    up to three significance markers:
      1. exact p-values in brackets above the markers - each non-baseline sd
         level vs. that architecture's own sd=0 (standard model) baseline.
      2. red stars just above the red human-consistency line - each sd level
         vs. pooled human consistency.
      3. (only when test_zero=True) black stars just below the y=0 line -
         each sd level tested against 0.
    The five sd levels are told apart by a light-to-bright alpha gradient
    (most negative shift = lightest) instead of x-axis tick labels.

    data: (n_sds, n_arch + 1, n_samples) array for a single metric, where
    index 0 along axis 1 is the human reference and indices 1..n_arch are
    the model architectures (in arch_labels order).
    """
    n_sds = data.shape[0]
    n_arch = data.shape[1] - 1
    block_width = n_sds - 1
    block_starts = [i * (block_width + gap) for i in range(n_arch)]
    marker_alphas = np.linspace(0.25, 1.0, n_sds)

    human_vals = data[:, 0, :]
    human_pooled = human_vals[~np.isnan(human_vals)]
    human_mean = np.nanmean(human_pooled) if human_pooled.size else np.nan
    if human_pooled.size:
        ax.axhline(human_mean, color='red', linestyle='dotted', linewidth=1.5, alpha=0.85, zorder=1)

    model_vals = data[:, 1:, :]
    has_data = np.any(~np.isnan(model_vals))
    global_min = np.nanmin(model_vals) if has_data else 0
    global_max = np.nanmax(model_vals) if has_data else 1
    y_range = (global_max - global_min) or 1
    bracket_y0 = global_max + 0.06 * y_range
    bracket_step = 0.07 * y_range
    max_bracket_level = max(baseline_idx, n_sds - 1 - baseline_idx)
    human_star_y = human_mean + 0.03 * y_range if human_pooled.size else None
    zero_star_y = -0.04 * y_range if test_zero else None

    for arch_idx in range(n_arch):
        map_idx = arch_idx + 1
        color = colors(map_idx)
        xs = [block_starts[arch_idx] + sd_idx for sd_idx in range(n_sds)]
        sd_dists = [data[sd_idx, map_idx, :] for sd_idx in range(n_sds)]

        if np.all(np.isnan(np.concatenate(sd_dists))):
            continue  # nothing to draw for this architecture in this panel (e.g. RT for AlexNet/ResNet18)

        means = []
        for sd_idx in range(n_sds):
            vals = sd_dists[sd_idx]
            valid_vals = vals[~np.isnan(vals)]
            if valid_vals.size == 0:
                means.append(np.nan)
                continue
            mean_val = np.mean(valid_vals)
            sem_val = sem(valid_vals) if valid_vals.size > 1 else 0
            marker_alpha = marker_alphas[sd_idx]
            ax.errorbar(xs[sd_idx], mean_val, yerr=sem_val, fmt='o', color=color,
                        markersize=marker_size, markeredgewidth=0, elinewidth=2.5,
                        capsize=4, capthick=2.5, alpha=marker_alpha, zorder=3)
            means.append(mean_val)

        valid_pts = [(x, m) for x, m in zip(xs, means) if not np.isnan(m)]
        if len(valid_pts) > 1:
            vx, vy = zip(*valid_pts)
            ax.plot(vx, vy, color=color, linewidth=3.5, alpha=0.85, zorder=2)

        # Stat 1: each non-baseline sd level vs. this architecture's own sd=0
        # baseline - exact p-values in brackets above the markers. Left and
        # right sides of the baseline reuse the same bracket heights
        # (nearest-left/nearest-right share a level, farthest-left/
        # farthest-right share the next) since they never overlap in x,
        # which keeps the stack compact.
        baseline_vals = sd_dists[baseline_idx]
        left_order = sorted((i for i in range(n_sds) if i < baseline_idx), key=lambda i: baseline_idx - i)
        right_order = sorted((i for i in range(n_sds) if i > baseline_idx), key=lambda i: i - baseline_idx)
        for side_order in (left_order, right_order):
            for level, sd_idx in enumerate(side_order, start=1):
                vals = sd_dists[sd_idx]
                if np.all(np.isnan(vals)) or np.all(np.isnan(baseline_vals)):
                    continue
                _, p_val = stats.ttest_ind(vals, baseline_vals, equal_var=False, nan_policy='omit')
                _, alpha = _stars_for_pval(p_val)
                y = bracket_y0 + bracket_step * level
                _annotate_bracket(xs[sd_idx], xs[baseline_idx], y, _format_pval(p_val),
                                   ax=ax, alpha=alpha, fontsize=6)

        # Stat 2: each sd level vs. pooled human consistency - red stars just
        # above the red human-consistency line.
        if human_pooled.size >= 2:
            for sd_idx in range(n_sds):
                vals = sd_dists[sd_idx]
                if np.all(np.isnan(vals)):
                    continue
                _, p_val = stats.ttest_ind(vals, human_pooled, equal_var=False, nan_policy='omit')
                star, alpha = _stars_for_pval(p_val)
                ax.annotate(star, (xs[sd_idx], human_star_y), ha='center', va='bottom', size=8,
                            fontweight='bold', color='red', alpha=alpha, zorder=4)

        # Stat 3: each sd level tested against 0 - black stars just below the
        # y=0 line. Only meaningful for panels with a zero-centered metric
        # (correlation-difference panels), gated by test_zero.
        if test_zero:
            for sd_idx in range(n_sds):
                valid_vals = sd_dists[sd_idx][~np.isnan(sd_dists[sd_idx])]
                if valid_vals.size < 2:
                    continue
                _, p_val = stats.ttest_1samp(valid_vals, 0)
                star, alpha = _stars_for_pval(p_val)
                ax.annotate(star, (xs[sd_idx], zero_star_y), ha='center', va='top', size=8,
                            fontweight='bold', color='black', alpha=alpha, zorder=4)

    ax.set_xticks([bs + block_width / 2 for bs in block_starts])
    ax.set_xticklabels(arch_labels, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', which='major', length=0, pad=10)

    # Text annotations (stars, brackets) don't participate in axes autoscale,
    # so the y-limits must be set explicitly wide enough to fit them - the
    # default autoscale only covers the mean+/-SEM markers/line, which is far
    # narrower than the star/bracket/human-star/zero-star band computed above.
    bottom = global_min - 0.05 * y_range
    if zero_star_y is not None:
        bottom = min(bottom, zero_star_y - 0.05 * y_range)
    top = bracket_y0 + bracket_step * max_bracket_level + 0.05 * y_range
    if human_star_y is not None:
        top = max(top, human_star_y + 0.05 * y_range)
    ax.set_ylim(bottom, top)


def _add_grouped_legend(fig, sds):
    """ Legend for the grouped accuracy-control figures: the human-consistency
    reference line, plus the light-to-bright alpha gradient that stands in for
    the sd x-axis labels. """
    alphas = np.linspace(0.25, 1.0, len(sds))
    handles = [Line2D([0], [0], color='red', linestyle='dotted', linewidth=1.5, label='Human consistency')]
    for sd, alpha in zip(sds, alphas):
        label = 'Standard (0)' if sd == 0 else f'SD {sd:+d}'
        handles.append(Patch(facecolor='black', alpha=alpha, label=label))
    fig.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)


def plot_corr_within_metric_consistency_accuracy_control_grouped(all_maps, sds, path, split_by='rand'):
    """ Single-figure, grouped-layout counterpart of
    plot_corr_within_metric_consistency_accuracy_control (kept separate/unmodified).

    One figure, one panel per metric (accuracy/confidence/RT); within each panel,
    the x-axis is grouped by model architecture, and within each architecture the
    five accuracy-shift levels are plotted as a mean +/- SEM marker and connected
    by a line. Human consistency is a single red dotted reference line. Three
    significance markers are drawn per architecture: exact p-values in brackets
    above the markers (each non-zero shift level vs. the architecture's own sd=0
    baseline), red stars just above the human-consistency line (each level vs.
    pooled human consistency), and black stars just below the y=0 line (each
    level tested against 0).
    """
    n_sds = len(all_maps)
    n_maps = len(all_maps[0]) + 1
    n_boots, n_metrics, n_subjs = all_maps[0][0].get_corr_results('subj', 'inst', 'subj', 'split', split_by=split_by).mat.shape

    plot_data = np.empty(shape=(2, n_sds, n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)
    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for sd_idx, data in enumerate(all_maps):
            for map_idx in range(n_maps):
                if map_idx == 0:
                    map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'split', split_by=split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    plot_data[type_idx, sd_idx, map_idx] = map_data
                else:
                    map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'split', split_by=split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.nanmean(map_data, axis=0)
                    try:
                        plot_data[type_idx, sd_idx, map_idx] = map_data
                    except ValueError:
                        plot_data[type_idx, sd_idx, map_idx, :2] = map_data

    plot_data = -np.diff(plot_data, axis=0)
    plot_data = np.squeeze(plot_data, axis=0)  # (n_sds, n_maps, n_metrics, n_subjs)
    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson')

    arch_labels = ['RTNet', 'AlexNet', 'ResNet18']
    colors = _COLORS
    met_titles = ['Accuracy', 'Confidence', 'Reaction time']
    baseline_idx = sds.index(0)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 3))
    for met_idx in range(n_metrics):
        ax = axes[met_idx]
        _plot_grouped_sd_by_arch(ax, plot_data_in_r[:, :, met_idx, :], sds, baseline_idx, colors, arch_labels,
                                  test_zero=True)
        ax.axhline(0, color='black', linestyle='dotted', linewidth=1, alpha=0.4, zorder=0)
        ax.set_title(met_titles[met_idx], fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if met_idx == 0:
            ax.set_ylabel('r(same subject) − r(other subjects)', fontsize=11, fontweight='bold')

    _add_grouped_legend(fig, sds)
    fig.suptitle('Correlation consistency across accuracy shift', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pdf(path / f'corr_within_metric_accuracy_control_grouped_split_{split_by}.pdf')


def plot_rank_within_metric_consistency_accuracy_control_grouped(all_maps, sds, path):
    """ Single-figure, grouped-layout counterpart of
    plot_rank_within_metric_consistency_accuracy_control (kept separate/unmodified).
    See plot_corr_within_metric_consistency_accuracy_control_grouped for the
    shared layout and significance-marker description.
    """
    n_sds = len(all_maps)
    n_maps = len(all_maps[0]) + 1
    n_boots, n_metrics, n_subjs = all_maps[0][0].get_corr_results('subj', 'inst', 'subj', 'split').mat.shape

    plot_data = np.empty(shape=(n_sds, n_maps, n_boots, n_metrics))
    plot_data.fill(np.nan)
    for sd_idx, data in enumerate(all_maps):
        for map_idx in range(n_maps):
            if map_idx == 0:
                map_data = data[map_idx].get_rank_results('subj', 'subj', 'split').mat
                plot_data[sd_idx, map_idx] = map_data
            else:
                map_data = data[map_idx-1].get_rank_results('subj', 'inst', 'split').mat
                try:
                    plot_data[sd_idx, map_idx] = map_data
                except ValueError:
                    plot_data[sd_idx, map_idx, :, :2] = map_data

    arch_labels = ['RTNet', 'AlexNet', 'ResNet18']
    colors = _COLORS
    met_titles = ['Accuracy', 'Confidence', 'Reaction time']
    baseline_idx = sds.index(0)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 3))
    for met_idx in range(n_metrics):
        ax = axes[met_idx]
        _plot_grouped_sd_by_arch(ax, plot_data[:, :, :, met_idx], sds, baseline_idx, colors, arch_labels)
        ax.set_title(met_titles[met_idx], fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if met_idx == 0:
            ax.set_ylabel('Rank consistency metric', fontsize=11, fontweight='bold')

    _add_grouped_legend(fig, sds)
    fig.suptitle('Rank consistency across accuracy shift', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pdf(path / 'rank_within_metric_accuracy_control_grouped.pdf')


def plot_corr_across_metric_consistency_accuracy_control_grouped(all_maps, sds, path, split_by='rand'):
    """ Grouped-layout counterpart of plot_corr_across_metric_consistency_accuracy_control
    (kept separate/unmodified), using the same panel-by-metric-pair /
    group-by-architecture / line-across-sd-levels layout as
    plot_corr_within_metric_consistency_accuracy_control_grouped. AlexNet and
    ResNet18 lack RT, so their markers/stats are only ever populated for the
    Acc-Conf pair (index 0), same as the original function.
    """
    n_sds = len(all_maps)
    n_maps = len(all_maps[0]) + 1
    n_boots, n_pairs, n_subjs = all_maps[0][0].get_corr_results('subj', 'inst', 'subj', 'var', split_by).mat.shape

    plot_data = np.empty(shape=(2, n_sds, n_maps, n_pairs, n_subjs))
    plot_data.fill(np.nan)
    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for sd_idx, data in enumerate(all_maps):
            for map_idx in range(n_maps):
                if map_idx == 0:
                    map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'var', split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.mean(map_data, axis=0)
                    plot_data[type_idx, sd_idx, map_idx] = map_data
                else:
                    map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'var', split_by).mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.mean(map_data, axis=0)
                    if map_idx > 1 and n_pairs > 1:
                        plot_data[type_idx, sd_idx, map_idx, 0] = map_data
                    else:
                        plot_data[type_idx, sd_idx, map_idx] = map_data

    plot_data = -np.diff(plot_data, axis=0)
    plot_data = np.squeeze(plot_data, axis=0)  # (n_sds, n_maps, n_pairs, n_subjs)
    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson')

    arch_labels = ['RTNet', 'AlexNet', 'ResNet18']
    colors = _COLORS
    pair_titles = ['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence']
    baseline_idx = sds.index(0)

    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 3))
    axes = np.atleast_1d(axes)
    for pair_idx in range(n_pairs):
        ax = axes[pair_idx]
        _plot_grouped_sd_by_arch(ax, plot_data_in_r[:, :, pair_idx, :], sds, baseline_idx, colors, arch_labels,
                                  test_zero=True)
        ax.axhline(0, color='black', linestyle='dotted', linewidth=1, alpha=0.4, zorder=0)
        ax.set_title(pair_titles[pair_idx], fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if pair_idx == 0:
            ax.set_ylabel('r(same subject) − r(other subjects)', fontsize=11, fontweight='bold')

    _add_grouped_legend(fig, sds)
    fig.suptitle('Correlation consistency across accuracy shift', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pdf(path / f'corr_across_metric_accuracy_control_grouped_split_{split_by}.pdf')


def plot_rank_across_metric_consistency_accuracy_control_grouped(all_maps, sds, path):
    """ Grouped-layout counterpart of plot_rank_across_metric_consistency_accuracy_control
    (kept separate/unmodified). Same caveat as the original: get_rank_results
    broadcasts a single-pair model's value across all pair slots, so for
    AlexNet/ResNet18 (which lack RT) only the Acc-Conf pair is meaningful -
    the other pair slots are explicitly masked to NaN here so they are
    skipped by _plot_grouped_sd_by_arch instead of showing spurious boxes.
    """
    n_sds = len(all_maps)
    n_maps = len(all_maps[0]) + 1
    n_boots, n_pairs, n_subjs = all_maps[0][0].get_corr_results('subj', 'inst', 'subj', 'var').mat.shape

    plot_data = np.empty(shape=(n_sds, n_maps, n_boots, n_pairs))
    plot_data.fill(np.nan)
    for sd_idx, data in enumerate(all_maps):
        for map_idx in range(n_maps):
            if map_idx == 0:
                map_data = data[map_idx].get_rank_results('subj', 'subj', 'var').mat
                plot_data[sd_idx, map_idx] = map_data
            else:
                map_data = data[map_idx-1].get_rank_results('subj', 'inst', 'var').mat
                plot_data[sd_idx, map_idx] = map_data
    if n_pairs > 1:
        plot_data[:, 2:, :, 1:] = np.nan  # AlexNet/ResNet18 only have a meaningful Acc-Conf pair

    arch_labels = ['RTNet', 'AlexNet', 'ResNet18']
    colors = _COLORS
    pair_titles = ['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence']
    baseline_idx = sds.index(0)

    fig, axes = plt.subplots(1, n_pairs, figsize=(3 * n_pairs, 3))
    axes = np.atleast_1d(axes)
    for pair_idx in range(n_pairs):
        ax = axes[pair_idx]
        _plot_grouped_sd_by_arch(ax, plot_data[:, :, :, pair_idx], sds, baseline_idx, colors, arch_labels)
        ax.set_title(pair_titles[pair_idx], fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if pair_idx == 0:
            ax.set_ylabel('Rank consistency metric', fontsize=11, fontweight='bold')

    _add_grouped_legend(fig, sds)
    fig.suptitle('Rank consistency across accuracy shift', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pdf(path / 'rank_across_metric_accuracy_control_grouped.pdf')


def _z_subj_to_inst_splits(obj, split_by):
    """ Elementwise Fisher-z-transform of an IndiMap object's subj-to-inst
    mapping matrix, keeping its own two random-image-split halves intact:
    (boot, split, metric, subj, inst) in z-space. Mirrors the elementwise
    r2z step in CorrMap.do_corr_analysis (including its ==10 NaN sentinel,
    left untouched here so downstream handling matches corr_btw_split). """
    mat = obj.get_corr_map('subj', 'inst', split_by).mat
    return stat_func.r2z(mat, metric='pearson')


def _crossed_split_consistency(data_i, data_j):
    """ Cross-SD counterpart of CorrMap.corr_btw_split, used for every cell
    (i, j) of the SD-by-SD matrix: split 1 of condition i against split 2 of
    condition j, a single non-averaged pairing. With data_i is data_j (the
    diagonal, sd_i == sd_j), this reproduces the standard within-condition
    split-half reliability exactly (verified to match IndiMap.get_corr_results
    to floating-point precision).

    This crossed pairing - rather than corr_btw_var's matched-split-then-
    average approach - is required for the off-diagonal (sd_i != sd_j) cells
    to be meaningfully directional: with a shared split index on both sides
    (corr_btw_var's pattern), full_mat(j, i) is exactly full_mat(i, j)
    transposed, so once "other subjects" is averaged over all subjects into
    one cell value, cell(i, j) and cell(j, i) are provably identical (verified
    numerically to float precision) - the matrix is forced symmetric no
    matter what, which defeats the point of a directional split-1-vs-split-2
    matrix. Crossing the splits (this function) breaks that forced identity:
    full_mat(i, j) and full_mat(j, i) are built from disjoint split
    combinations, not transposes of each other, so (i, j) and (j, i) come out
    genuinely (if closely, since split 1 and split 2 are statistically
    similar random halves) different.

    data_i, data_j: (boot, split, met, subj, inst), z-space (own-condition
    random-image-split halves intact, as returned by _z_subj_to_inst_splits).
    Returns (within, between), each (boot, met, subj), in r-space.
    """
    n_boot, _, n_met, n_subj, _ = data_i.shape
    within = np.empty((n_boot, n_met, n_subj))
    between = np.empty((n_boot, n_met, n_subj))

    for b in range(n_boot):
        for m in range(n_met):
            full_mat = map_func.compute_full_corr_matrix(data_i[b, 0, m], data_j[b, 1, m])
            within[b, m, :] = np.diagonal(full_mat)

            fm = stat_func.r2z(full_mat, 'pearson')
            np.fill_diagonal(fm, np.nan)
            between_col = np.nanmean(fm, axis=0)
            between_col[between_col == 10] = np.nan
            between[b, m, :] = stat_func.z2r(between_col, 'pearson')

    within_z = stat_func.r2z(within, 'pearson')
    within_z[within_z == 10] = np.nan
    within = stat_func.z2r(within_z, 'pearson')

    return within, between


def _plot_sd_matrix_heatmap(grid_r, grid_p, sd_labels, title):
    n_sds = len(sd_labels)
    vmax = np.nanmax(np.abs(grid_r))
    vmax = vmax if vmax > 0 else 1.0

    plt.figure(figsize=(5.5, 4.5))
    ax = plt.gca()
    im = ax.imshow(grid_r, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    for i in range(n_sds):
        for j in range(n_sds):
            if np.isnan(grid_r[i, j]):
                continue
            star, alpha = _stars_for_pval(grid_p[i, j])
            ax.text(j, i, f'{grid_r[i, j]:.2f}\n{star}', ha='center', va='center',
                    fontsize=8, alpha=max(alpha, 0.6), color='black')

    ax.set_xticks(range(n_sds))
    ax.set_xticklabels(sd_labels)
    ax.set_yticks(range(n_sds))
    ax.set_yticklabels(sd_labels)
    ax.set_xlabel('Accuracy shift (SD) - split 2', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy shift (SD) - split 1', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('r(same subject) − r(other subjects)', fontsize=10, fontweight='bold')
    plt.tight_layout()


def plot_corr_sd_matrix_consistency_accuracy_control(all_maps, sds, path, split_by='rand'):
    """ Cross-SD counterpart of plot_corr_within_metric_consistency_accuracy_control
    (MNIST only).

    Every cell (i, j) uses _crossed_split_consistency: split 1 of the SD
    assigned to the row against split 2 of the SD assigned to the column, a
    single non-averaged pairing. The diagonal (i == j) falls directly out of
    this same definition (split 1 vs split 2 of one condition), reproducing
    the standard within-SD split-half reliability with no special-casing
    (verified to match IndiMap.get_corr_results to floating-point precision).

    Note this deliberately does NOT mirror CorrMap.corr_btw_var's matched-
    split-then-average approach, even though sd_i and sd_j are conceptually
    a second "variable" much like a second metric: corr_btw_var shares one
    split index between the two things being compared, which forces
    full_mat(j, i) to be exactly full_mat(i, j) transposed - so once "other
    subjects" is averaged over all subjects into a single cell value, cell
    (i, j) and cell (j, i) become provably, exactly identical (verified
    numerically). That collapses this matrix to always being symmetric,
    which defeats the point of a directional split-1-(row) vs split-2-
    (column) matrix. Crossing the splits avoids that forced identity - (i, j)
    and (j, i) come out genuinely (if closely) different.

    One n_sds x n_sds heatmap (row = SD for split 1, column = SD for split 2)
    is produced per architecture per behavioral metric.
    """
    n_sds = len(all_maps)
    model_labels = ['RTNet', 'AlexNet', 'ResNet18']
    met_titles_all = ['Accuracy', 'Confidence', 'Reaction time']
    met_fnames_all = ['acc', 'conf', 'rt']
    sd_labels = [str(sd) for sd in sds]

    for model_idx, model_label in enumerate(model_labels):
        n_boots, n_metrics, n_subjs = all_maps[0][model_idx].get_corr_results(
            'subj', 'inst', 'subj', 'split', split_by=split_by).mat.shape

        z_splits = [_z_subj_to_inst_splits(all_maps[sd_idx][model_idx], split_by) for sd_idx in range(n_sds)]

        grid_r = np.full((n_metrics, n_sds, n_sds), np.nan)
        grid_p = np.full((n_metrics, n_sds, n_sds), np.nan)

        for i in range(n_sds):
            for j in range(n_sds):
                same, other = _crossed_split_consistency(z_splits[i], z_splits[j])

                same_z = stat_func.r2z(same, metric='pearson')
                other_z = stat_func.r2z(other, metric='pearson')
                subj_z = np.nanmean(same_z - other_z, axis=0)  # (met, subj), averaged over bootstraps

                for met_idx in range(n_metrics):
                    vals = subj_z[met_idx]
                    if np.all(np.isnan(vals)):
                        continue
                    grid_r[met_idx, i, j] = stat_func.z2r(np.nanmean(vals), metric='pearson')
                    _, grid_p[met_idx, i, j] = stats.ttest_1samp(vals, 0, nan_policy='omit')

        for met_idx in range(n_metrics):
            if np.all(np.isnan(grid_r[met_idx])):
                continue
            _plot_sd_matrix_heatmap(grid_r[met_idx], grid_p[met_idx], sd_labels,
                                     f'{model_label} - {met_titles_all[met_idx]}')
            _save_pdf(path / f'corr_sd_matrix_accuracy_control_{model_label.lower()}'
                             f'_{met_fnames_all[met_idx]}_split_{split_by}.pdf')


def _add_control_legend(colors, n_maps):
    legend_elements = [
        Line2D([0], [0], marker='*', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label='Standard')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=7, frameon=False)


def _add_same_other_legend():
    legend_elements = [
        Patch(facecolor='grey', alpha=0.5, label='Same subject'),
        Patch(facecolor='grey', alpha=0.5, hatch='///', label='Other subjects'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=False)


def _plot_same_vs_other_debug(untr_z_raw, n_maps, n_groups, n_subjs, colors, x_pos_fn, dot_offset=0.0):
    """ Debug scatter+box of the untrained network's raw same-subject vs
    other-subjects values (pre-diff, in r-space), side by side per group. """
    same_other_r = stat_func.z2r(untr_z_raw, metric='pearson')  # (2, n_maps, n_groups, n_subjs)
    for map_idx in range(n_maps):
        for group_idx in range(n_groups):
            for type_idx in range(2):
                vals = same_other_r[type_idx, map_idx, group_idx, :]
                if np.all(np.isnan(vals)):
                    continue
                x_pos = x_pos_fn(group_idx, map_idx) + (type_idx - 0.5) * 0.3
                box = plt.boxplot(vals, positions=[x_pos], widths=0.25, patch_artist=True, showfliers=False)
                _style_boxplot(box, colors(map_idx))
                if type_idx == 1:
                    for patch in box['boxes']:
                        patch.set_hatch('///')
                for k in range(n_subjs):
                    plt.scatter(x_pos - dot_offset, vals[k], color=colors(map_idx), s=4, alpha=0.6, zorder=3)


def plot_corr_within_metric_consistency_control(standard_data, merged_control, name, path, split_by):
    """ Control-instance counterpart of plot_corr_within_metric_consistency.

    standard_data: [rtnet, alexnet, resnet18] IndiMap objects (variant='standard').
    merged_control: [rtnet, alexnet, resnet18] merged-result dicts produced by
    analyze_control.py's merge_combo (already r2z-averaged over the 60 control
    instances, n_boots dimension intact). Boxes show the control distribution;
    a star overlays the single standard-model reference (Human-Human is skipped
    since it does not depend on which model's control instances were used).
    """
    n_maps = len(standard_data) + 1
    n_boots, n_metrics, n_subjs = standard_data[0].get_corr_results('subj', 'inst', 'subj', 'split', split_by=split_by).mat.shape

    std_z = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    std_z.fill(np.nan)
    ctrl_z = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    ctrl_z.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                std_mat = standard_data[map_idx].get_corr_results('subj', 'subj', map_type, 'split', split_by=split_by).mat
                std_z[type_idx, map_idx] = np.nanmean(stat_func.r2z(std_mat, metric='pearson'), axis=0)

                ctrl_mat = merged_control[map_idx][f'corr_split_subj_{map_type}_{split_by}']
                ctrl_z[type_idx, map_idx] = np.nanmean(ctrl_mat, axis=0)
            else:
                std_mat = standard_data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'split', split_by=split_by).mat
                std_mat = np.nanmean(stat_func.r2z(std_mat, metric='pearson'), axis=0)
                try:
                    std_z[type_idx, map_idx] = std_mat
                except ValueError:
                    std_z[type_idx, map_idx, :2] = std_mat

                ctrl_mat = merged_control[map_idx-1][f'corr_split_inst_{map_type}_{split_by}']
                ctrl_mat = np.nanmean(ctrl_mat, axis=0)
                try:
                    ctrl_z[type_idx, map_idx] = ctrl_mat
                except ValueError:
                    ctrl_z[type_idx, map_idx, :2] = ctrl_mat

    std_z = np.squeeze(-np.diff(std_z, axis=0), axis=0)    # (n_maps, n_metrics, n_subjs)
    ctrl_z = np.squeeze(-np.diff(ctrl_z, axis=0), axis=0)  # (n_maps, n_metrics, n_subjs)
    std_r = stat_func.z2r(std_z, metric='pearson')
    ctrl_r = stat_func.z2r(ctrl_z, metric='pearson')

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS

    plt.figure(figsize=(6, 4))

    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            vals = ctrl_r[map_idx, met_idx, :]
            if np.all(np.isnan(vals)):
                continue
            x_pos = met_idx * 4 + map_idx * 0.8
            box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, colors(map_idx))
            for k in range(n_subjs):
                plt.scatter(x_pos - 0.3, vals[k], color=colors(map_idx), s=5)

            if map_idx == 0:
                continue  # Human-Human standard == control reference, redundant to show
            std_vals = std_r[map_idx, met_idx, :]
            if not np.all(np.isnan(std_vals)):
                plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=80,
                            color=colors(map_idx), edgecolor='black', linewidth=1, zorder=5)

    data_all = np.concatenate([ctrl_r[np.isfinite(ctrl_r)], std_r[np.isfinite(std_r)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    vs_zero_y = data_min - 0.15 * data_range
    std_vs_ctrl_y = vs_zero_y - 0.12 * data_range
    max_bracket = data_max

    for met_idx in range(n_metrics):
        sub_data = ctrl_z[:, met_idx, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):
                if map_labels[i] != 'Human-Human':
                    continue
                if np.all(np.isnan(sub_data[j])):
                    continue

                # two-sided: Human-Human vs Human-Model
                t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]}, "
                      f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                      f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 8)}, "
                      f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}")

                y_max = np.nanmax(ctrl_r[:, met_idx, :]) + 0.08 * data_range * j
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(met_idx * 4 + i * 0.8, met_idx * 4 + j * 0.8, y_max, _format_pval(p_val),
                                   alpha=alpha, fontsize=9, xytext=(0, 3))

                # one-sided: Human-Model vs 0
                t_stat0, p_val0 = stats.ttest_1samp(sub_data[j], 0, alternative='greater', nan_policy='omit')
                x_pos0 = (met_idx * 4 + j * 0.8) + 0.15
                star, star_alpha = _stars_for_pval(p_val0)
                plt.annotate(star, (x_pos0, vs_zero_y), ha='center', size=8, alpha=star_alpha, fontweight='bold')

                # two-sided: standard vs control, per box
                std_sc_vals = std_z[j, met_idx, :]
                if not np.all(np.isnan(std_sc_vals)):
                    t_stat_sc, p_val_sc = stats.ttest_ind(sub_data[j], std_sc_vals, equal_var=False, nan_policy='omit')
                    print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]}, "
                          f"Comparison: {map_labels[j]} standard vs control - "
                          f"t-stat: {t_stat_sc:.4f}, p-value: {_format_pval_print(p_val_sc, 8)}")
                    plt.annotate(_format_pval(p_val_sc), (met_idx * 4 + j * 0.8, std_vs_ctrl_y),
                                 ha='center', size=7)

    plt.xticks([1.2, 5.2, 8.4], ['Accuracy', 'Confidence', 'RT'], fontsize=12)
    plt.xlim(-1, 10)
    plt.ylim(std_vs_ctrl_y - 0.08 * data_range, max_bracket + 0.15 * data_range)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('r(same subject) − r(other subjects)', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency (control)', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_bs_control_{name}_split_{split_by}.pdf')


def plot_rank_within_metric_consistency_control(standard_data, merged_control, name, path):
    """ Control-instance counterpart of plot_rank_within_metric_consistency. """
    n_maps = len(standard_data) + 1
    n_boots, n_metrics, n_subjs = standard_data[0].get_corr_results('subj', 'inst', 'subj', 'split').mat.shape

    std_data = np.empty(shape=(n_maps, n_boots, n_metrics))
    std_data.fill(np.nan)
    ctrl_data = np.empty(shape=(n_maps, n_boots, n_metrics))
    ctrl_data.fill(np.nan)

    for map_idx in range(n_maps):
        if map_idx == 0:
            std_data[map_idx] = standard_data[map_idx].get_rank_results('subj', 'subj', 'split').mat
            ctrl_data[map_idx] = merged_control[map_idx]['rank_split_subj']
        else:
            std_mat = standard_data[map_idx-1].get_rank_results('subj', 'inst', 'split').mat
            try:
                std_data[map_idx] = std_mat
            except ValueError:
                std_data[map_idx, :, :2] = std_mat

            ctrl_mat = merged_control[map_idx-1]['rank_split_inst']
            try:
                ctrl_data[map_idx] = ctrl_mat
            except ValueError:
                ctrl_data[map_idx, :, :2] = ctrl_mat

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS

    plt.figure(figsize=(6, 4))

    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            if map_idx > 1 and met_idx > 1:
                continue
            vals = ctrl_data[map_idx, :, met_idx]
            if np.all(np.isnan(vals)):
                continue
            x_pos = met_idx * 4 + map_idx * 0.8
            box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, colors(map_idx))

            if map_idx == 0:
                continue  # Human-Human standard == control reference, redundant to show
            std_vals = std_data[map_idx, :, met_idx]
            if not np.all(np.isnan(std_vals)):
                plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=80,
                            color=colors(map_idx), edgecolor='black', linewidth=1, zorder=5)

    data_all = np.concatenate([ctrl_data[np.isfinite(ctrl_data)], std_data[np.isfinite(std_data)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    # two-sided: standard vs control, per box
    for map_idx in range(1, n_maps):
        for met_idx in range(n_metrics):
            if map_idx > 1 and met_idx > 1:
                continue
            ctrl_vals = ctrl_data[map_idx, :, met_idx]
            std_vals = std_data[map_idx, :, met_idx]
            if np.all(np.isnan(ctrl_vals)) or np.all(np.isnan(std_vals)):
                continue
            diff = ctrl_vals - std_vals
            n_diff = len(diff)
            p_val_sc = 2 * min(
                (len(diff[diff >= 0]) + 1) / (n_diff + 1),
                (len(diff[diff < 0]) + 1) / (n_diff + 1)
            )
            print(f"Comparison: {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} standard vs control - "
                  f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]} - p-value: {_format_pval_print(p_val_sc, 4)}")
            x_pos = met_idx * 4 + map_idx * 0.8
            y_pos = np.nanmin([np.nanmin(ctrl_vals), np.nanmin(std_vals)]) - 0.04 * data_range
            min_annot = min(min_annot, y_pos)
            plt.annotate(_format_pval(p_val_sc), (x_pos, y_pos), ha='center', size=7)

    if name == 'mnist':
        for map_idx in range(1, n_maps):
            diff = ctrl_data[0] - ctrl_data[map_idx]
            for met_idx in range(n_metrics):
                for_proportion = diff[:, met_idx]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} - "
                      f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met_idx == 2 and not (map_idx == 1):
                    continue
                y_max = np.nanmax(ctrl_data[:, :, met_idx]) + 0.08 * data_range * map_idx
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(met_idx * 4, met_idx * 4 + map_idx * 0.8, y_max, _format_pval_simple(p_val),
                                   alpha=alpha, fontsize=9, xytext=(0, 3))

    if name == 'ecoset10':
        for map_idx in range(1, n_maps):
            for met_idx in range(n_metrics):
                for_proportion = ctrl_data[map_idx, :, met_idx]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} - "
                      f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met_idx == 2 and not (map_idx == 1):
                    continue
                x_pos = (met_idx * 4 + map_idx * 0.8)
                y_max = np.nanpercentile(ctrl_data[map_idx, :, met_idx], 0) - 0.08 * data_range
                min_annot = min(min_annot, y_max)
                anno, alpha = _stars_for_pval(p_val)
                plt.annotate(anno, (x_pos, y_max), ha='center', size=9, alpha=alpha, fontweight='bold')

    plt.xticks([1.2, 5.2, 8.4], ['Accuracy', 'Confidence', 'RT'], fontsize=12)
    plt.ylim(min_annot - 0.08 * data_range, max_bracket + 0.15 * data_range)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title('Rank consistency (control)', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    _save_pdf(path / f'rank_btw_bs_control_{name}.pdf')


def plot_corr_across_metric_consistency_control(standard_data, merged_control, name, path, split_by):
    """ Control-instance counterpart of plot_corr_across_metric_consistency. """
    n_maps = len(standard_data) + 1
    n_boots, n_pairs, n_subjs = standard_data[0].get_corr_results('subj', 'inst', 'subj', 'var', split_by).mat.shape

    std_z = np.empty(shape=(2, n_maps, n_pairs, n_subjs))
    std_z.fill(np.nan)
    ctrl_z = np.empty(shape=(2, n_maps, n_pairs, n_subjs))
    ctrl_z.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                std_mat = standard_data[map_idx].get_corr_results('subj', 'subj', map_type, 'var', split_by).mat
                std_z[type_idx, map_idx] = np.mean(stat_func.r2z(std_mat, metric='pearson'), axis=0)

                ctrl_mat = merged_control[map_idx][f'corr_var_subj_{map_type}_{split_by}']
                ctrl_z[type_idx, map_idx] = np.nanmean(ctrl_mat, axis=0)
            else:
                std_mat = standard_data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'var', split_by).mat
                std_mat = np.mean(stat_func.r2z(std_mat, metric='pearson'), axis=0)
                if map_idx > 1 and n_pairs > 1:
                    std_z[type_idx, map_idx, 0] = std_mat
                else:
                    std_z[type_idx, map_idx] = std_mat

                ctrl_mat = merged_control[map_idx-1][f'corr_var_inst_{map_type}_{split_by}']
                ctrl_mat = np.nanmean(ctrl_mat, axis=0)
                if map_idx > 1 and n_pairs > 1:
                    ctrl_z[type_idx, map_idx, 0] = ctrl_mat
                else:
                    ctrl_z[type_idx, map_idx] = ctrl_mat

    std_z = np.squeeze(-np.diff(std_z, axis=0), axis=0)
    ctrl_z = np.squeeze(-np.diff(ctrl_z, axis=0), axis=0)
    std_r = stat_func.z2r(std_z, metric='pearson')
    ctrl_r = stat_func.z2r(ctrl_z, metric='pearson')

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS

    plt.figure(figsize=(6, 4))

    for map_idx in range(n_maps):
        for pair_idx in range(n_pairs):
            vals = ctrl_r[map_idx, pair_idx, :]
            if np.all(np.isnan(vals)):
                continue
            x_pos = pair_idx * 4 + map_idx * 0.8 if pair_idx < 2 else pair_idx * 3.2 + map_idx * 0.8
            box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, colors(map_idx))
            for k in range(n_subjs):
                plt.scatter(x_pos - 0.3, vals[k], color=colors(map_idx), s=5)

            if map_idx == 0:
                continue  # Human-Human standard == control reference, redundant to show
            std_vals = std_r[map_idx, pair_idx, :]
            if not np.all(np.isnan(std_vals)):
                plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=80,
                            color=colors(map_idx), edgecolor='black', linewidth=1, zorder=5)

    data_all = np.concatenate([ctrl_r[np.isfinite(ctrl_r)], std_r[np.isfinite(std_r)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    vs_zero_y = data_min - 0.15 * data_range
    std_vs_ctrl_y = vs_zero_y - 0.12 * data_range
    max_bracket = data_max

    for pair_idx in range(n_pairs):
        sub_data = ctrl_z[:, pair_idx, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):
                if map_labels[i] != 'Human-Human':
                    continue
                if np.all(np.isnan(sub_data[j])):
                    continue

                # two-sided: Human-Human vs Human-Model
                t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                print(f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]}, "
                      f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                      f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 6)}, "
                      f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}")

                if pair_idx < 2:
                    plot_x_pos = [pair_idx * 4, pair_idx * 4 + j * 0.8]
                else:
                    plot_x_pos = [pair_idx * 3.2, pair_idx * 3.2 + j * 0.8]

                y_max = np.nanmax(ctrl_r[:, pair_idx, :]) + 0.08 * data_range * j
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(plot_x_pos[0], plot_x_pos[1], y_max, _format_pval(p_val),
                                   alpha=alpha, fontsize=9, xytext=(0, 3))

                # one-sided: Human-Model vs 0
                t_stat0, p_val0 = stats.ttest_1samp(sub_data[j], 0, alternative='greater', nan_policy='omit')
                if pair_idx < 2:
                    x_pos0 = pair_idx * 4 + j * 0.8 + 0.15
                else:
                    x_pos0 = pair_idx * 3.2 + j * 0.8 + 0.15
                star, star_alpha = _stars_for_pval(p_val0)
                plt.annotate(star, (x_pos0, vs_zero_y), ha='center', size=8, alpha=star_alpha, fontweight='bold')

                # two-sided: standard vs control, per box
                std_sc_vals = std_z[j, pair_idx, :]
                if not np.all(np.isnan(std_sc_vals)):
                    t_stat_sc, p_val_sc = stats.ttest_ind(sub_data[j], std_sc_vals, equal_var=False, nan_policy='omit')
                    print(f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]}, "
                          f"Comparison: {map_labels[j]} standard vs control - "
                          f"t-stat: {t_stat_sc:.4f}, p-value: {_format_pval_print(p_val_sc, 6)}")
                    plt.annotate(_format_pval(p_val_sc), (plot_x_pos[1], std_vs_ctrl_y),
                                 ha='center', size=7)

    plt.xticks([1.2, 4.4, 6.8], ['Acc-Conf', 'Acc-RT', 'Conf-RT'], fontsize=12)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlim(-1, 8.5)
    plt.ylim(std_vs_ctrl_y - 0.08 * data_range, max_bracket + 0.15 * data_range)
    plt.xlabel('Pairs of behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel('r(same subject) − r(other subjects)', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency (control)', fontsize=14, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_var_control_{name}_split_{split_by}.pdf')


def plot_rank_across_metric_consistency_control(standard_data, merged_control, name, path):
    """ Control-instance counterpart of plot_rank_across_metric_consistency. """
    n_maps = len(standard_data) + 1
    n_boots, n_pairs, n_subjs = standard_data[0].get_corr_results('subj', 'inst', 'subj', 'var').mat.shape

    std_data = np.empty(shape=(n_maps, n_boots, n_pairs))
    std_data.fill(np.nan)
    ctrl_data = np.empty(shape=(n_maps, n_boots, n_pairs))
    ctrl_data.fill(np.nan)

    for map_idx in range(n_maps):
        if map_idx == 0:
            std_data[map_idx] = standard_data[map_idx].get_rank_results('subj', 'subj', 'var').mat
            ctrl_data[map_idx] = merged_control[map_idx]['rank_var_subj']
        else:
            std_data[map_idx] = standard_data[map_idx-1].get_rank_results('subj', 'inst', 'var').mat
            ctrl_data[map_idx] = merged_control[map_idx-1]['rank_var_inst']

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS

    plt.figure(figsize=(6, 4))

    for map_idx in range(n_maps):
        for pair_idx in range(n_pairs):
            if pair_idx != 0 and not (map_idx <= 1):
                continue
            x_pos = pair_idx * 4 + map_idx * 0.8 if pair_idx < 2 else pair_idx * 3.2 + map_idx * 0.8
            box = plt.boxplot(ctrl_data[map_idx, :, pair_idx], positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, colors(map_idx))

            if map_idx == 0:
                continue  # Human-Human standard == control reference, redundant to show
            std_vals = std_data[map_idx, :, pair_idx]
            plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=80,
                        color=colors(map_idx), edgecolor='black', linewidth=1, zorder=5)

    data_all = np.concatenate([ctrl_data[np.isfinite(ctrl_data)], std_data[np.isfinite(std_data)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    # two-sided: standard vs control, per box
    for map_idx in range(1, n_maps):
        for pair_idx in range(n_pairs):
            if pair_idx != 0 and not (map_idx <= 1):
                continue
            ctrl_vals = ctrl_data[map_idx, :, pair_idx]
            std_vals = std_data[map_idx, :, pair_idx]
            if np.all(np.isnan(ctrl_vals)) or np.all(np.isnan(std_vals)):
                continue
            diff = ctrl_vals - std_vals
            n_diff = len(diff)
            p_val_sc = 2 * min(
                (len(diff[diff >= 0]) + 1) / (n_diff + 1),
                (len(diff[diff < 0]) + 1) / (n_diff + 1)
            )
            print(f"Comparison: {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} standard vs control - "
                  f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]} - "
                  f"p-value: {_format_pval_print(p_val_sc, 4)}")
            x_pos = pair_idx * 4 + map_idx * 0.8 if pair_idx < 2 else pair_idx * 3.2 + map_idx * 0.8
            y_pos = np.nanmin([np.nanmin(ctrl_vals), np.nanmin(std_vals)]) - 0.04 * data_range
            min_annot = min(min_annot, y_pos)
            plt.annotate(_format_pval(p_val_sc), (x_pos, y_pos), ha='center', size=7)

    if name == 'mnist':
        for map_idx in range(1, n_maps):
            diff = ctrl_data[0] - ctrl_data[map_idx]
            for pair_idx in range(n_pairs):
                if pair_idx != 0 and not (map_idx == 1):
                    continue
                for_proportion = diff[:, pair_idx]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} - "
                      f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if pair_idx < 2:
                    plot_x_pos = [pair_idx * 4, pair_idx * 4 + map_idx * 0.8]
                else:
                    plot_x_pos = [pair_idx * 3.2, pair_idx * 3.2 + map_idx * 0.8]
                y_max = np.nanmax(ctrl_data[:, :, pair_idx]) + 0.08 * data_range * map_idx
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(plot_x_pos[0], plot_x_pos[1], y_max, _format_pval_simple(p_val),
                                   alpha=alpha, fontsize=9, xytext=(0, 3))

    if name == 'ecoset10':
        for map_idx in range(1, n_maps):
            for pair_idx in range(n_pairs):
                if pair_idx != 0 and not (map_idx == 1):
                    continue
                for_proportion = ctrl_data[map_idx, :, pair_idx]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} - "
                      f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if pair_idx < 2:
                    x_pos = pair_idx * 4 + map_idx * 0.8
                else:
                    x_pos = pair_idx * 3.2 + map_idx * 0.8
                y_max = min_annot - 0.08 * data_range
                min_annot = min(min_annot, y_max)
                anno, alpha = _stars_for_pval(p_val)
                plt.annotate(anno, (x_pos, y_max), ha='center', size=9, alpha=alpha, fontweight='bold')

    plt.xticks([1.2, 4.4, 6.8], ['Acc-Conf', 'Acc-RT', 'Conf-RT'], fontsize=12)
    plt.xlim(-1, 8.5)
    plt.ylim(min_annot - 0.08 * data_range, max_bracket + 0.15 * data_range)
    plt.xlabel('Pairs of behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title('Rank consistency (control)', fontsize=14, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    _save_pdf(path / f'rank_btw_var_control_{name}.pdf')


def plot_top_identifiability_untrained(standard_data, untrained_data, name, path):
    """ Untrained-network counterpart of plot_top_identifiability.

    standard_data / untrained_data: [rtnet, alexnet, resnet18] IndiMap objects
    (variant='standard' / variant='untrained'). Boxes show the untrained
    network's per-subject distribution; a star overlays the standard
    (trained) network's reference value. Human-Human is skipped for the star
    since it does not depend on the model's training state.
    """
    n_maps = len(untrained_data) + 1
    _, n_metrics, n_subjs = untrained_data[0].get_top_iden('subj', 'inst', 'pair').mat.shape

    std_z = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    std_z.fill(np.nan)
    untr_z = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    untr_z.fill(np.nan)

    for type_idx, map_type in enumerate(['pair', 'gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                std_mat = standard_data[map_idx].get_top_iden('subj', 'subj', map_type).mat
                std_z[type_idx, map_idx] = np.nanmean(stat_func.r2z(std_mat, metric='pearson'), axis=0)

                untr_mat = untrained_data[map_idx].get_top_iden('subj', 'subj', map_type).mat
                untr_z[type_idx, map_idx] = np.nanmean(stat_func.r2z(untr_mat, metric='pearson'), axis=0)
            else:
                std_mat = standard_data[map_idx-1].get_top_iden('subj', 'inst', map_type).mat
                std_mat = np.nanmean(stat_func.r2z(std_mat, metric='pearson'), axis=0)
                try:
                    std_z[type_idx, map_idx] = std_mat
                except ValueError:
                    std_z[type_idx, map_idx, :2] = std_mat

                untr_mat = untrained_data[map_idx-1].get_top_iden('subj', 'inst', map_type).mat
                untr_mat = np.nanmean(stat_func.r2z(untr_mat, metric='pearson'), axis=0)
                try:
                    untr_z[type_idx, map_idx] = untr_mat
                except ValueError:
                    untr_z[type_idx, map_idx, :2] = untr_mat

    untr_z_raw = untr_z.copy()  # (2, n_maps, n_metrics, n_subjs), pre-diff, for the debug plot below

    std_z = np.squeeze(-np.diff(std_z, axis=0), axis=0)
    untr_z = np.squeeze(-np.diff(untr_z, axis=0), axis=0)
    std_r = stat_func.z2r(std_z, metric='pearson')
    untr_r = stat_func.z2r(untr_z, metric='pearson')

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = plt.cm.get_cmap('Set1', 8)

    plt.clf()
    plt.figure(figsize=(5, 4))

    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            vals = untr_r[map_idx, met_idx, :]
            if np.all(np.isnan(vals)):
                continue
            x_pos = met_idx * 4 + map_idx * 0.8
            box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_control_boxplot(box, colors(map_idx))
            for k in range(n_subjs):
                plt.scatter(x_pos - 0.3, vals[k], color=colors(map_idx), s=5)

            if map_idx == 0:
                continue  # Human-Human standard == untrained reference, redundant to show
            std_vals = std_r[map_idx, met_idx, :]
            if not np.all(np.isnan(std_vals)):
                plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=60,
                            color=colors(map_idx), edgecolor='black', linewidth=0.5, zorder=5)

    data_all = np.concatenate([untr_r[np.isfinite(untr_r)], std_r[np.isfinite(std_r)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    vs_zero_y = data_min - 0.15 * data_range
    max_bracket = data_max

    for met_idx in range(n_metrics):
        sub_data = untr_z[:, met_idx, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):
                if map_labels[i] != 'Human-Human':
                    continue
                if np.all(np.isnan(sub_data[j])):
                    continue

                # two-sided: Human-Human vs Human-Model
                t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                try:
                    bayes10 = float(pg.ttest(sub_data[i], sub_data[j], paired=False)['BF10'].values[0])
                except Exception:
                    bayes10 = np.nan
                bayes01 = 1 / bayes10
                print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]}, "
                      f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                      f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 8)}, "
                      f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}")

                x_mid = (met_idx * 4 + i * 0.8 + met_idx * 4 + j * 0.8) / 2
                y_max = np.nanmax(untr_r[:, met_idx, :]) + 0.08 * data_range * j
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                if p_val < 1e-3:
                    power = int(np.floor(np.log10(p_val)))
                    coefficient = p_val / (10 ** power)
                    anno = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
                else:
                    anno = r"$p = {:.3f}$".format(p_val)
                plt.plot([met_idx * 4 + i * 0.8, met_idx * 4 + j * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                plt.annotate(anno, (x_mid, y_max), textcoords="offset points", xytext=(0, 3), ha='center', size=9, alpha=alpha)

                # one-sided: Human-Model vs 0
                t_stat0, p_val0 = stats.ttest_1samp(sub_data[j], 0, alternative='greater', nan_policy='omit')
                x_pos0 = (met_idx * 4 + j * 0.8) + 0.15
                if p_val0 < 1e-3:
                    star = '***'
                elif p_val0 < 0.01:
                    star = '**'
                elif p_val0 < 0.05:
                    star = '*'
                else:
                    star = 'n.s.'
                alpha0 = 1 if p_val0 < 0.05 else 0.5
                plt.annotate(star, (x_pos0, vs_zero_y), ha='center', size=8, alpha=alpha0, fontweight='bold')

    plt.xticks([1.2, 5.2, 8.4], ['Accuracy', 'Confidence', 'RT'], fontsize=12)
    plt.xlim(-1, 10)
    plt.ylim(vs_zero_y, max_bracket)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel(r'$r_{best\ pair} - r_{other\ pairs}$ ', fontsize=12, fontweight='bold')
    plt.title('Best-pair advantage', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    path_name = path / f'top_btw_bs_untrained_{name}.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()

    # ---- debug: untrained network's same-subject vs other-subjects side by side (not differenced) ----
    plt.clf()
    plt.figure(figsize=(7, 4))
    _plot_same_vs_other_debug(untr_z_raw, n_maps, n_metrics, n_subjs, colors,
                               x_pos_fn=lambda met_idx, map_idx: met_idx * 4 + map_idx * 0.8)
    _add_same_other_legend()
    plt.xticks([1.2, 5.2, 8.4], ['Accuracy', 'Confidence', 'RT'], fontsize=12)
    plt.xlim(-1, 10)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel(r'$r$', fontsize=12, fontweight='bold')
    plt.title('Best-pair advantage - debug: same vs other', fontsize=13, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    path_name = path / f'top_btw_bs_untrained_{name}_debug.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()


def plot_corr_within_metric_consistency_untrained(standard_data, untrained_data, name, path, split_by):
    """ Untrained-network counterpart of plot_corr_within_metric_consistency.

    Boxes show the untrained network's per-subject consistency; a star
    overlays the standard (trained) network's reference value.
    """
    n_maps = len(untrained_data) + 1
    n_boots, n_metrics, n_subjs = untrained_data[0].get_corr_results('subj', 'inst', 'subj', 'split', split_by=split_by).mat.shape

    std_z = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    std_z.fill(np.nan)
    untr_z = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    untr_z.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                std_mat = standard_data[map_idx].get_corr_results('subj', 'subj', map_type, 'split', split_by=split_by).mat
                std_z[type_idx, map_idx] = np.nanmean(stat_func.r2z(std_mat, metric='pearson'), axis=0)

                untr_mat = untrained_data[map_idx].get_corr_results('subj', 'subj', map_type, 'split', split_by=split_by).mat
                untr_z[type_idx, map_idx] = np.nanmean(stat_func.r2z(untr_mat, metric='pearson'), axis=0)
            else:
                std_mat = standard_data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'split', split_by=split_by).mat
                std_mat = np.nanmean(stat_func.r2z(std_mat, metric='pearson'), axis=0)
                try:
                    std_z[type_idx, map_idx] = std_mat
                except ValueError:
                    std_z[type_idx, map_idx, :2] = std_mat

                untr_mat = untrained_data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'split', split_by=split_by).mat
                untr_mat = np.nanmean(stat_func.r2z(untr_mat, metric='pearson'), axis=0)
                try:
                    untr_z[type_idx, map_idx] = untr_mat
                except ValueError:
                    untr_z[type_idx, map_idx, :2] = untr_mat

    untr_z_raw = untr_z.copy()  # (2, n_maps, n_metrics, n_subjs), pre-diff, for the debug plot below

    std_z = np.squeeze(-np.diff(std_z, axis=0), axis=0)    # (n_maps, n_metrics, n_subjs)
    untr_z = np.squeeze(-np.diff(untr_z, axis=0), axis=0)  # (n_maps, n_metrics, n_subjs)
    std_r = stat_func.z2r(std_z, metric='pearson')
    untr_r = stat_func.z2r(untr_z, metric='pearson')

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS
    group_center = (n_maps - 1) * 0.8 / 2
    group_spacing = 2.2

    plt.figure(figsize=(4, 4))

    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            vals = untr_r[map_idx, met_idx, :]
            if np.all(np.isnan(vals)):
                continue
            x_pos = met_idx * group_spacing + map_idx * 0.8
            box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, colors(map_idx))
            for k in range(n_subjs):
                plt.scatter(x_pos - 0.3, vals[k], color=colors(map_idx), s=5)

            if map_idx == 0:
                continue  # Human-Human standard == untrained reference, redundant to show
            std_vals = std_r[map_idx, met_idx, :]
            if not np.all(np.isnan(std_vals)):
                plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=120,
                            color=colors(map_idx), edgecolor='black', linewidth=0.5, zorder=5)

    data_all = np.concatenate([untr_r[np.isfinite(untr_r)], std_r[np.isfinite(std_r)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    for met_idx in range(n_metrics):
        sub_data = untr_z[:, met_idx, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):
                if map_labels[i] != 'Human-Human':
                    continue
                if np.all(np.isnan(sub_data[j])):
                    continue

                # two-sided: Human-Human vs Human-Model
                t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]}, "
                      f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                      f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 8)}, "
                      f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}")

                y_max = np.nanmax(untr_r[:, met_idx, :]) + 0.08 * data_range * j
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(met_idx * group_spacing + i * 0.8, met_idx * group_spacing + j * 0.8, y_max,
                                   _format_pval(p_val), alpha=alpha, fontsize=9, xytext=(0, 3))

                # one-sided: Human-Model vs 0
                t_stat0, p_val0 = stats.ttest_1samp(sub_data[j], 0, alternative='greater', nan_policy='omit')
                x_pos0 = (met_idx * group_spacing + j * 0.8) + 0.15
                star, star_alpha = _stars_for_pval(p_val0)
                star_y = np.nanmin(untr_r[j, met_idx, :]) - 0.06 * data_range
                min_annot = min(min_annot, star_y)
                plt.annotate(star, (x_pos0, star_y), ha='center', size=8, alpha=star_alpha, fontweight='bold')

                # two-sided: standard vs untrained, per box
                std_su_vals = std_z[j, met_idx, :]
                if not np.all(np.isnan(std_su_vals)):
                    t_stat_su, p_val_su = stats.ttest_ind(sub_data[j], std_su_vals, equal_var=False, nan_policy='omit')
                    print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]}, "
                          f"Comparison: {map_labels[j]} standard vs untrained - "
                          f"t-stat: {t_stat_su:.4f}, p-value: {_format_pval_print(p_val_su, 8)}")
                    std_su_y = star_y - 0.08 * data_range
                    min_annot = min(min_annot, std_su_y)
                    plt.annotate(_format_pval(p_val_su), (met_idx * group_spacing + j * 0.8, std_su_y),
                                 ha='center', size=7)

    plt.xticks([met_idx * group_spacing + group_center for met_idx in range(n_metrics)], ['Accuracy', 'Confidence', 'RT'], fontsize=14)
    plt.xlim(-1, 6.5)
    plt.ylim(min_annot, max_bracket)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('r(same subject) − r(other subjects)', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_bs_untrained_{name}_split_{split_by}.pdf')

    # ---- debug: untrained network's same-subject vs other-subjects side by side (not differenced) ----
    plt.figure(figsize=(4, 4))
    _plot_same_vs_other_debug(untr_z_raw, n_maps, n_metrics, n_subjs, colors,
                               x_pos_fn=lambda met_idx, map_idx: met_idx * group_spacing + map_idx * 0.8)
    _add_same_other_legend()
    plt.xticks([met_idx * group_spacing + group_center for met_idx in range(n_metrics)], ['Accuracy', 'Confidence', 'RT'], fontsize=12)
    plt.xlim(-1, 6.5)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('r', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency - debug: same vs other', fontsize=13, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_bs_untrained_{name}_split_{split_by}_debug.pdf')


def plot_rank_within_metric_consistency_untrained(standard_data, untrained_data, name, path):
    """ Untrained-network counterpart of plot_rank_within_metric_consistency. """
    n_maps = len(untrained_data) + 1
    n_boots, n_metrics, n_subjs = untrained_data[0].get_corr_results('subj', 'inst', 'subj', 'split').mat.shape

    std_data = np.empty(shape=(n_maps, n_boots, n_metrics))
    std_data.fill(np.nan)
    untr_data = np.empty(shape=(n_maps, n_boots, n_metrics))
    untr_data.fill(np.nan)

    for map_idx in range(n_maps):
        if map_idx == 0:
            std_data[map_idx] = standard_data[map_idx].get_rank_results('subj', 'subj', 'split').mat
            untr_data[map_idx] = untrained_data[map_idx].get_rank_results('subj', 'subj', 'split').mat
        else:
            std_mat = standard_data[map_idx-1].get_rank_results('subj', 'inst', 'split').mat
            try:
                std_data[map_idx] = std_mat
            except ValueError:
                std_data[map_idx, :, :2] = std_mat

            untr_mat = untrained_data[map_idx-1].get_rank_results('subj', 'inst', 'split').mat
            try:
                untr_data[map_idx] = untr_mat
            except ValueError:
                untr_data[map_idx, :, :2] = untr_mat

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS
    group_center = (n_maps - 1) * 0.8 / 2
    group_spacing = 2.2

    plt.figure(figsize=(4, 4))

    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            if map_idx > 1 and met_idx > 1:
                continue
            vals = untr_data[map_idx, :, met_idx]
            if np.all(np.isnan(vals)):
                continue
            x_pos = met_idx * group_spacing + map_idx * 0.8
            box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, colors(map_idx))

            if map_idx == 0:
                continue  # Human-Human standard == untrained reference, redundant to show
            std_vals = std_data[map_idx, :, met_idx]
            if not np.all(np.isnan(std_vals)):
                plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=60,
                            color=colors(map_idx), edgecolor='black', linewidth=0.5, zorder=5)

    data_all = np.concatenate([untr_data[np.isfinite(untr_data)], std_data[np.isfinite(std_data)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    # two-sided: standard vs untrained, per box
    for map_idx in range(1, n_maps):
        for met_idx in range(n_metrics):
            if map_idx > 1 and met_idx > 1:
                continue
            untr_vals = untr_data[map_idx, :, met_idx]
            std_vals = std_data[map_idx, :, met_idx]
            if np.all(np.isnan(untr_vals)) or np.all(np.isnan(std_vals)):
                continue
            diff = untr_vals - std_vals
            n_diff = len(diff)
            p_val_su = 2 * min(
                (len(diff[diff >= 0]) + 1) / (n_diff + 1),
                (len(diff[diff < 0]) + 1) / (n_diff + 1)
            )
            print(f"Comparison: {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} standard vs untrained - "
                  f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]} - p-value: {_format_pval_print(p_val_su, 4)}")
            x_pos = met_idx * group_spacing + map_idx * 0.8
            y_pos = np.nanmin([np.nanmin(untr_vals), np.nanmin(std_vals)]) - 0.04 * data_range
            min_annot = min(min_annot, y_pos)
            plt.annotate(_format_pval(p_val_su), (x_pos, y_pos), ha='center', size=7)

    if name == 'mnist':
        for map_idx in range(1, n_maps):
            diff = untr_data[0] - untr_data[map_idx]
            for met_idx in range(n_metrics):
                for_proportion = diff[:, met_idx]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} - "
                      f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met_idx == 2 and not (map_idx == 1):
                    continue
                y_max = np.nanmax(untr_data[:, :, met_idx]) + 0.08 * data_range * map_idx
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(met_idx * group_spacing, met_idx * group_spacing + map_idx * 0.8, y_max,
                                   _format_pval_simple(p_val), alpha=alpha, fontsize=9, xytext=(0, 3))

    if name == 'ecoset10':
        for map_idx in range(1, n_maps):
            for met_idx in range(n_metrics):
                for_proportion = untr_data[map_idx, :, met_idx]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} - "
                      f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met_idx == 2 and not (map_idx == 1):
                    continue
                x_pos = (met_idx * group_spacing + map_idx * 0.8)
                y_max = np.nanpercentile(untr_data[map_idx, :, met_idx], 0) - 0.08 * data_range
                min_annot = min(min_annot, y_max)
                anno, alpha = _stars_for_pval(p_val)
                plt.annotate(anno, (x_pos, y_max), ha='center', size=9, alpha=alpha, fontweight='bold')

    plt.xticks([met_idx * group_spacing + group_center for met_idx in range(n_metrics)], ['Accuracy', 'Confidence', 'RT'], fontsize=12)
    plt.ylim(min_annot, max_bracket)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title('Rank consistency', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    _save_pdf(path / f'rank_btw_bs_untrained_{name}.pdf')


def plot_corr_across_metric_consistency_untrained(standard_data, untrained_data, name, path, split_by):
    """ Untrained-network counterpart of plot_corr_across_metric_consistency. """
    n_maps = len(untrained_data) + 1
    n_boots, n_pairs, n_subjs = untrained_data[0].get_corr_results('subj', 'inst', 'subj', 'var', split_by).mat.shape

    std_z = np.empty(shape=(2, n_maps, n_pairs, n_subjs))
    std_z.fill(np.nan)
    untr_z = np.empty(shape=(2, n_maps, n_pairs, n_subjs))
    untr_z.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                std_mat = standard_data[map_idx].get_corr_results('subj', 'subj', map_type, 'var', split_by).mat
                std_z[type_idx, map_idx] = np.mean(stat_func.r2z(std_mat, metric='pearson'), axis=0)

                untr_mat = untrained_data[map_idx].get_corr_results('subj', 'subj', map_type, 'var', split_by).mat
                untr_z[type_idx, map_idx] = np.mean(stat_func.r2z(untr_mat, metric='pearson'), axis=0)
            else:
                std_mat = standard_data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'var', split_by).mat
                std_mat = np.mean(stat_func.r2z(std_mat, metric='pearson'), axis=0)
                if map_idx > 1 and n_pairs > 1:
                    std_z[type_idx, map_idx, 0] = std_mat
                else:
                    std_z[type_idx, map_idx] = std_mat

                untr_mat = untrained_data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'var', split_by).mat
                untr_mat = np.mean(stat_func.r2z(untr_mat, metric='pearson'), axis=0)
                if map_idx > 1 and n_pairs > 1:
                    untr_z[type_idx, map_idx, 0] = untr_mat
                else:
                    untr_z[type_idx, map_idx] = untr_mat

    untr_z_raw = untr_z.copy()  # (2, n_maps, n_pairs, n_subjs), pre-diff, for the debug plot below

    std_z = np.squeeze(-np.diff(std_z, axis=0), axis=0)
    untr_z = np.squeeze(-np.diff(untr_z, axis=0), axis=0)
    std_r = stat_func.z2r(std_z, metric='pearson')
    untr_r = stat_func.z2r(untr_z, metric='pearson')

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS
    group_center = (n_maps - 1) * 0.8 / 2
    group_spacing = 3.0

    plt.figure(figsize=(4, 4))

    for map_idx in range(n_maps):
        for pair_idx in range(n_pairs):
            vals = untr_r[map_idx, pair_idx, :]
            if np.all(np.isnan(vals)):
                continue
            x_pos = pair_idx * group_spacing + map_idx * 0.8
            box = plt.boxplot(vals, positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, colors(map_idx))
            for k in range(n_subjs):
                plt.scatter(x_pos - 0.3, vals[k], color=colors(map_idx), s=5)

            if map_idx == 0:
                continue  # Human-Human standard == untrained reference, redundant to show
            std_vals = std_r[map_idx, pair_idx, :]
            if not np.all(np.isnan(std_vals)):
                plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=120,
                            color=colors(map_idx), edgecolor='black', linewidth=0.5, zorder=5)

    data_all = np.concatenate([untr_r[np.isfinite(untr_r)], std_r[np.isfinite(std_r)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    for pair_idx in range(n_pairs):
        sub_data = untr_z[:, pair_idx, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):
                if map_labels[i] != 'Human-Human':
                    continue
                if np.all(np.isnan(sub_data[j])):
                    continue

                # two-sided: Human-Human vs Human-Model
                t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                bayes10, bayes01 = _bayes_factors(sub_data[i], sub_data[j], paired=False)
                print(f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]}, "
                      f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                      f"t-stat: {t_stat:.4f}, p-value: {_format_pval_print(p_val, 6)}, "
                      f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}")

                plot_x_pos = [pair_idx * group_spacing, pair_idx * group_spacing + j * 0.8]

                y_max = np.nanmax(untr_r[:, pair_idx, :]) + 0.08 * data_range * j
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(plot_x_pos[0], plot_x_pos[1], y_max, _format_pval(p_val),
                                   alpha=alpha, fontsize=9, xytext=(0, 3))

                # one-sided: Human-Model vs 0
                t_stat0, p_val0 = stats.ttest_1samp(sub_data[j], 0, alternative='greater', nan_policy='omit')
                x_pos0 = pair_idx * group_spacing + j * 0.8 + 0.15
                star, star_alpha = _stars_for_pval(p_val0)
                star_y = np.nanmin(untr_r[j, pair_idx, :]) - 0.06 * data_range
                min_annot = min(min_annot, star_y)
                plt.annotate(star, (x_pos0, star_y), ha='center', size=8, alpha=star_alpha, fontweight='bold')

                # two-sided: standard vs untrained, per box
                std_su_vals = std_z[j, pair_idx, :]
                if not np.all(np.isnan(std_su_vals)):
                    t_stat_su, p_val_su = stats.ttest_ind(sub_data[j], std_su_vals, equal_var=False, nan_policy='omit')
                    print(f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]}, "
                          f"Comparison: {map_labels[j]} standard vs untrained - "
                          f"t-stat: {t_stat_su:.4f}, p-value: {_format_pval_print(p_val_su, 6)}")
                    std_su_y = star_y - 0.08 * data_range
                    min_annot = min(min_annot, std_su_y)
                    plt.annotate(_format_pval(p_val_su), (plot_x_pos[1], std_su_y),
                                 ha='center', size=7)

    plt.xticks([pair_idx * group_spacing + group_center for pair_idx in range(n_pairs)], ['Acc-Conf', 'Acc-RT', 'Conf-RT'], fontsize=14)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlim(-1, 8)
    plt.ylim(min_annot, max_bracket)
    plt.xlabel('Pairs of behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('r(same subject) − r(other subjects)', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_var_untrained_{name}_split_{split_by}.pdf')

    # ---- debug: untrained network's same-subject vs other-subjects side by side (not differenced) ----
    plt.figure(figsize=(4, 4))
    _plot_same_vs_other_debug(
        untr_z_raw, n_maps, n_pairs, n_subjs, colors,
        x_pos_fn=lambda pair_idx, map_idx: pair_idx * group_spacing + map_idx * 0.8
    )
    _add_same_other_legend()
    plt.xticks([pair_idx * group_spacing + group_center for pair_idx in range(n_pairs)], ['Acc-Conf', 'Acc-RT', 'Conf-RT'], fontsize=12)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlim(-1, 8)
    plt.xlabel('Pairs of behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel('r', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency - debug: same vs other', fontsize=12, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_var_untrained_{name}_split_{split_by}_debug.pdf')


def plot_rank_across_metric_consistency_untrained(standard_data, untrained_data, name, path):
    """ Untrained-network counterpart of plot_rank_across_metric_consistency. """
    n_maps = len(untrained_data) + 1
    n_boots, n_pairs, n_subjs = untrained_data[0].get_corr_results('subj', 'inst', 'subj', 'var').mat.shape

    std_data = np.empty(shape=(n_maps, n_boots, n_pairs))
    std_data.fill(np.nan)
    untr_data = np.empty(shape=(n_maps, n_boots, n_pairs))
    untr_data.fill(np.nan)

    for map_idx in range(n_maps):
        if map_idx == 0:
            std_data[map_idx] = standard_data[map_idx].get_rank_results('subj', 'subj', 'var').mat
            untr_data[map_idx] = untrained_data[map_idx].get_rank_results('subj', 'subj', 'var').mat
        else:
            std_data[map_idx] = standard_data[map_idx-1].get_rank_results('subj', 'inst', 'var').mat
            untr_data[map_idx] = untrained_data[map_idx-1].get_rank_results('subj', 'inst', 'var').mat

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = _COLORS
    group_center = (n_maps - 1) * 0.8 / 2
    group_spacing = 3.0

    plt.figure(figsize=(4, 4))

    for map_idx in range(n_maps):
        for pair_idx in range(n_pairs):
            if pair_idx != 0 and not (map_idx <= 1):
                continue
            x_pos = pair_idx * group_spacing + map_idx * 0.8
            box = plt.boxplot(untr_data[map_idx, :, pair_idx], positions=[x_pos], widths=0.4, patch_artist=True, showfliers=False)
            _style_boxplot(box, colors(map_idx))

            if map_idx == 0:
                continue  # Human-Human standard == untrained reference, redundant to show
            std_vals = std_data[map_idx, :, pair_idx]
            plt.scatter(x_pos + 0.3, np.nanmean(std_vals), marker='*', s=60,
                        color=colors(map_idx), edgecolor='black', linewidth=0.5, zorder=5)

    data_all = np.concatenate([untr_data[np.isfinite(untr_data)], std_data[np.isfinite(std_data)]])
    data_min, data_max = np.nanmin(data_all), np.nanmax(data_all)
    data_range = data_max - data_min
    max_bracket = data_max
    min_annot = data_min

    # two-sided: standard vs untrained, per box
    for map_idx in range(1, n_maps):
        for pair_idx in range(n_pairs):
            if pair_idx != 0 and not (map_idx <= 1):
                continue
            untr_vals = untr_data[map_idx, :, pair_idx]
            std_vals = std_data[map_idx, :, pair_idx]
            if np.all(np.isnan(untr_vals)) or np.all(np.isnan(std_vals)):
                continue
            diff = untr_vals - std_vals
            n_diff = len(diff)
            p_val_su = 2 * min(
                (len(diff[diff >= 0]) + 1) / (n_diff + 1),
                (len(diff[diff < 0]) + 1) / (n_diff + 1)
            )
            print(f"Comparison: {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} standard vs untrained - "
                  f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]} - "
                  f"p-value: {_format_pval_print(p_val_su, 4)}")
            x_pos = pair_idx * group_spacing + map_idx * 0.8
            y_pos = np.nanmin([np.nanmin(untr_vals), np.nanmin(std_vals)]) - 0.04 * data_range
            min_annot = min(min_annot, y_pos)
            plt.annotate(_format_pval(p_val_su), (x_pos, y_pos), ha='center', size=7)

    if name == 'mnist':
        for map_idx in range(1, n_maps):
            diff = untr_data[0] - untr_data[map_idx]
            for pair_idx in range(n_pairs):
                if pair_idx != 0 and not (map_idx == 1):
                    continue
                for_proportion = diff[:, pair_idx]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} - "
                      f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                plot_x_pos = [pair_idx * group_spacing, pair_idx * group_spacing + map_idx * 0.8]
                y_max = np.nanmax(untr_data[:, :, pair_idx]) + 0.08 * data_range * map_idx
                max_bracket = max(max_bracket, y_max)
                alpha = 0.5 if p_val < 0.05 else 1
                _annotate_bracket(plot_x_pos[0], plot_x_pos[1], y_max, _format_pval_simple(p_val),
                                   alpha=alpha, fontsize=9, xytext=(0, 3))

    if name == 'ecoset10':
        for map_idx in range(1, n_maps):
            for pair_idx in range(n_pairs):
                if pair_idx != 0 and not (map_idx == 1):
                    continue
                for_proportion = untr_data[map_idx, :, pair_idx]
                p_val = 2 * min(
                    len(for_proportion[for_proportion >= 0]) / len(for_proportion),
                    len(for_proportion[for_proportion < 0]) / len(for_proportion)
                )
                ci_lower = np.percentile(for_proportion, 2.5)
                ci_upper = np.percentile(for_proportion, 97.5)
                print(f"Comparison: Subject vs {['RTNet', 'AlexNet', 'ResNet18'][map_idx-1]} - "
                      f"Pair: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][pair_idx]} - "
                      f"p-value: {_format_pval_print(p_val, 4)}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                x_pos = pair_idx * group_spacing + map_idx * 0.8
                y_max = min_annot - 0.08 * data_range
                min_annot = min(min_annot, y_max)
                anno, alpha = _stars_for_pval(p_val)
                plt.annotate(anno, (x_pos, y_max), ha='center', size=9, alpha=alpha, fontweight='bold')

    plt.xticks([pair_idx * group_spacing + group_center for pair_idx in range(n_pairs)], ['Acc-Conf', 'Acc-RT', 'Conf-RT'], fontsize=12)
    plt.xlim(-1, 8)
    plt.ylim(min_annot, max_bracket)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Pairs of behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title('Rank consistency', fontsize=14, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    _add_control_legend(colors, n_maps)
    plt.tight_layout()
    _save_pdf(path / f'rank_btw_var_untrained_{name}.pdf')


def plot_corr_within_metric_consistency_same_vs_other(data, name, path, split_by):
    """ Debug variant of plot_corr_within_metric_consistency.

    Instead of plotting the same-subject-minus-other-subjects difference,
    plots the raw same-subject and other-subjects correlation values side by
    side (in r-space), so the two components of the difference can be
    inspected independently. Mirrors the same/other debug plot used for the
    untrained-network analysis, applied here to the standard network.
    """
    n_maps = len(data) + 1
    n_boots, n_metrics, n_subjs = data[0].get_corr_results('subj', 'inst', 'subj', 'split', split_by=split_by).mat.shape

    plot_z = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    plot_z.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'split', split_by=split_by).mat
                map_data = np.nanmean(stat_func.r2z(map_data, metric='pearson'), axis=0)
                plot_z[type_idx, map_idx] = map_data
            else:
                map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'split', split_by=split_by).mat
                map_data = np.nanmean(stat_func.r2z(map_data, metric='pearson'), axis=0)
                try:
                    plot_z[type_idx, map_idx] = map_data
                except ValueError:
                    plot_z[type_idx, map_idx, :2] = map_data

    colors = _COLORS

    plt.figure(figsize=(7, 4))
    _plot_same_vs_other_debug(plot_z, n_maps, n_metrics, n_subjs, colors,
                               x_pos_fn=lambda met_idx, map_idx: met_idx * 4 + map_idx * 0.8,
                               dot_offset=0.15)
    _add_same_other_legend()
    plt.xticks([1.2, 5.2, 8.4], ['Accuracy', 'Confidence', 'RT'], fontsize=12)
    plt.xlim(-1, 10)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel(r'$r$', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency - debug: same vs other', fontsize=13, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_bs_{name}_split_{split_by}_same_vs_other.pdf')


def plot_corr_across_metric_consistency_same_vs_other(data, name, path, split_by):
    """ Debug variant of plot_corr_across_metric_consistency.

    Instead of plotting the same-subject-minus-other-subjects difference,
    plots the raw same-subject and other-subjects correlation values side by
    side (in r-space), so the two components of the difference can be
    inspected independently.
    """
    n_maps = len(data) + 1
    n_boots, n_pairs, n_subjs = data[0].get_corr_results('subj', 'inst', 'subj', 'var', split_by).mat.shape

    plot_z = np.empty(shape=(2, n_maps, n_pairs, n_subjs))
    plot_z.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'var', split_by).mat
                map_data = np.mean(stat_func.r2z(map_data, metric='pearson'), axis=0)
                plot_z[type_idx, map_idx] = map_data
            else:
                map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'var', split_by).mat
                map_data = np.mean(stat_func.r2z(map_data, metric='pearson'), axis=0)
                if map_idx > 1 and n_pairs > 1:
                    plot_z[type_idx, map_idx, 0] = map_data
                else:
                    plot_z[type_idx, map_idx] = map_data

    colors = _COLORS

    plt.figure(figsize=(6, 4))
    _plot_same_vs_other_debug(
        plot_z, n_maps, n_pairs, n_subjs, colors,
        x_pos_fn=lambda pair_idx, map_idx: pair_idx * 4 + map_idx * 0.8 if pair_idx < 2 else pair_idx * 3.2 + map_idx * 0.8,
        dot_offset=0.15
    )
    _add_same_other_legend()
    plt.xticks([1.2, 4.4, 6.8], ['Acc-Conf', 'Acc-RT', 'Conf-RT'], fontsize=12)
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1.5, alpha=0.75)
    plt.xlim(-1, 8.5)
    plt.xlabel('Pairs of behavioral metrics', fontsize=12, fontweight='bold')
    plt.ylabel(r'$r$', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency - debug: same vs other', fontsize=12, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    _save_pdf(path / f'corr_btw_var_{name}_split_{split_by}_same_vs_other.pdf')