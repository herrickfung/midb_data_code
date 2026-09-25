from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from itertools import combinations
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


def _add_control_legend(colors, n_maps):
    legend_elements = [
        Line2D([0], [0], marker='*', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label='Standard')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=7, frameon=False)


def plot_corr_within_metric_consistency_control(standard_data, merged_control, name, path, split_by):
    """ Control-instance counterpart of plot_corr_within_metric_consistency.

    standard_data: [rtnet, alexnet, resnet18] IndiMap objects (variant='standard').
    merged_control: [rtnet, alexnet, resnet18] merged-result dicts produced by
    analyze.py's merge_control (already r2z-averaged over the 60 control
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


