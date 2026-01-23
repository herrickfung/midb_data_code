from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import combinations
from scipy.stats import sem
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pingouin as pg

from indimap.util import map_func, stat_func

def plot_raincloud(data, name, path):
    n_maps = len(data) + 1
    n_conds, n_metrics, n_subjs, n_imgs = data[0].dims_map.human_arr.shape
    data_arr = np.empty(shape=(n_maps, n_metrics, n_subjs, n_imgs))
    data_arr.fill(np.nan)

    for map_idx in range(n_maps):
        # avg across conditions
        if map_idx == 0:
            data_arr[0] = data[map_idx].dims_map.human_arr.mean(axis=0)
        else:
            try:
                data_arr[map_idx] = data[map_idx-1].dims_map.model_arr.mean(axis=0)
            except ValueError:
                data_arr[map_idx, :2] = data[map_idx-1].dims_map.model_arr.mean(axis=0)

    data_arr = np.nanmean(data_arr, axis = 3) # average across trials
    # normalized confidence and rt by average before plotting
    for map_idx in range(n_maps):
        for met_idx in range(n_metrics):
            if met_idx > 0:
                vec = data_arr[map_idx, met_idx]
                vec = vec / np.mean(vec)
                data_arr[map_idx, met_idx] = vec

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']

    plt.clf()
    figure, ax = plt.subplots(1, n_metrics, figsize=(8, 3))
    met_titles = ['Accuracy', 'Confidence', 'RT']
    met_labels = ['Accuracy', 'Normalized confidence', 'Normalized RT']
    colors = plt.cm.get_cmap('Set1', 8)

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
    plt.savefig(f"{path}/raincloud_{name}.png", dpi=384, transparent=True)
    plt.close()


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

    for metric in range(n_metrics):
        plt.clf()
        fig, ax = plt.subplots(2, n_maps, figsize=(20, 7),
                                gridspec_kw={'height_ratios': [4, 1]}
                            )
        plt.subplots_adjust(hspace=0.001, wspace=0.01)
        counts = np.zeros((n_maps, n_subjs))

        for map_idx in range(n_maps):
            im = ax[0, map_idx].imshow(
                results[map_idx, metric],
                cmap='Purples',
                vmin=0, vmax=1,
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

            ax[1, map_idx].bar(np.arange(n_subjs), counts[map_idx], color='Purple', alpha=1)
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

        plt.tight_layout()
        path_name = path / f'raw_sim_map_{metrics[metric]}_{name}.png'
        plt.savefig(path_name, dpi=384, transparent=True)
        plt.close()
    return results


def plot_alignment_average(data, name, path):
    plot_avg_data = data.copy()
    plt.figure(figsize=(5, 4))
    colors = plt.cm.get_cmap('Set1', 8)
    n_maps, n_metrics, n_subjs, _ = plot_avg_data.shape

    map_labels = ['Human-Human',
                'Human-RTNet',
                'Human-AlexNet',
                'Human-ResNet18',
                ]
    plt.xticks([1.2, 6.2, 10.4], 
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
                )
    plt.ylim(-0.05, 0.9)
    plt.xlim(-1.5, 12.5)
    # y_poss = [0.15, 0.15, 0.15]
    y_poss = [0.18, 0.20, 0.15]

    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 5 + map * 0.8
            if map == 0:
                plot_avg_data[map, met][plot_avg_data[map, met] == 1] = np.nan
            if map > 1 and met == 2:
                continue

            avg_data = np.nanmean(plot_avg_data[map, met, :, :], axis=1)
            print(map, met)
            print(np.nanmean(avg_data))
            plt.bar(x_pos, np.nanmean(avg_data),
                    yerr=sem(avg_data), alpha=0.5,
                    color=colors(map), label=map_labels[map] if met == 0 else None
                    )
            for subj in range(n_subjs):
                plt.scatter(x_pos-0.25, avg_data[subj], color=colors(map), alpha=0.75, s = 10)

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
                x_mid = (met * 5 + met * 5 + j * 0.8) / 2  # Midpoint between bars
                y_max = np.nanmean(plot_avg_data[:, met]) + y_poss[met] + 0.06 * j
                if p_val < 1e-3:
                    power = int(np.floor(np.log10(p_val)))
                    coefficient = p_val / (10 ** power)
                    anno = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
                else:
                    anno = r"$p = {:.3f}$".format(p_val)
                alpha= 1
                plt.plot([met * 5, met * 5 + j * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                plt.annotate(anno, (x_mid, y_max+0.01), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha)

    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Correlation coefficient', fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc='upper right', fontsize=10, frameon=False)
    plt.tight_layout()
    path_name = path / f'align_avg_{name}.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()


def plot_alignment_variance(data, name, path):
    n_maps = len(data) + 1
    n_boots, n_splits, n_metrics, n_subjs, _ = data[0].get_corr_map('subj', 'subj').mat.shape
    results = np.empty(shape=(n_maps, n_boots, n_splits, n_metrics))
    results.fill(np.nan)

    for i in range(n_maps):
        if i == 0:
            result = data[i].get_corr_map('subj', 'subj').mat
        else:
            result = data[i-1].get_corr_map('subj', 'inst').mat
        result = stat_func.r2z(result, metric='pearson')
        result = np.std(result, axis = (3, 4))
        try:
            results[i] = result
        except ValueError:
            results[i, :, :, :2] = result
    
    results = results.reshape(n_maps, n_boots * n_splits, n_metrics)

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    colors = plt.cm.get_cmap('Set1', 8)

    map_labels = ['Human-Human',
                'Human-RTNet',
                'Human-AlexNet',
                'Human-ResNet18'
                ]
    plt.xticks([1.2, 6.2, 10.4], 
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
                )
    plt.ylim(0, 0.25)

    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 5 + map * 0.8
            avg = np.mean(results[map, :,  met], axis=0)
            ax.bar(x_pos,
                    np.mean(avg),
                    yerr=sem(avg),
                    color=colors(map), label=map_labels[map] if met == 0 else None,
                    alpha = 0.5,
                    )

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
                print(f'{map_labels[i]} vs {map_labels[j]} - p-value: {p_val:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
                x_mid = (met * 5 + met * 5 + j * 0.8) / 2  # Midpoint between bars

                if name == 'mnist' or name == 'ecoset10':
                    y_max = max(np.nanmean(results[:, :, met]), np.nanmean(results[:, :, met])) + 0.05 + 0.015 * j
                    # y_max = max(np.nanmean(results[:, :, met]), np.nanmean(results[:, :, met])) + 0.03 + 0.015 * j
                else:
                    y_max = max(np.nanmean(results[:, :, met]), np.nanmean(results[:, :, met])) + 0.075 + 0.03 * j

                if p_val < 0.001:
                    anno = f'$p$ < 5 x 10$^{{-4}}$'
                else:
                    anno = f'$p$ = {p_val:.3f}'
                alpha = 1
                plt.plot([met * 5, met * 5 + j * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                plt.annotate(anno, (x_mid, y_max+0.0005), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha)

    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Standard deviation', fontsize=14)
    plt.ylim(0, 0.275)
    # plt.title('Variance', fontsize=16, fontweight='bold')  
    plt.legend(loc='upper left', fontsize=10, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    path_name = path / f'align_var_{name}.png'
    plt.savefig(path_name, dpi=384, transparent=True)


def plot_across_metric_illustration(data, name, path, indicate_best_match=False):
    n_maps, n_metrics, n_subj, _ = data.shape
    corr_consistency = np.full((n_maps, n_metrics, n_subj), np.nan)
    stats_results = np.full((n_maps, n_metrics), np.nan)
    stat_data = stat_func.r2z(data, metric='pearson')
    colors = plt.cm.get_cmap('Set1', 8)

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
                p_value = r"{:.2f} \times 10^{{{}}}".format(coefficient, power)
            else:
                p_value = r"{:.3f}".format(p_value)
            ax[j, i].set_ylim(np.min(met2_data[mask]-0.05), np.max(met2_data[mask]+0.18))
            # Annotate correlation and p-value in upper left corner
            ax[j, i].annotate(rf'$r = {r_value:.2f};  p = {p_value}$', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=7, ha='left', va='top',
            color=colors(i))
    plt.tight_layout()
    fig.savefig(path / f'corr_btw_var_illustration_{name}.png', dpi=384, transparent=True)
    plt.close()

    plt.figure(figsize=(5, 3.333))
    if name == 'mnist' or name == 'ecoset10':
        plt.xticks([1.2, 5.4, 10.4], 
                    ['Acc-Conf', 'Acc-RT', 'Conf-RT'],
                    fontsize=12
                    )
    else:
        plt.xticks([0.8], 
                    ['Accuracy\nConfidence'],
                    fontsize=12
                    )

    for i in range(n_maps):
        for j, (met1, met2) in enumerate(combinations(range(n_metrics), 2)):
            if i > 1 and j > 0:
                continue
            x_pos = j * 5 + i * 0.8
            plt.bar(x_pos, np.nanmean(corr_consistency[i, j]),
                    yerr=sem(corr_consistency[i, j]), alpha=0.5,
                    color=colors(i), label=model_labels[i] if j == 0 else None
            )
            for subj in range(n_subj):
                if subj == best_match and indicate_best_match:
                    color='black'
                    plt.scatter(x_pos-0.25, corr_consistency[i, j, subj], color=color, alpha=1, s = 50, zorder=2, marker='*',
                                label='Illustrated subject' if i == 0 and j == 0 else None)
                else:
                    color = colors(i)
                    plt.scatter(x_pos-0.25, corr_consistency[i, j, subj], color=color, alpha=0.75, s = 10,
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
                bf10 = float(pg.ttest(consistency_stat_data[i, k], consistency_stat_data[j, k], paired=True, alternative='two-sided')['BF10'].values[0])
                bf01 = 1 / bf10
                print(f"Model: {model_labels[i]}, Comparison: {model_labels[j]}, Metric: {k}, BF10: {bf10}, BF01: {bf01}")
                print("T-test results:", stat_results)
                p_val = stat_results.pvalue
                if p_val < 0.001:
                    power = int(np.floor(np.log10(p_val)))
                    coefficient = p_val / (10 ** power)
                    p_val = r"{:.2f} \times 10^{{{}}}".format(coefficient, power)
                else:
                    p_val = r"{:.3f}".format(p_val)
                x_mid = (k * 5 + k * 5 + j * 0.8) / 2  # Midpoint between bars
                y_max = np.nanmax(corr_consistency[:, k]) + 0.15 * j
                plt.plot([k * 5, k * 5 + j * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=1)
                plt.annotate(rf"$p = {p_val}$", (x_mid, y_max+0.02), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=1)

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
                x_pos = k * 5 + j * 0.8 + 0.15
                y_max = -0.1
                if p_val < 1e-3:
                    anno = '***'
                    alpha = 1
                elif p_val < 0.01:
                    anno = '**'
                    alpha = 1
                elif p_val < 0.05:
                    anno = '*'
                    alpha = 1
                else:
                    anno = 'n.s.'
                    alpha = 0.5
                plt.annotate(anno, (x_pos, y_max), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha, fontweight='bold')


    plt.ylim(-0.4, 2.2)
    plt.xlabel('Across-metric alignment', fontsize=14, fontweight='bold')
    plt.ylabel('Correlation coefficient', fontsize=14)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc='upper right', fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(path / f'corr_btw_var_correlation_{name}.png', dpi=384, transparent=True)
    plt.close()


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

    plt.clf()
    fig, ax = plt.subplots(1, n_metrics, figsize=(9, 2.5))
    colors = plt.cm.get_cmap('Set1', 8)
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
    path_name = path / f'top_count_dist_{name}.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()


def plot_within_metric_consistency(data, name, path):
    # plot corr consistency
    n_maps = len(data) + 1
    n_boots, n_metrics, n_subjs = data[0].get_corr_results('subj', 'inst', 'subj', 'split').mat.shape

    plot_data = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            for met_idx in range(n_metrics):
                if map_idx == 0:
                    map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'split').mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.mean(map_data, axis=0)
                    plot_data[type_idx, map_idx] = map_data
                else:
                    map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'split').mat
                    map_data = stat_func.r2z(map_data, metric='pearson')
                    map_data = np.mean(map_data, axis=0)
                    try:
                        plot_data[type_idx, map_idx] = map_data
                    except ValueError:
                        plot_data[type_idx, map_idx, :2] = map_data

    plot_data = -np.diff(plot_data, axis = 0)
    plot_data = np.squeeze(plot_data, axis = 0)

    plt.clf()
    plt.figure(figsize=(5, 4))

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = plt.cm.get_cmap('Set1', 8)
    plt.xticks([1.2, 6.2, 10.4], 
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
            )

    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson') 
    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 5 + map * 0.8
            plt.bar(x_pos,
                np.nanmean(plot_data_in_r[map,met, :]),
                yerr=sem(plot_data_in_r[map,met,:]),
                color=colors(map), label=map_labels[map] if met == 0 else None,
                alpha = 0.5
                )
            for k in range(n_subjs):
                plt.scatter(x_pos - 0.25,
                            plot_data_in_r[map, met, k],
                            color=colors(map), s=5
                            )

    for met in range(n_metrics):
        sub_data = plot_data[:, met, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):

                if name == 'mnist':
                    t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                    try:
                        bayes10 = float(pg.ttest(sub_data[i], sub_data[j], paired=False)['BF10'].values[0])
                    except Exception:
                        bayes10 = np.nan
                    bayes01 = 1 / bayes10
                    mean_diff = np.nanmean(sub_data[i]) - np.nanmean(sub_data[j])
                    pooled_std = np.sqrt((np.nanvar(sub_data[i], ddof=1) + np.nanvar(sub_data[j], ddof=1)) / 2)
                    cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
                    print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {p_val:.8f}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # Check for significance
                    if ((map_labels[i] == 'Human-Human')):
                        if met == 2 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        x_mid = (met * 5 + i * 0.8 + met * 5 + j * 0.8) / 2  # Midpoint between bars
                        y_max = max(np.nanmax(plot_data_in_r[:, met, :]), np.nanmax(plot_data_in_r[:, met, :])) + 0.1 * abs(j - i)
                        if p_val < 0.05:
                            alpha = 0.5
                        else:
                            alpha = 1
                        if p_val < 1e-3:
                            power = int(np.floor(np.log10(p_val)))
                            coefficient = p_val / (10 ** power)
                            anno = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
                        else:
                            anno = r"$p = {:.3f}$".format(p_val)
                        plt.plot([met * 5 + i * 0.8, met * 5 + j * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                        plt.annotate(anno, (x_mid, y_max+0.02), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha)
                        plt.ylim(-0.4, 1.3)
                        plt.legend(loc='upper right', fontsize=8, frameon=False)

                if name == 'ecoset10':
                    t_stat, p_val = stats.ttest_1samp(sub_data[j], 0, nan_policy='omit')
                    try:
                        bayes10 = float(pg.ttest(sub_data[i], sub_data[j], paired=False)['BF10'].values[0])
                    except Exception:
                        bayes10 = np.nan
                    bayes01 = 1 / bayes10
                    cohen_d = t_stat / np.sqrt(n_subjs) if n_subjs != 0 else np.nan
                    print(f"Metric: {['Accuracy', 'Confidence', 'Reaction time'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {p_val:.8f}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # compare to 0 signfiicance
                    if ((map_labels[i] == 'Human-Human')):
                        if met == 2 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        x_pos = (met * 5 + j * 0.8) + 0.15
                        y_max = -0.04
                        if p_val < 1e-3:
                            anno = '***'
                            alpha = 1
                        elif p_val < 0.01:
                            anno = '**'
                            alpha = 1
                        elif p_val < 0.05:
                            anno = '*'
                            alpha = 1
                        else:
                            anno = 'n.s.'
                            alpha = 0.5
                        plt.annotate(anno, (x_pos, y_max), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha, fontweight='bold')
                        plt.legend(loc='upper left', fontsize=8, frameon=False)


    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel(r'$r_{same\ subject} - r_{other\ subjects}$ ', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency', fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    path_name = path / f'corr_btw_bs_{name}.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()

    # plot rank consistency
    plot_data = np.empty(shape=(n_maps, n_boots, n_metrics))
    plot_data.fill(np.nan)
    for map_idx in range(n_maps):
        if map_idx == 0:
            map_data = data[map_idx].get_rank_results('subj', 'subj', 'split').mat
            plot_data[map_idx] = map_data
        else:
            map_data = data[map_idx-1].get_rank_results('subj', 'inst', 'split').mat
            try:
                plot_data[map_idx] = map_data
            except ValueError:
                plot_data[map_idx, :, :2] = map_data

    plt.clf()
    plt.figure(figsize=(5, 4))

    plt.xticks([1.2, 6.2, 10.4], 
                ['Accuracy', 'Confidence', 'RT'],
                fontsize=12
            )
    p_bar_pos = 15
    p_val_pos = 2

    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 5 + map * 0.8
            if map > 1 and met > 1:
                continue
            plt.bar(x_pos,
                np.nanmean(plot_data[map,:,met]),
                yerr=sem(plot_data[map,:, met]),
                color=colors(map), label=map_labels[map] if met == 0 else None,
                alpha = 0.5
                )

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
                    f"p-value: {p_val:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met == 2 and not (map == 1):
                    continue
                x_mid = (met * 5 + met * 5 + map * 0.8) / 2  # Midpoint between bars
                # y_max = max(np.nanmax(plot_data[:, :, met]), np.nanmax(plot_data[:, :, met])) - 30 + 20 * map
                y_max = np.nanmax(plot_data[:, :, met]) + p_bar_pos * abs(map - 0)
                if p_val < 0.05:
                    alpha = 0.5
                else:
                    alpha = 1
                if p_val < 0.001:
                    anno = f'$p$ < 0.001'
                else:
                    anno = f'$p$ = {p_val:.3f}'
                plt.plot([met * 5, met * 5 + map * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                plt.annotate(anno, (x_mid, y_max+p_val_pos), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha)

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
                    f"p-value: {p_val:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met == 2 and not (_map == 1):
                    continue
                x_pos = (met * 5 + _map * 0.8)
                y_max = -0.02
                if p_val < 1e-3:
                    anno = '***'
                    alpha = 1
                elif p_val < 0.01:
                    anno = '**'
                    alpha = 1
                elif p_val < 0.05:
                    anno = '*'
                    alpha = 1
                else:
                    anno = 'n.s.'
                    alpha = 0.5
                plt.annotate(anno, (x_pos, y_max), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha, fontweight='bold')  

    plt.xlabel('Behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title('Rank consistency', fontsize=16, fontweight='bold')  
    plt.legend(loc='best', fontsize=8, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    path_name = path / f'rank_btw_bs_{name}.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()


def plot_across_metric_consistency(data, name, path):
        # plot corr consistency
    n_maps = len(data) + 1
    n_boots, n_metrics, n_subjs = data[0].get_corr_results('subj', 'inst', 'subj', 'var').mat.shape

    plot_data = np.empty(shape=(2, n_maps, n_metrics, n_subjs))
    plot_data.fill(np.nan)

    for type_idx, map_type in enumerate(['subj', 'subj_gp']):
        for map_idx in range(n_maps):
            if map_idx == 0:
                map_data = data[map_idx].get_corr_results('subj', 'subj', map_type, 'var').mat
                map_data = stat_func.r2z(map_data, metric='pearson')
                map_data = np.mean(map_data, axis=0)
                plot_data[type_idx, map_idx] = map_data
            else:
                map_data = data[map_idx-1].get_corr_results('subj', 'inst', map_type, 'var').mat
                map_data = stat_func.r2z(map_data, metric='pearson')
                map_data = np.mean(map_data, axis=0)
                if map_idx > 1 and n_metrics > 1:
                    plot_data[type_idx, map_idx, 0] = map_data
                else:
                    plot_data[type_idx, map_idx] = map_data

    plot_data = -np.diff(plot_data, axis = 0)
    plot_data = np.squeeze(plot_data, axis = 0)

    plt.clf()
    plt.figure(figsize=(5, 4))

    model_labels = ['Human', 'RTNet', 'AlexNet', 'ResNet18']
    map_labels = [f'Human-{label}' for label in model_labels]
    colors = plt.cm.get_cmap('Set1', 8)
    plt.xticks([1.2, 5.4, 10.4], 
                ['Acc-Conf', 'Acc-RT', 'Conf-RT'],
                fontsize=12
            )

    plot_data_in_r = stat_func.z2r(plot_data, metric='pearson') 
    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 5 + map * 0.8
            plt.bar(x_pos,
                np.nanmean(plot_data_in_r[map,met, :]),
                yerr=sem(plot_data_in_r[map,met,:]),
                color=colors(map), label=map_labels[map] if met == 0 else None,
                alpha = 0.5
                )
            for k in range(n_subjs):
                plt.scatter(x_pos - 0.25,
                            plot_data_in_r[map, met, k],
                            color=colors(map), s=5
                            )

    for met in range(n_metrics):
        sub_data = plot_data[:, met, :]
        for i in range(sub_data.shape[0]):
            for j in range(i + 1, sub_data.shape[0]):
                if name == 'mnist':
                    t_stat, p_val = stats.ttest_ind(sub_data[i], sub_data[j], equal_var=False, nan_policy='omit')
                    try:
                        bayes10 = float(pg.ttest(sub_data[i], sub_data[j], paired=False)['BF10'].values[0])
                    except Exception:
                        bayes10 = np.nan
                    bayes01 = 1 / bayes10
                    mean_diff = np.nanmean(sub_data[i]) - np.nanmean(sub_data[j])
                    pooled_std = np.sqrt((np.nanvar(sub_data[i], ddof=1) + np.nanvar(sub_data[j], ddof=1)) / 2)
                    cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
                    print(f"Metric: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {p_val:.6f}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # Check for significance
                    if ((map_labels[i] == 'Human-Human')):
                        if met != 0 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        x_mid = (met * 5 + i * 0.8 + met * 5 + j * 0.8) / 2  # Midpoint between bars
                        y_max = max(np.nanmax(plot_data_in_r[:, met, :]), np.nanmax(plot_data_in_r[:, met, :])) + 0.1 * abs(j - i)
                        if p_val < 0.05:
                            alpha = 0.5
                        else:
                            alpha = 1
                        if p_val < 1e-3:
                            power = int(np.floor(np.log10(p_val)))
                            coefficient = p_val / (10 ** power)
                            anno = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
                        else:
                            anno = r"$p = {:.3f}$".format(p_val)
                        plt.plot([met * 5 + i * 0.8, met * 5 + j * 0.8], [y_max, y_max], color='black', linewidth=1.5,alpha=alpha)
                        plt.plot([met * 5 + i * 0.8, met * 5 + j * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                        plt.annotate(anno, (x_mid, y_max+0.02), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha)
                        plt.ylim(-0.4, 1.3)

                if name == 'ecoset10':
                    t_stat, p_val = stats.ttest_1samp(sub_data[j], 0, nan_policy='omit')
                    try:
                        bayes10 = float(pg.ttest(sub_data[i], sub_data[j], paired=False)['BF10'].values[0])
                    except Exception:
                        bayes10 = np.nan
                    bayes01 = 1 / bayes10
                    mean_diff = np.nanmean(sub_data[i]) - np.nanmean(sub_data[j])
                    pooled_std = np.sqrt((np.nanvar(sub_data[i], ddof=1) + np.nanvar(sub_data[j], ddof=1)) / 2)
                    # cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
                    cohen_d = t_stat / np.sqrt(n_subjs) if n_subjs != 0 else np.nan
                    print(f"Metric: {['Accuracy-Confidence', 'Accuracy-Reaction time', 'Reaction time-Confidence'][met]}, "
                        f"Comparison: {map_labels[i]} vs {map_labels[j]} - "
                        f"t-stat: {t_stat:.4f}, p-value: {p_val:.6f}, "
                        f"BF10: {bayes10:.4f}, BF01: {bayes01:.4f}, "
                        f"Cohen's d: {cohen_d:.4f}")

                    # compare to 0 signfiicance
                    if ((map_labels[i] == 'Human-Human')):
                        if met != 0 and not (map_labels[i] == 'Human-Human' and map_labels[j] == 'Human-RTNet'):
                            continue
                        x_pos = (met * 5 + j * 0.8) + 0.15
                        y_max = -0.04
                        if p_val < 1e-3:
                            anno = '***'
                            alpha = 1
                        elif p_val < 0.01:
                            anno = '**'
                            alpha = 1
                        elif p_val < 0.05:
                            anno = '*'
                            alpha = 1
                        else:
                            anno = 'n.s.'
                            alpha = 0.5
                        plt.annotate(anno, (x_pos, y_max), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha, fontweight='bold')
                        plt.ylim(-0.25, 0.6)

    plt.xlabel('Pairs of behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel(r'$r_{same\ subject} - r_{other\ subjects}$ ', fontsize=12, fontweight='bold')
    plt.title('Correlation consistency', fontsize=16, fontweight='bold')  
    plt.legend(loc='upper right', fontsize=8, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    path_name = path / f'corr_btw_var_{name}.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()

    # plot rank consistency
    plot_data = np.empty(shape=(n_maps, n_boots, n_metrics))
    plot_data.fill(np.nan)
    for map_idx in range(n_maps):
        if map_idx == 0:
            map_data = data[map_idx].get_rank_results('subj', 'subj', 'var').mat
            plot_data[map_idx] = map_data
        else:
            map_data = data[map_idx-1].get_rank_results('subj', 'inst', 'var').mat
            if map_idx > 1 and n_metrics > 1:
                plot_data[map_idx, :] = map_data
            else:
                plot_data[map_idx] = map_data

    plt.clf()
    plt.figure(figsize=(5, 4))

    p_bar_pos = 10
    p_val_pos = 1
    plt.xticks([1.2, 5.4, 10.4], 
                ['Acc-Conf', 'Acc-RT', 'Conf-RT'],
                fontsize=12
            )

    for map in range(n_maps):
        for met in range(n_metrics):
            x_pos = met * 5 + map * 0.8
            if met != 0 and not (map <= 1):
                continue
            plt.bar(x_pos,
                np.nanmean(plot_data[map,:,met]),
                yerr=sem(plot_data[map,:, met]),
                color=colors(map), label=map_labels[map] if met == 0 else None,
                alpha = 0.5
                )

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
                    f"p-value: {p_val:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met != 0 and not (map == 1):
                    continue
                x_mid = (met * 5 + met * 5 + map * 0.8) / 2  # Midpoint between bars
                y_max = np.nanmax(plot_data[:, :, met]) + p_bar_pos * abs(map - 0) - 20
                # y_max = np.nanmax(plot_data[:, :, met]) + p_bar_pos * abs(map - 0)
                alpha = 1
                if p_val < 0.001:
                    anno = f'$p$ < 0.001'
                else:
                    anno = f'$p$ = {p_val:.3f}'
                plt.plot([met * 5, met * 5 + map * 0.8], [y_max, y_max], color='black', linewidth=1.5, alpha=alpha)
                plt.annotate(anno, (x_mid, y_max+p_val_pos), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha)
                plt.ylim(0, 140)

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
                    f"p-value: {p_val:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                if met != 0 and not (_map == 1):
                    continue
                x_pos = (met * 5 + _map * 0.8)
                y_max = -0.02
                if p_val < 1e-3:
                    anno = '***'
                    alpha = 1
                elif p_val < 0.01:
                    anno = '**'
                    alpha = 1
                elif p_val < 0.05:
                    anno = '*'
                    alpha = 1
                else:
                    anno = 'n.s.'
                    alpha = 0.5
                plt.annotate(anno, (x_pos, y_max), textcoords="offset points", xytext=(0, 1), ha='center', size=8, alpha=alpha, fontweight='bold')  
                plt.ylim(0, 100)

    plt.xlabel('Pairs of behavioral metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Rank consistency metric', fontsize=12)
    plt.title('Rank consistency', fontsize=16, fontweight='bold')  
    plt.legend(loc='upper right', fontsize=8, frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    path_name = path / f'rank_btw_var_{name}.png'
    plt.savefig(path_name, dpi=384, transparent=True)
    plt.close()

