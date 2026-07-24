""" merge the 60 control-instance results (n=0..59) into a single averaged npy
per model/dataset combination, then plot the control-vs-standard consistency
figures (MNIST and ecoset10). """
import argparse
import pathlib

import numpy as np

from indimap import IndiMap
from indimap.util import stat_func
import util.dataset as dataset
import util.plotting as plotting


N_INSTANCES = 60

COMBOS = {
    'mnist_rtnet': dataset.get_rtnet_on_mnist_control,
    'mnist_alexnet': dataset.get_alexnet_on_mnist_control,
    'mnist_resnet18': dataset.get_resnet18_on_mnist_control,
    'ecoset10_rtnet': dataset.get_rtnet_on_ecoset10_control,
    'ecoset10_alexnet': dataset.get_alexnet_on_ecoset10_control,
    'ecoset10_resnet18': dataset.get_resnet18_on_ecoset10_control,
}

MNIST_COMBOS = ['mnist_rtnet', 'mnist_alexnet', 'mnist_resnet18']
ECOSET10_COMBOS = ['ecoset10_rtnet', 'ecoset10_alexnet', 'ecoset10_resnet18']

# standard (non-control, single-instance) getters - used as the star+SEM reference
STANDARD_GETTERS = {
    'mnist_rtnet': dataset.get_rtnet_on_mnist,
    'mnist_alexnet': dataset.get_alexnet_on_mnist,
    'mnist_resnet18': dataset.get_resnet18_on_mnist,
    'ecoset10_rtnet': dataset.get_rtnet_on_ecoset10,
    'ecoset10_alexnet': dataset.get_alexnet_on_ecoset10,
    'ecoset10_resnet18': dataset.get_resnet18_on_ecoset10,
}

# instance numbers available per combo - all combos have 0..59 except
# ecoset10_rtnet, which is only available for 1..59
INSTANCE_RANGES = {
    combo_name: range(N_INSTANCES) for combo_name in COMBOS
}
INSTANCE_RANGES['ecoset10_rtnet'] = range(1, N_INSTANCES)

# (mode, group2, map_type, split_by) combinations needed to later reproduce the
# within/across-metric correlation-consistency plots (mirrors the accuracy_control
# plotting functions' calls to get_corr_results('subj', group2, map_type, mode, split_by))
CORR_MODES = ['split', 'var']
CORR_GROUP2 = ['subj', 'inst']
CORR_MAP_TYPES = ['subj', 'subj_gp']
CORR_SPLIT_BY = ['rand', 'cate']

# (mode, group2) combinations needed for the rank-consistency plots
# (mirrors calls to get_rank_results('subj', group2, mode))
RANK_MODES = ['split', 'var']
RANK_GROUP2 = ['subj', 'inst']


def manage_path():
    current_path = pathlib.Path(__file__).parent.absolute()
    merged_path = current_path / 'IndiMap_results' / 'control' / 'merged'
    merged_path.mkdir(parents=True, exist_ok=True)
    graph_path = current_path / 'graphs' / 'control'
    graph_path.mkdir(parents=True, exist_ok=True)
    return merged_path, graph_path


def merge_combo(getter, instance_range, load=True):
    """ Merge the given control instances of a single model/dataset
    combination into one dict of arrays, each shaped like a single instance's
    raw result (n_boots, ...) but averaged across instances.

    Correlations are averaged in Fisher z-space (stat_func.r2z) and left in
    z-space - the diff-then-z2r step belongs to the eventual plotting script,
    same as how a single instance's bootstrap draws are handled elsewhere in
    this codebase. Rank results are averaged as-is.
    """
    corr_lists = {}
    rank_lists = {}

    for n in instance_range:
        obj = IndiMap(getter(n))
        obj.compute_corr(load_exists=load)
        obj.compute_rank(load_exists=load)

        for mode in CORR_MODES:
            for group2 in CORR_GROUP2:
                for map_type in CORR_MAP_TYPES:
                    for split_by in CORR_SPLIT_BY:
                        key = (mode, group2, map_type, split_by)
                        mat = obj.get_corr_results('subj', group2, map_type, mode, split_by).mat
                        z_mat = stat_func.r2z(mat, metric='pearson')
                        corr_lists.setdefault(key, []).append(z_mat)

        for mode in RANK_MODES:
            for group2 in RANK_GROUP2:
                key = (mode, group2)
                mat = obj.get_rank_results('subj', group2, mode).mat
                rank_lists.setdefault(key, []).append(mat)

    merged = {}
    for (mode, group2, map_type, split_by), mats in corr_lists.items():
        stacked = np.stack(mats, axis=0)  # (n_instances, n_boots, n_metrics, n_subjs)
        merged[f'corr_{mode}_{group2}_{map_type}_{split_by}'] = np.nanmean(stacked, axis=0)

    for (mode, group2), mats in rank_lists.items():
        stacked = np.stack(mats, axis=0)  # (n_instances, n_boots, n_metrics)
        merged[f'rank_{mode}_{group2}'] = np.nanmean(stacked, axis=0)

    merged['n_instances'] = len(instance_range)
    return merged


def load_merged(merged_path, combo_name):
    file_path = merged_path / f'{combo_name}_merged.npy'
    return np.load(file_path, allow_pickle=True).item()


def init_standard_maps():
    """ Build the standard (non-control) IndiMap objects used as the star+SEM reference. """
    return {
        combo_name: IndiMap(getter(variant='standard'))
        for combo_name, getter in STANDARD_GETTERS.items()
    }


def compute_standard(standard_maps, load=True):
    for obj in standard_maps.values():
        obj.compute_corr(load_exists=load)
        obj.compute_rank(load_exists=load)


def graph(standard_maps, merged_results, path):
    """ Plot control-vs-standard consistency figures for MNIST and ecoset10. """
    for combo_names in [MNIST_COMBOS, ECOSET10_COMBOS]:
        name = combo_names[0].split('_')[0]
        standard_data = [standard_maps[c] for c in combo_names]
        merged_control = [merged_results[c] for c in combo_names]

        plotting.plot_corr_within_metric_consistency_control(standard_data, merged_control, name, path, split_by='rand')
        plotting.plot_corr_within_metric_consistency_control(standard_data, merged_control, name, path, split_by='cate')
        plotting.plot_rank_within_metric_consistency_control(standard_data, merged_control, name, path)
        plotting.plot_corr_across_metric_consistency_control(standard_data, merged_control, name, path, split_by='rand')
        plotting.plot_corr_across_metric_consistency_control(standard_data, merged_control, name, path, split_by='cate')
        plotting.plot_rank_across_metric_consistency_control(standard_data, merged_control, name, path)


def main():
    parser = argparse.ArgumentParser(description="Merge MNIST/ecoset10 control-instance results (n=0..59) and plot")
    parser.add_argument('--combo', default=None, choices=list(COMBOS.keys()),
                        help='Only (re)merge this combination (default: all six)')
    parser.add_argument('--recompute', default=False, action='store_true',
                        help='Recompute per-instance/standard results instead of loading them')
    parser.add_argument('--remerge', default=False, action='store_true',
                        help='Recompute the merge across instances instead of reading the saved merged .npy files')
    parser.add_argument('--no-plot', default=False, action='store_true',
                        help='Skip plotting (useful when only (re)merging one combo via --combo on a server)')
    args = parser.parse_args()
    load = not args.recompute

    merged_path, graph_path = manage_path()
    combos_to_merge = COMBOS if args.combo is None else {args.combo: COMBOS[args.combo]}

    if args.remerge:
        for combo_name, getter in combos_to_merge.items():
            instance_range = INSTANCE_RANGES[combo_name]
            print(f'Merging {combo_name} ({len(instance_range)} instances: '
                  f'{instance_range.start}..{instance_range.stop - 1})...')
            merged = merge_combo(getter, instance_range, load=load)
            out_path = merged_path / f'{combo_name}_merged.npy'
            np.save(out_path, merged, allow_pickle=True)
            print(f'Saved {out_path}')

    if args.no_plot:
        return

    print('Reading merged control results...')
    merged_results = {combo_name: load_merged(merged_path, combo_name) for combo_name in COMBOS}

    print('Loading standard (non-control) reference maps...')
    standard_maps = init_standard_maps()
    compute_standard(standard_maps, load=load)

    graph(standard_maps, merged_results, graph_path)


if __name__ == "__main__":
    main()
