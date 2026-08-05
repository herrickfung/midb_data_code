""" demeaning analysis script (standard/uncontrolled trained models, MNIST
and ecoset10).

Mapping stays trial-by-trial at the item level (map_together=
'mnist_index'/'image_index'), exactly like the plain standard analysis -
the addition is a control step applied before the IndiMap mapping stage:
the average-performer effect is removed from the response matrix (see
util/dataset.py's remove_category_average()/remove_item_average(), applied
inside each get_{model}_on_{task}_demeaning(variant) getter). --variant
category removes the category-level effect (grouped by 'stim'+'cond');
--variant item removes the item-level effect instead (grouped by the item
index+'cond' - finer-grained than category). Either way, each subject's/
instance's own trial-level acc/conf/rt is centered on the average response
across subjects/instances for that group/condition.

Plots reuse the same plot_corr_*/plot_rank_* functions as the standard
analysis (analyze_main.py), since the demeaning getters keep item-level
mapping (map_together='mnist_index'/'image_index') and so produce the
same [rtnet, alexnet, resnet18]-per-expt IndiMap list shape those
functions expect.
"""
import argparse
import pathlib

from indimap import IndiMap
import util.dataset as dataset
import util.plotting as plotting


def manage_path(variant):
    current_path = pathlib.Path(__file__).parent.absolute()
    graph_path = current_path / 'graphs' / f'{variant}_demeaning'
    graph_path.mkdir(parents=True, exist_ok=True)
    return current_path, graph_path


def init_map(variant):
    """ Initialize IndiMap objects for the demeaning analysis:
    [rtnet, alexnet, resnet18] for MNIST, followed by
    [rtnet, alexnet, resnet18] for ecoset10. """
    return [
        IndiMap(dataset.get_rtnet_on_mnist_demeaning(variant)),
        IndiMap(dataset.get_alexnet_on_mnist_demeaning(variant)),
        IndiMap(dataset.get_resnet18_on_mnist_demeaning(variant)),
        IndiMap(dataset.get_rtnet_on_ecoset10_demeaning(variant)),
        IndiMap(dataset.get_alexnet_on_ecoset10_demeaning(variant)),
        IndiMap(dataset.get_resnet18_on_ecoset10_demeaning(variant)),
    ]


def compute(all_maps, load=True):
    """ Compute/load all results """
    for obj in all_maps:
        obj.compute_corr(load_exists=load)
        obj.compute_rank(load_exists=load)
        obj.compute_top(load_exists=load)


def graph(all_maps, path):
    """ Plot figures """
    mnists, ecosets = all_maps[:3], all_maps[3:]
    for expt in ['mnist', 'ecoset10']:
        data = mnists if expt == 'mnist' else ecosets

        for split_by in ['rand', 'cate']:
            plotting.plot_corr_within_metric_consistency(data, expt, path, split_by=split_by)
            plotting.plot_rank_within_metric_consistency(data, expt, path, split_by=split_by)
            plotting.plot_corr_across_metric_consistency(data, expt, path, split_by=split_by)
            plotting.plot_rank_across_metric_consistency(data, expt, path, split_by=split_by)


def main():
    parser = argparse.ArgumentParser(description="Run demeaning analysis")
    parser.add_argument('--variant', default='category', choices=['category', 'item'],
                        help="Whether to remove the category-level or item-level average-performer effect")
    parser.add_argument('--recompute', default=False, action='store_true',
                        help='Recompute all results instead of loading')
    args = parser.parse_args()
    load = not args.recompute

    _, graph_path = manage_path(args.variant)
    all_maps = init_map(args.variant)
    compute(all_maps, load=load)
    graph(all_maps, graph_path)


if __name__ == "__main__":
    main()
