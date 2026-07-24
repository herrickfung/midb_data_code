""" category-map analysis and plot script.

Groups behavioral consistency by stimulus category (map_together='stim')
instead of by individual item, for both MNIST and ecoset10. This reuses the
original plotting functions directly - no new plotting code was needed since
they only depend on task_name ('mnist'/'ecoset10'), which is unchanged here,
and this analysis has no "standard"/reference results to overlay. There is no
split-by-category bootstrap variant for this analysis (it is already grouped
by category throughout), so the corr functions are only called with
split_by='rand'.
"""
import argparse
import pathlib

from indimap import IndiMap
import util.dataset as dataset
import util.plotting as plotting


def manage_path():
    current_path = pathlib.Path(__file__).parent.absolute()
    graph_path = current_path / 'graphs' / 'category_map'
    graph_path.mkdir(parents=True, exist_ok=True)
    return current_path, graph_path


def init_map():
    """ Initialize IndiMap objects for the category-map analysis """
    return [
        IndiMap(dataset.get_rtnet_on_mnist_category_map()),
        IndiMap(dataset.get_alexnet_on_mnist_category_map()),
        IndiMap(dataset.get_resnet18_on_mnist_category_map()),
        IndiMap(dataset.get_rtnet_on_ecoset10_category_map()),
        IndiMap(dataset.get_alexnet_on_ecoset10_category_map()),
        IndiMap(dataset.get_resnet18_on_ecoset10_category_map()),
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

        plotting.plot_top_identifiability(data, expt, path)
        plotting.plot_corr_within_metric_consistency(data, expt, path, split_by='rand')
        plotting.plot_rank_within_metric_consistency(data, expt, path)
        plotting.plot_corr_across_metric_consistency(data, expt, path, split_by='rand')
        plotting.plot_rank_across_metric_consistency(data, expt, path)


def main():
    parser = argparse.ArgumentParser(description="Run category-map analysis and plotting for IndiMap experiments")
    parser.add_argument('--recompute', default=False, action='store_true',
                        help='Recompute all results instead of loading')
    args = parser.parse_args()
    load = not args.recompute

    _, graph_path = manage_path()
    all_maps = init_map()
    compute(all_maps, load=load)
    graph(all_maps, graph_path)


if __name__ == "__main__":
    main()
