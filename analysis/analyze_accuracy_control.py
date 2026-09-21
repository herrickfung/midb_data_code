""" accuracy-control analysis and plot script (MNIST only) """
import argparse
import pathlib

from indimap import IndiMap
import util.dataset as dataset
import util.plotting as plotting


SDS = [-2, -1, 0, 1, 2]


def manage_path():
    current_path = pathlib.Path(__file__).parent.absolute()
    graph_path = current_path / 'graphs' / 'accuracy_control'
    graph_path.mkdir(parents=True, exist_ok=True)
    return current_path, graph_path


def init_map():
    """ Initialize IndiMap objects for the MNIST accuracy-control analysis.

    Returns a list indexed by sd position (matching SDS), each entry a
    [rtnet, alexnet, resnet18] list of IndiMap objects for that sd. sd=0 uses
    the standard (uncontrolled) trained models as the reference point.
    """
    standard_getters = [
        dataset.get_rtnet_on_mnist,
        dataset.get_alexnet_on_mnist,
        dataset.get_resnet18_on_mnist,
    ]
    control_getters = [
        dataset.get_rtnet_on_mnist_accuracy_control,
        dataset.get_alexnet_on_mnist_accuracy_control,
        dataset.get_resnet18_on_mnist_accuracy_control,
    ]

    all_maps = []
    for sd in SDS:
        if sd == 0:
            sd_maps = [IndiMap(getter(variant='standard')) for getter in standard_getters]
        else:
            sd_maps = [IndiMap(getter(sd)) for getter in control_getters]
        all_maps.append(sd_maps)
    return all_maps


def compute(all_maps, load=True):
    """ Compute/load all results """
    for sd_maps in all_maps:
        for obj in sd_maps:
            obj.compute_corr(load_exists=load)
            obj.compute_rank(load_exists=load)
            obj.compute_top(load_exists=load)


def graph(all_maps, path):
    """ Plot accuracy-control figures """
    plotting.plot_corr_within_metric_consistency_accuracy_control(all_maps, SDS, path, split_by='rand')
    # plotting.plot_corr_within_metric_consistency_accuracy_control(all_maps, SDS, path, split_by='cate')
    plotting.plot_rank_within_metric_consistency_accuracy_control(all_maps, SDS, path)
    plotting.plot_corr_across_metric_consistency_accuracy_control(all_maps, SDS, path, split_by='rand')
    # plotting.plot_corr_across_metric_consistency_accuracy_control(all_maps, SDS, path, split_by='cate')
    plotting.plot_rank_across_metric_consistency_accuracy_control(all_maps, SDS, path)
    plotting.plot_corr_within_metric_consistency_accuracy_control_grouped(all_maps, SDS, path, split_by='rand')
    plotting.plot_rank_within_metric_consistency_accuracy_control_grouped(all_maps, SDS, path)
    plotting.plot_corr_across_metric_consistency_accuracy_control_grouped(all_maps, SDS, path, split_by='rand')
    plotting.plot_rank_across_metric_consistency_accuracy_control_grouped(all_maps, SDS, path)
    plotting.plot_corr_sd_matrix_consistency_accuracy_control(all_maps, SDS, path, split_by='rand')


def main():
    parser = argparse.ArgumentParser(description="Run MNIST accuracy-control analysis and plotting")
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
