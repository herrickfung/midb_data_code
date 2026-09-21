""" untrained-network analysis and plot script.

Compares an untrained (randomly initialized) network against the standard
(trained) network, for both MNIST and ecoset10. The untrained network's
per-subject results form the boxplots; the standard network's reference value
is overlaid as a star marker (see util/plotting.py's *_untrained functions).
"""
import argparse
import pathlib

from indimap import IndiMap
import util.dataset as dataset
import util.plotting as plotting


def manage_path():
    current_path = pathlib.Path(__file__).parent.absolute()
    graph_path = current_path / 'graphs' / 'untrained'
    graph_path.mkdir(parents=True, exist_ok=True)
    return current_path, graph_path


def init_maps():
    """ Build the untrained and standard (trained) IndiMap objects, in the
    same [rtnet, alexnet, resnet18] x [mnist, ecoset10] order. """
    untrained_maps = [
        IndiMap(dataset.get_rtnet_on_mnist('untrained')),
        IndiMap(dataset.get_alexnet_on_mnist('untrained')),
        IndiMap(dataset.get_resnet18_on_mnist('untrained')),
        IndiMap(dataset.get_rtnet_on_ecoset10('untrained')),
        IndiMap(dataset.get_alexnet_on_ecoset10('untrained')),
        IndiMap(dataset.get_resnet18_on_ecoset10('untrained')),
    ]
    standard_maps = [
        IndiMap(dataset.get_rtnet_on_mnist('standard')),
        IndiMap(dataset.get_alexnet_on_mnist('standard')),
        IndiMap(dataset.get_resnet18_on_mnist('standard')),
        IndiMap(dataset.get_rtnet_on_ecoset10('standard')),
        IndiMap(dataset.get_alexnet_on_ecoset10('standard')),
        IndiMap(dataset.get_resnet18_on_ecoset10('standard')),
    ]
    return untrained_maps, standard_maps


def compute(all_maps, load=True):
    """ Compute/load all results """
    for obj in all_maps:
        # obj.dims_map._convert_data_array()
        obj.compute_corr(load_exists=load)
        obj.compute_rank(load_exists=load)
        obj.compute_top(load_exists=load)


def graph(untrained_maps, standard_maps, path):
    """ Plot untrained-vs-standard figures for MNIST and ecoset10. """
    untrained_mnist, untrained_ecoset = untrained_maps[:3], untrained_maps[3:]
    standard_mnist, standard_ecoset = standard_maps[:3], standard_maps[3:]

    for expt in ['mnist', 'ecoset10']:
        if expt == 'mnist':
            untrained_data, standard_data = untrained_mnist, standard_mnist
        else:
            untrained_data, standard_data = untrained_ecoset, standard_ecoset

        # plotting.plot_raw_matrix(untrained_data, expt, path)
        # plotting.plot_top_identifiability_untrained(standard_data, untrained_data, expt, path)

        # Human vs RTNet only (drop AlexNet/ResNet18), random split only
        rtnet_untrained, rtnet_standard = untrained_data[:1], standard_data[:1]
        plotting.plot_corr_within_metric_consistency_untrained(rtnet_standard, rtnet_untrained, expt, path, split_by='rand')
        plotting.plot_rank_within_metric_consistency_untrained(rtnet_standard, rtnet_untrained, expt, path)
        plotting.plot_corr_across_metric_consistency_untrained(rtnet_standard, rtnet_untrained, expt, path, split_by='rand')
        plotting.plot_rank_across_metric_consistency_untrained(rtnet_standard, rtnet_untrained, expt, path)


def main():
    parser = argparse.ArgumentParser(description="Run untrained-vs-standard analysis and plotting for IndiMap experiments")
    parser.add_argument('--recompute', default=False, action='store_true',
                        help='Recompute all results instead of loading')
    args = parser.parse_args()
    load = not args.recompute

    _, graph_path = manage_path()
    untrained_maps, standard_maps = init_maps()
    compute(untrained_maps, load=load)
    compute(standard_maps, load=load)
    graph(untrained_maps, standard_maps, graph_path)


if __name__ == "__main__":
    main()
