""" main analysis and plot script """
import argparse
import pathlib
import requests
import tarfile

from indimap import IndiMap
import util.dataset as dataset
import util.plotting as plotting



def manage_path():
    current_path = pathlib.Path(__file__).parent.absolute()
    graph_path = current_path / 'graphs'
    graph_path.mkdir(parents=True, exist_ok=True)
    return current_path, graph_path


def download_and_extract_data(current_path):
    OSF_DATA_URL = "https://files.osf.io/v1/resources/n6m7b/providers/osfstorage/69741b25a94a8b2b80bb82dc"
    tar_path = current_path / 'midb_data.tar.gz'

    data_path_for_check = [current_path / 'dataset', current_path / 'IndiMap_results']
    if all([p.exists() for p in data_path_for_check]):
        return

    print("Downloading data from OSF, please wait ...")
    response = requests.get(OSF_DATA_URL, stream=True)
    with open(tar_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=current_path)

    tar_path.unlink()


def init_map():
    """ Initialize IndiMap objects for all models and datasets """
    return [
        IndiMap(dataset.get_rtnet_on_mnist()),
        IndiMap(dataset.get_alexnet_on_mnist()),
        IndiMap(dataset.get_resnet18_on_mnist()),
        IndiMap(dataset.get_rtnet_on_ecoset10()),
        IndiMap(dataset.get_alexnet_on_ecoset10()),
        IndiMap(dataset.get_resnet18_on_ecoset10()),
    ]


def compute(all_maps, load=False):
    """ Compute/load all results """
    for obj in all_maps:
        obj.dims_map._convert_data_array()
        obj.compute_all(load_exists=load)


def graph(all_data, path):
    """ Plot figures """
    mnists, ecosets = all_data[:3], all_data[3:]
    for expt in ['mnist', 'ecoset10']:
        if expt == 'mnist':
            data = mnists
        else:
            data = ecosets

        plotting.plot_raincloud(data, expt, path)
        map_mat = plotting.plot_raw_matrix(data, expt, path)
        plotting.plot_alignment_average(map_mat, expt, path)
        plotting.plot_alignment_variance(data, expt, path)
        plotting.plot_across_metric_illustration(map_mat, expt, path)
        plotting.plot_best_count_distribution(data, expt, path)
        plotting.plot_within_metric_consistency(data, expt, path)
        plotting.plot_across_metric_consistency(data, expt, path)
        plotting.plot_pca_shuffle_comparison(data, expt, path)
        plotting.plot_pca_cumulative_evidence_lineplot(data, expt, path)
        plotting.plot_pca_total_variance_comparison(data, expt, path)
        plotting.plot_pca_total_variance_comparison_split_graph(data, expt, path)
        raw_pred = plotting.plot_within_metric_prediction_raw(data, expt, path)
        plotting.plot_within_metric_prediction_diff(raw_pred, expt, path)


def main():
    parser = argparse.ArgumentParser(description="Run analysis and plotting for IndiMap experiments")
    parser.add_argument('--recompute', default=False, action='store_true',
                        help='Recompute all results instead of loading')
    args = parser.parse_args()
    load = not args.recompute

    current_path, graph_path = manage_path()
    download_and_extract_data(current_path)
    all_maps = init_map()
    compute(all_maps, load=load)
    graph(all_maps, graph_path)


if __name__ == "__main__":
    main()