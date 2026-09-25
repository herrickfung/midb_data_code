""" Reproduce all figures and statistics reported in the paper.

By default, the data and precomputed IndiMap results are downloaded from
Harvard Dataverse (only the archives needed for the requested analyses) and
loaded, so figures and statistics are regenerated without recomputing anything.

    python analyze.py                          # all four analyses
    python analyze.py main control             # a subset
    python analyze.py untrained --recompute    # recompute results from the raw data
    python analyze.py control --remerge        # re-merge the 60 per-instance control results

Analyses:
    main              - Figures 1-5 and Supplementary Figures 1-3 and 6-11
    accuracy_control  - Figure 4a and Supplementary Figure 4 (MNIST)
    control           - Figure 4b and Supplementary Figure 5 (MNIST pseudo-instances)
    untrained         - Supplementary Figure 12 (untrained RTNet, MNIST)

Figures are saved to graphs/<analysis>/ and the printed statistics to
graphs/<analysis>/stats.txt.
"""
import argparse
import os
import pathlib
import sys
import tarfile

import numpy as np
import requests

from indimap import IndiMap
from indimap.util import stat_func
import util.dataset as dataset
import util.plotting as plotting


ROOT = pathlib.Path(__file__).parent.absolute()
GRAPH_ROOT = ROOT / 'graphs'
ANALYSES = ['main', 'accuracy_control', 'control', 'untrained']

# Harvard Dataverse dataset hosting the data archives (<name>.tar.gz)
DATAVERSE_URL = 'https://dataverse.harvard.edu'
DATAVERSE_DOI = 'doi:10.7910/DVN/DVVXJL'
DATAVERSE_VERSION = ':latest-published'
# Harvard Dataverse rejects requests' default User-Agent with 403
HTTP_HEADERS = {'User-Agent': 'midb_data_code (https://github.com/herrickfung/midb_data_code)'}

# archive name -> a file whose presence means the archive is already extracted
DATA_ARCHIVES = {
    'midb_data': 'dataset/mnist/standard/human.csv',
    'midb_results_standard_mnist': 'IndiMap_results/standard/mnist_rtnet/CorrMap_results.npz',
    'midb_results_standard_ecoset10': 'IndiMap_results/standard/ecoset10_rtnet/CorrMap_results.npz',
    'midb_results_accuracy_control_rtnet': 'IndiMap_results/accuracy_control/mnist_rtnet/2sd/CorrMap_results.npz',
    'midb_results_accuracy_control_alexnet_resnet18': 'IndiMap_results/accuracy_control/mnist_resnet18/2sd/CorrMap_results.npz',
    'midb_results_control': 'IndiMap_results/control/merged/mnist_rtnet_merged.npy',
    'midb_results_untrained': 'IndiMap_results/untrained/mnist_rtnet/CorrMap_results.npz',
}

# precomputed-result archives needed by each analysis (on top of midb_data)
RESULT_ARCHIVES = {
    'main': ['midb_results_standard_mnist', 'midb_results_standard_ecoset10'],
    'accuracy_control': ['midb_results_standard_mnist', 'midb_results_accuracy_control_rtnet',
                         'midb_results_accuracy_control_alexnet_resnet18'],
    'control': ['midb_results_standard_mnist', 'midb_results_control'],
    'untrained': ['midb_results_standard_mnist', 'midb_results_untrained'],
}

STANDARD_GETTERS = {
    'mnist': [dataset.get_rtnet_on_mnist, dataset.get_alexnet_on_mnist, dataset.get_resnet18_on_mnist],
    'ecoset10': [dataset.get_rtnet_on_ecoset10, dataset.get_alexnet_on_ecoset10, dataset.get_resnet18_on_ecoset10],
}


# ---------------------------------------------------------------------------
# data download
# ---------------------------------------------------------------------------
def dataverse_file_ids():
    """ Map each file name in the Dataverse dataset to its file id. """
    response = requests.get(
        f'{DATAVERSE_URL}/api/datasets/:persistentId/versions/{DATAVERSE_VERSION}/files',
        params={'persistentId': DATAVERSE_DOI}, headers=HTTP_HEADERS)
    response.raise_for_status()
    return {f['dataFile']['filename']: f['dataFile']['id'] for f in response.json()['data']}


def download_and_extract(name, file_ids):
    tar_path = ROOT / f'{name}.tar.gz'
    print(f"Downloading {name} from Harvard Dataverse, please wait ...")
    url = f'{DATAVERSE_URL}/api/access/datafile/{file_ids[tar_path.name]}'
    with requests.get(url, headers=HTTP_HEADERS, stream=True) as response:
        response.raise_for_status()
        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    print(f"Extracting {name} ...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=ROOT)
    tar_path.unlink()


def ensure_data(analyses, recompute):
    """ Download the raw data, plus the precomputed results unless recomputing. """
    needed = ['midb_data']
    if not recompute:
        for analysis in analyses:
            needed += [a for a in RESULT_ARCHIVES[analysis] if a not in needed]
    missing = [name for name in needed if not (ROOT / DATA_ARCHIVES[name]).exists()]
    if not missing:
        return
    file_ids = dataverse_file_ids()
    for name in missing:
        download_and_extract(name, file_ids)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Tee:
    """ Write to the console and to a stats file at the same time. """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def compute(obj, parts, load):
    """ Compute/load the requested result parts ('corr', 'rank', 'top', 'dims',
    'pred') of an IndiMap object, skipping parts already done in this run. """
    done = obj.__dict__.setdefault('_done', set())
    for part in parts:
        if part not in done:
            getattr(obj, f'compute_{part}')(load_exists=load)
            done.add(part)


_standard_maps = {}


def standard_maps(expt, parts, load):
    """ [rtnet, alexnet, resnet18] standard IndiMap objects for 'mnist' or
    'ecoset10', built once and shared by all analyses. """
    if expt not in _standard_maps:
        _standard_maps[expt] = [IndiMap(getter(variant='standard')) for getter in STANDARD_GETTERS[expt]]
    for obj in _standard_maps[expt]:
        compute(obj, parts, load)
    return _standard_maps[expt]


# ---------------------------------------------------------------------------
# main analysis: Figures 1-5, Supplementary Figures 1-3, 6-11
# ---------------------------------------------------------------------------
def run_main(args, path):
    load = not args.recompute
    plotting.plot_raw_matrix_colorbar(path)
    for expt in ['mnist', 'ecoset10']:
        print(f'\n===== {expt} =====')
        data = standard_maps(expt, ['corr', 'rank', 'top', 'pred'], load)
        for obj in data:
            # subject/instance response arrays used by the raincloud and raw-matrix plots
            obj.dims_map._convert_data_array()

        plotting.plot_raincloud(data, expt, path)                   # Fig 1b/5b, Supp Fig 1/8 (Wasserstein)
        map_mat, _ = plotting.plot_raw_matrix(data, expt, path)     # Fig 2a, Supp Fig 2
        plotting.plot_best_count_distribution(data, expt, path)     # Fig 2a histograms
        plotting.plot_alignment_variance(data, expt, path)          # Fig 2b/5c
        plotting.plot_alignment_average(map_mat, expt, path)        # Fig 2c/5d
        plotting.plot_corr_within_metric_consistency(data, expt, path, split_by='rand')   # Fig 2d/5e
        plotting.plot_rank_within_metric_consistency(data, expt, path, split_by='rand')   # Fig 2d, Supp Fig 9a
        plotting.plot_across_metric_illustration(map_mat, expt, path)                     # Fig 3a-b/5f
        plotting.plot_corr_across_metric_consistency(data, expt, path, split_by='rand')   # Fig 3c/5g
        plotting.plot_rank_across_metric_consistency(data, expt, path, split_by='rand')   # Fig 3c, Supp Fig 9b
        plotting.plot_best_match_freq_raw(data, expt, path)         # Supp Fig 3a
        plotting.plot_best_match_freq_fit(data, expt, path)         # Supp Fig 3b
        plotting.report_best_match_decay_significance(data, expt)
        raw_pred = plotting.plot_within_metric_prediction_raw(data, expt, path)  # Supp Fig 6/10b
        plotting.plot_within_metric_prediction_diff(raw_pred, expt, path)        # Fig 4c, Supp Fig 10a
        plotting.plot_top_identifiability(data, expt, path)         # Fig 4d, Supp Fig 11a
        plotting.plot_top_identifiability_raw(data, expt, path)     # Supp Fig 7/11b


# ---------------------------------------------------------------------------
# accuracy control: Figure 4a, Supplementary Figure 4
# ---------------------------------------------------------------------------
SDS = [-2, -1, 0, 1, 2]
ACCURACY_CONTROL_GETTERS = [
    dataset.get_rtnet_on_mnist_accuracy_control,
    dataset.get_alexnet_on_mnist_accuracy_control,
    dataset.get_resnet18_on_mnist_accuracy_control,
]


def run_accuracy_control(args, path):
    """ ANN populations whose average accuracy is matched to the human
    mean -2..+2 SD. sd=0 is the standard (uncontrolled) trained population. """
    load = not args.recompute
    all_maps = []
    for sd in SDS:
        if sd == 0:
            sd_maps = standard_maps('mnist', ['corr', 'rank'], load)
        else:
            sd_maps = [IndiMap(getter(sd)) for getter in ACCURACY_CONTROL_GETTERS]
            for obj in sd_maps:
                compute(obj, ['corr', 'rank'], load)
        all_maps.append(sd_maps)

    plotting.plot_corr_within_metric_consistency_accuracy_control_grouped(all_maps, SDS, path, split_by='rand')  # Fig 4a
    plotting.plot_corr_across_metric_consistency_accuracy_control_grouped(all_maps, SDS, path, split_by='rand')  # Supp Fig 4


# ---------------------------------------------------------------------------
# pseudo-instance control: Figure 4b, Supplementary Figure 5
# ---------------------------------------------------------------------------
N_CONTROL_INSTANCES = 60
CONTROL_GETTERS = {
    'mnist_rtnet': dataset.get_rtnet_on_mnist_control,
    'mnist_alexnet': dataset.get_alexnet_on_mnist_control,
    'mnist_resnet18': dataset.get_resnet18_on_mnist_control,
}
CONTROL_MERGED_PATH = ROOT / 'IndiMap_results' / 'control' / 'merged'

# (mode, group2, map_type, split_by) combinations needed to reproduce the
# within/across-metric correlation-consistency plots (mirrors the plotting
# functions' calls to get_corr_results('subj', group2, map_type, mode, split_by))
CORR_MODES = ['split', 'var']
CORR_GROUP2 = ['subj', 'inst']
CORR_MAP_TYPES = ['subj', 'subj_gp']
CORR_SPLIT_BY = ['rand', 'cate']

# (mode, group2) combinations needed for the rank-consistency plots
# (mirrors calls to get_rank_results('subj', group2, mode))
RANK_MODES = ['split', 'var']
RANK_GROUP2 = ['subj', 'inst']


def merge_control(getter, load):
    """ Merge the 60 pseudo-instance control results of a single model into
    one dict of arrays, each shaped like a single instance's raw result
    (n_boots, ...) but averaged across instances.

    Correlations are averaged in Fisher z-space (stat_func.r2z) and left in
    z-space - the diff-then-z2r step belongs to the plotting functions, same
    as how a single instance's bootstrap draws are handled elsewhere. Rank
    results are averaged as-is.
    """
    corr_lists = {}
    rank_lists = {}

    for n in range(N_CONTROL_INSTANCES):
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

    merged['n_instances'] = N_CONTROL_INSTANCES
    return merged


def run_control(args, path):
    load = not args.recompute
    if args.remerge:
        if not (ROOT / 'dataset' / 'mnist' / 'control').exists():
            sys.exit("--remerge needs the per-instance control data in dataset/mnist/control/, "
                     "which is not distributed with the data archives (see README).")
        CONTROL_MERGED_PATH.mkdir(parents=True, exist_ok=True)
        for combo_name, getter in CONTROL_GETTERS.items():
            print(f'Merging {combo_name} ({N_CONTROL_INSTANCES} instances) ...')
            merged = merge_control(getter, load)
            np.save(CONTROL_MERGED_PATH / f'{combo_name}_merged.npy', merged, allow_pickle=True)

    merged_control = [
        np.load(CONTROL_MERGED_PATH / f'{combo_name}_merged.npy', allow_pickle=True).item()
        for combo_name in CONTROL_GETTERS
    ]
    standard_data = standard_maps('mnist', ['corr', 'rank'], load)

    plotting.plot_corr_within_metric_consistency_control(standard_data, merged_control, 'mnist', path, split_by='rand')  # Fig 4b
    plotting.plot_rank_within_metric_consistency_control(standard_data, merged_control, 'mnist', path)                   # Fig 4b
    plotting.plot_corr_across_metric_consistency_control(standard_data, merged_control, 'mnist', path, split_by='rand')  # Supp Fig 5
    plotting.plot_rank_across_metric_consistency_control(standard_data, merged_control, 'mnist', path)                   # Supp Fig 5


# ---------------------------------------------------------------------------
# untrained RTNet: Supplementary Figure 12
# ---------------------------------------------------------------------------
def run_untrained(args, path):
    """ Untrained (randomly initialized) RTNet instances vs the trained RTNet
    instances on MNIST. Untrained AlexNet/ResNet18 are not analyzed as they
    give the same response to (almost) all images. """
    load = not args.recompute
    untrained = [IndiMap(dataset.get_rtnet_on_mnist('untrained'))]
    compute(untrained[0], ['corr', 'rank'], load)
    trained = standard_maps('mnist', ['corr', 'rank'], load)[:1]

    plotting.plot_corr_within_metric_consistency_untrained(trained, untrained, 'mnist', path, split_by='rand')  # Supp Fig 12a
    plotting.plot_corr_across_metric_consistency_untrained(trained, untrained, 'mnist', path, split_by='rand')  # Supp Fig 12b


RUNNERS = {
    'main': run_main,
    'accuracy_control': run_accuracy_control,
    'control': run_control,
    'untrained': run_untrained,
}


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the figures and statistics of the paper.")
    # choices are validated below: argparse on Python 3.9 rejects an empty
    # nargs='*' positional when `choices` is set
    parser.add_argument('analyses', nargs='*', metavar='ANALYSIS',
                        help='Analyses to run: ' + ', '.join(ANALYSES) + ' (default: all)')
    parser.add_argument('--recompute', default=False, action='store_true',
                        help='Recompute all results from the raw data instead of loading precomputed results')
    parser.add_argument('--remerge', default=False, action='store_true',
                        help='(control) Re-merge the 60 per-instance control results; '
                             'needs the per-instance data, which is not distributed')
    args = parser.parse_args()
    invalid = [a for a in args.analyses if a not in ANALYSES]
    if invalid:
        parser.error(f"invalid analysis {invalid}; choose from {ANALYSES}")
    analyses = [a for a in ANALYSES if a in (args.analyses or ANALYSES)]

    # util/dataset.py and IndiMap use paths relative to this directory
    os.chdir(ROOT)
    ensure_data(analyses, args.recompute)

    stdout = sys.stdout
    for analysis in analyses:
        path = GRAPH_ROOT / analysis
        path.mkdir(parents=True, exist_ok=True)
        print(f'\n########## {analysis} ##########')
        with open(path / 'stats.txt', 'w') as stats_file:
            sys.stdout = Tee(stdout, stats_file)
            try:
                RUNNERS[analysis](args, path)
            finally:
                sys.stdout = stdout
        print(f'Figures and stats saved to {path}')


if __name__ == "__main__":
    main()
