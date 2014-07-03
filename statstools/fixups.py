from .jobs import run_pool
from .histfactory import process_measurement
from rootpy.stats.histfactory import measurements_from_xml, write_measurement
from rootpy.io import MemFile
import os

from . import log; log = log[__name__]

from multiprocessing import Process


class Worker(Process):
    def __init__(self, path, output_path, verbose=False, **kwargs):
        super(Worker, self).__init__()
        self.path = path
        self.output_path = output_path
        self.verbose = verbose
        self.kwargs = kwargs

    def run(self):
        path = self.path
        output_path = self.output_path
        kwargs = self.kwargs
        measurements = measurements_from_xml(
            path,
            cd_parent=True,
            collect_histograms=True,
            silence=not self.verbose)
        for meas in measurements:
            root_file = os.path.join(output_path, '{0}.root'.format(meas.name))
            xml_path = os.path.join(output_path, meas.name)
            with MemFile():
                fix_measurement(meas, **kwargs)
                write_measurement(meas,
                    root_file=root_file,
                    xml_path=xml_path,
                    write_workspaces=True,
                    silence=not self.verbose)


def find_measurements(path):
    for dirpath, dirnames, filenames in os.walk(path):
        # does this directory contain a workspace?
        if not 'HistFactorySchema.dtd' in filenames:
            continue
        # find the top-level combination XMLs
        for filename in filenames:
            with open(os.path.join(dirpath, filename)) as f:
                if '<Combination' in f.read():
                    yield dirpath, filename


def fix(inputs, suffix='fixed', verbose=False, n_jobs=-1, **kwargs):
    """
    Traverse all workspaces and apply HSG4 fixes
    """
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    workers = []
    for input in inputs:
        if not os.path.isdir(input):
            raise ValueError("input must be an existing directory")
        input = os.path.normpath(input)
        output = input + '_' + suffix
        for dirpath, measurement_file in find_measurements(input):
            output_path = os.path.join(output, dirpath.replace(input, '', 1)[1:])
            path = os.path.join(dirpath, measurement_file)
            log.info("fixing {0} ...".format(path))
            workers.append(Worker(path, output_path, verbose=verbose, **kwargs))
    # run the workers
    run_pool(workers, n_jobs=n_jobs)


def fix_measurement(meas,
                    prune_shapes=False,
                    chi2_threshold=0.99,
                    symmetrize=False,
                    symmetrize_partial=False,
                    merge=False):
    """
    Apply the HSG4 fixes on a HistFactory::Measurement
    Changes are applied in-place
    """

    if merge:
        # merge bins in some channels
        process_measurement(meas,
                            merge_bins=[(6, 7), (8, 9)],
                            merge_bins_channels=['*cuts_vbf_highdr_loose_12*'])
        process_measurement(meas,
                            merge_bins=[(2, 3), (4, 5), (6, 7)],
                            merge_bins_channels=['*cuts_vbf_highdr_tight_12*'])
        process_measurement(meas,
                            merge_bins=[(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18)],
                            merge_bins_channels=['*cuts_boosted_tight_12*'])
        process_measurement(meas,
                            merge_bins=[(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20)],
                            merge_bins_channels=['*cuts_boosted_loose_12*'])

    # fill empty bins with the average sample weight
    # the so-called "Kyle-fix"
    process_measurement(meas,
        fill_empties=True,
        fill_empties_samples=['Fakes', 'Ztautau'])

    if symmetrize:
        # symmetrize NPs with double minima or kinks
        # do this before splitting into shape+norm
        process_measurement(
            meas,
            #symmetrize_names=[
            #    "*TES_TRUE_FINAL_2011*"
            #    "*TES_TRUE_MODELING*",
            #    "*ANA_EMB_ISOL*",
            #    "*ANA_EMB_MFS_2011*"],
            symmetrize_names=['*'],
            symmetrize_types=["histosys"],
            symmetrize_partial=symmetrize_partial,
            asymmetry_threshold=0.5)

    process_measurement(meas,
        split_norm_shape=True,
        drop_np_names=["*"],
        drop_np_types=['histosys'],
        drop_np_samples=['Others'],
        prune_overallsys=True,
        prune_overallsys_threshold=0.5, # percent
        uniform_binning=True)

    # ignore OverallSys on Ztt that is redundant with Ztt norm
    #process_measurement(meas,
    #    drop_np_names=["*TAU_ID*"],
    #    drop_np_types=["OverallSys"],
    #    drop_np_samples=['Ztautau'])

    if prune_shapes:
        # prune NPs with chi2 method
        process_measurement(meas,
            prune_histosys=True,
            prune_histosys_method='chi2',
            prune_histosys_threshold=chi2_threshold)
            #prune_histosys_blacklist=['QCDscale_ggH3in']) ?
        # prune NPs with max deviation method (only background)
        process_measurement(meas,
            prune_histosys=True,
            prune_histosys_method='max',
            prune_histosys_threshold=0.1, # 10%
            prune_histosys_samples=['Fakes', 'Others', 'Ztautau'])


