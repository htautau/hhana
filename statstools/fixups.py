from .histfactory import process_measurement
from rootpy.stats.histfactory import measurements_from_xml, write_measurement
from rootpy.io import MemFile
import os

from . import log; log = log[__name__]


def fix(input, suffix='fixed', verbose=False):
    """
    Traverse all workspaces and apply HSG4 fixes
    """
    if not os.path.isdir(input):
        raise ValueError("input must be an existing directory")
    input = os.path.normpath(input)
    output = input + '_' + suffix
    for dirpath, dirnames, filenames in os.walk(input):
        # does this directory contain a workspace?
        if not 'HistFactorySchema.dtd' in filenames:
            continue
        # find the top-level combination XMLs
        for filename in filenames:
            with open(os.path.join(dirpath, filename)) as f:
                if '<Combination' in f.read():
                    path = os.path.join(dirpath, filename)
                    log.info("fixing {0} ...".format(path))
                    measurements = measurements_from_xml(
                        path,
                        cd_parent=True,
                        collect_histograms=True,
                        silence=not verbose)
                    for meas in measurements:
                        with MemFile():
                            fix_measurement(meas)
                            write_measurement(meas,
                                output_path=os.path.join(
                                    output, dirpath.replace(input, '', 1)[1:]),
                                write_workspaces=True,
                                silence=not verbose)


def fix_measurement(meas):
    """
    Apply the HSG4 fixes on a HistFactory::Measurement
    Changes are applied in-place
    """
    process_measurement(meas,
        split_norm_shape=True,
        drop_np_names=["*"],
        drop_np_types=['histosys'],
        drop_np_samples=['Others'],
        prune_overallsys=True,
        prune_overallsys_threshold=1., # percent
        uniform_binning=True,
        #fill_empties=True,
        #fill_empties_samples=['Fakes', 'Ztautau']
        )
    #--prune-histosys --prune-histosys-method chi2 --prune-histosys-threshold 0.99 \
    #--prune-histosys-blacklist QCDscale_ggH3in \
    #--rebin-channels channel_vbf_${mass} channel_boosted_${mass} \
    #--rebin 10 from Makefile

    process_measurement(meas,
        drop_np_names=["*TAU_ID*"],
        drop_np_types=["OverallSys"],
        drop_np_samples=['Ztautau'],
        #symmetrize_names=["*JVF*", "*TES_TRUE*", "*EMB_MFS*"],
        #symmetrize_types=["overallsys", "histosys"]
        )
    #--smooth-histosys --smooth-histosys-iterations 1 \
    #--smooth-histosys-samples Fakes Ztautau Others "Signal_VBF_*" "Signal_gg_*" \
    #--prune-histosys --prune-histosys-samples Fakes Others Ztautau \
    #--prune-histosys-method max --prune-histosys-threshold 0.1
