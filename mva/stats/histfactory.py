from . import log; log = log[__name__]
from rootpy.plotting import Hist
from rootpy.memory.keepalive import keepalive
import ROOT


def to_uniform_binning(hist):
    """
    For some obscure technical reason, HistFactory can't handle histograms with
    variable width bins. This function takes any 1D histogram and outputs a new
    histogram with constant width bins by using the bin indices of the input
    histogram as the x-axis of the new histogram.
    """
    new_hist = Hist(len(hist), 0, len(hist))
    # assume yerrh == yerrl (as usual for ROOT histograms)
    for i, (value, error) in enumerate(zip(hist, hist.yerrh())):
        new_hist.SetBinContent(i + 1, value)
        new_hist.SetBinError(i + 1, error)
    return new_hist


def make_channel(name, samples, data=None):

    log.info("creating channel %s" % name)
    chan = ROOT.RooStats.HistFactory.Channel(name)
    if data is not None:
        log.info("setting data")
        chan.SetData(data)
    chan.SetStatErrorConfig(0.05, "Poisson")

    for sample in samples:
        log.info("adding sample %s" % sample.GetName())
        chan.AddSample(sample)
    keepalive(chan, *samples)

    return chan


def make_measurement(name, title,
                     channels,
                     lumi=1.0, lumi_rel_error=0.,
                     output_prefix='./histfactory',
                     POI=None):
    """
    Note: to use the workspace in-memory use::
         static RooWorkspace* MakeCombinedModel(Measurement& measurement);
    """

    # Create the measurement
    log.info("creating measurement %s" % name)
    meas = ROOT.RooStats.HistFactory.Measurement(name, title)

    meas.SetOutputFilePrefix(output_prefix)
    if POI is not None:
        if isinstance(POI, basestring):
            log.info("setting POI %s" % POI)
            meas.SetPOI(POI)
        else:
            for p in POI:
                log.info("adding POI %s" % p)
                meas.AddPOI(p)

    log.info("setting lumi=%f +/- %f" % (lumi, lumi_rel_error))
    meas.SetLumi(lumi)
    meas.SetLumiRelErr(lumi_rel_error)
    meas.SetExportOnly(True)

    for channel in channels:
        log.info("adding channel %s" % channel.GetName())
        meas.AddChannel(channel)
    keepalive(meas, *channels)

    return meas


def make_model(measurement, channel=None):

    hist2workspace = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast(measurement)
    if channel is not None:
        return hist2workspace.MakeSingleChannelModel(measurement, channel)
    return hist2workspace.MakeCombinedModel(measurement)
