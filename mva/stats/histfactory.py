from . import log; log = log[__name__]
import ROOT

"""
# Create the signal sample
signal = ROOT.RooStats.HistFactory.Sample("signal", "signal", InputFile)
signal.AddOverallSys("syst1",  0.95, 1.05)
signal.AddNormFactor("SigXsecOverSM", 1, 0, 3)
chan.AddSample(signal)

# Background 1
background1 = ROOT.RooStats.HistFactory.Sample("background1", "background1", InputFile)
background1.ActivateStatError("background1_statUncert", InputFile)
background1.AddOverallSys("syst2", 0.95, 1.05 )
chan.AddSample(background1)

# Background 1
background2 = ROOT.RooStats.HistFactory.Sample("background2", "background2", InputFile)
background2.ActivateStatError()
background2.AddOverallSys("syst3", 0.95, 1.05 )
"""

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

    return meas


def make_model(measurement, channel=None):

    hist2workspace = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast(measurement)
    if channel is not None:
        return hist2workspace.MakeSingleChannelModel(measurement, channel)
    return hist2workspace.MakeCombinedModel(measurement)
