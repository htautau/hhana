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

    chan = ROOT.RooStats.HistFactory.Channel(name)
    if data is not None:
        # data is a TF1
        chan.SetData(data)
    chan.SetStatErrorConfig(0.05, "Poisson")

    for sample in samples:
        chan.AddSample(sample)

    return chan


def make_measurement(name, title,
                     channels,
                     lumi=1.0, lumi_rel_error=0.,
                     output_prefix='./histfactory',
                     POI='SigXsecOverSM'):
    """
    Note: to use the workspace in-memory use::
         static RooWorkspace* MakeCombinedModel(Measurement& measurement);
    """

    # Create the measurement
    meas = ROOT.RooStats.HistFactory.Measurement(name, title)

    meas.SetOutputFilePrefix(output_prefix)
    meas.SetPOI(POI)
    meas.AddConstantParam("Lumi")

    meas.SetLumi(lumi)
    meas.SetLumiRelErr(lumi_rel_error)
    meas.SetExportOnly(False)

    for channel in channels:
        meas.AddChannel(channel)

    return meas


def make_workspace(measurement):

    return ROOT.RooStats.HistFactory.HistoToWorkspaceFactory.MakeCombinedModel(measurement)
