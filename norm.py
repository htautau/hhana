import ROOT

from mva.samples import Data, MC_Ztautau, Others, QCD
from mva.histfactory import make_channel, make_measurement, make_workspace

data = Data(2012)
ztt = MC_Ztautau(2012)
others = Others(2012)
qcd = QCD(data, (ztt, others))

expr = 'tau1_numTrack_recounted:tau2_numTrack_recounted'
min, max = .5, 5.5
bins = int(max - min)
category = '2j'
region = 'OS'

ztt_sample = ztt.get_histfactory_sample(expr, category, region, bins, min, max)
others_sample = others.get_histfactory_sample(expr, category, region, bins, min, max)
qcd_sample = qcd.get_histfactory_sample(expr, category, region, bins, min, max)
data_sample = data.get_histfactory_sample(expr, category, region, bins, min, max)

channel = make_channel(category,
        [ztt_sample, others_sample, qcd_sample], data_sample.GetHisto())
measurement = make_measurement('trackfit', '', [channel], lumi_rel_error=0.039,
        POI='Z_fraction')

hist2workspace = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast()
workspace = hist2workspace.MakeSingleChannelModel(
        measurement,
        channel)
#workspace = make_workspace(measurement)

print workspace
