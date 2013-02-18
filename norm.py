import ROOT

from mva.samples import Data, MC_Ztautau, Others, QCD
from mva.histfactory import make_channel, make_measurement, make_workspace

from ROOT import RooMinimizer

data = Data(2012)
ztt = MC_Ztautau(2012, systematics=False)
others = Others(2012, systematics=False)
qcd = QCD(data, (ztt, others))

expr = 'tau1_numTrack_recounted:tau2_numTrack_recounted'
min_edge, max_edge = .5, 5.5
bins = int(max_edge - min_edge)
category = '2j'
region = 'OS'

ztt_sample = ztt.get_histfactory_sample(expr, category, region, bins,
        min_edge, max_edge,
        p1p3=False)
others_sample = others.get_histfactory_sample(expr, category, region, bins,
        min_edge, max_edge,
        p1p3=False)
qcd_sample = qcd.get_histfactory_sample(expr, category, region, bins,
        min_edge, max_edge,
        p1p3=False)
data_sample = data.get_histfactory_sample(expr, category, region, bins,
        min_edge, max_edge,
        p1p3=False)

ztt_sample.AddNormFactor('z_scale', 1., 0.5, 2.0)
qcd_sample.AddNormFactor('qcd_scale', 1., 0.5, 2.0)

channel = make_channel(category,
        [ztt_sample, others_sample, qcd_sample], data_sample.GetHisto())
measurement = make_measurement('trackfit', '', [channel], lumi_rel_error=0.039,
        POI=['z_scale', 'qcd_scale'])

hist2workspace = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast(measurement)
workspace = hist2workspace.MakeSingleChannelModel(
        measurement,
        channel)
#workspace = make_workspace(measurement)

obs_data = workspace.data('obsData')
pdf = workspace.pdf('model_2j')

nll = pdf.createNLL(obs_data)
minim = RooMinimizer(nll)
strategy = ROOT.Math.MinimizerOptions.DefaultStrategy()
minim.setStrategy(strategy)
tol = ROOT.Math.MinimizerOptions.DefaultTolerance()
minim.setEps(max(tol, 1.))
minim.setPrintLevel(0)
minim.optimizeConst(2)
minimizer = ROOT.Math.MinimizerOptions.DefaultMinimizerType()
algorithm = ROOT.Math.MinimizerOptions.DefaultMinimizerAlgo()
status = -1

tries = 1
maxtries = 4

while tries <= maxtries:
    status = minim.minimize(minimizer, algorithm)
    if status % 1000 == 0:
        # ignore erros from Improve
        break
    elif tries == 1:
            #Logger << kINFO << "    ----> Doing a re-scan first" << GEndl;
            minim.minimize(minimizer, "Scan")
    elif tries == 2:
        if ROOT.Math.MinimizerOptions.DefaultStrategy() == 0:
            #Logger << kINFO << "    ----> trying with strategy = 1" << GEndl;
            minim.setStrategy(1)
        else:
            tries += 1 # skip this trial if stratehy is already 1
    elif tries == 3:
        #Logger << kINFO << "    ----> trying with improve" << GEndl;
        minimizer = "Minuit";
        algorithm = "migradimproved";

fit_result = minim.save()

z_scale_arg = fit_result.floatParsFinal().find("z_scale")
z_scale = z_scale_arg.getValV()
z_scale_error = z_scale_arg.getError()

qcd_scale_arg = fit_result.floatParsFinal().find("qcd_scale")
qcd_scale = qcd_scale_arg.getValV()
qcd_scale_error = qcd_scale_arg.getError()

print z_scale, qcd_scale
