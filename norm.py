
from mva.samples import Data, MC_Ztautau, Others, QCD


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
print ztt_sample
