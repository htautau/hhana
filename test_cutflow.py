from mva.samples import Higgs
import mva.regions
import mva.categories.hadhad.common

higgs = Higgs(2015, mode='gg')

cut1 = mva.regions.Q
cut2 = mva.regions.P1P3
cut3 = mva.regions.ID_MEDIUM
cut4 = mva.categories.hadhad.common.LEAD_TAU_40
cut5 = mva.categories.hadhad.common.SUBLEAD_TAU_30
cut6 = mva.regions.OS
cut7 = mva.categories.hadhad.common.MET
cut8 = mva.categories.hadhad.common.DETA_TAUS
cut9 = mva.categories.hadhad.common.DR_TAUS

a = higgs.events(weighted=False)[1].value
b = higgs.events(weighted=False, cuts = (cut1))[1].value
c = higgs.events(weighted=False, cuts = (cut1 & cut2))[1].value
d = higgs.events(weighted=False, cuts = (cut1 & cut2 & cut3))[1].value
e = higgs.events(weighted=False, cuts = (cut1 & cut2 & cut3 & cut4))[1].value
f = higgs.events(weighted=False, cuts = (cut1 & cut2 & cut3 & cut4 & cut5))[1].value
g = higgs.events(weighted=False, cuts = (cut1 & cut2 & cut3 & cut4 & cut5 & cut6))[1].value
h = higgs.events(weighted=False, cuts = (cut1 & cut2 & cut3 & cut4 & cut5 & cut6 & cut7))[1].value
i = higgs.events(weighted=False, cuts = (cut1 & cut2 & cut3 & cut4 & cut5 & cut6 & cut7 & cut8))[1].value
j = higgs.events(weighted=False, cuts = (cut1 & cut2 & cut3 & cut4 & cut5 & cut6 & cut7 & cut8 & cut9))[1].value

print 'No selection: ', a
print '      Charge: ', b
print '     nTracks: ', c
print '       tauID: ', d
print '     Leading: ', e
print '  Subleading: ', f
print '          OS: ', g
print '         MET: ', h
print '        Deta: ', i
print '          Dr: ', j