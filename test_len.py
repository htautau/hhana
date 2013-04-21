from mva.samples import Embedded_Ztautau, MC_Ztautau
from mva.categories import Category_Preselection

samples = [
        Embedded_Ztautau(2012),
        MC_Ztautau(2012)
]

for a in samples:
    print a.merged_records(Category_Preselection, 'OS').shape
    print a.events(Category_Preselection, 'OS', raw=True)
