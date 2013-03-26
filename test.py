from mva.samples import Data, MC_Ztautau
from mva.categories import Category_VBF

a = Data(2012)
print a.records(Category_VBF, 'OS', ['tau1_pt'])
print a.partitioned_records(Category_VBF, 'OS', ['tau1_pt'], num_partitions=3)

z = MC_Ztautau(2012, systematics=True)

print z.records(Category_VBF, 'OS', ['tau1_pt'])
print z.records(Category_VBF, 'OS', ['tau1_pt'], systematic=('TES_UP',))
