from mva.samples import Data, MC_Ztautau
from mva.categories import Category_VBF

#a = Data(2012)
#print a.records(Category_VBF, 'OS', ['tau1_pt'])
#print a.partitioned_records(Category_VBF, 'OS', ['tau1_pt'], num_partitions=3)

z = MC_Ztautau(2012, systematics=True)

nom = z.merged_records(Category_VBF, 'OS', ['tau1_pt'], include_weight=False)
up = z.merged_records(Category_VBF, 'OS', ['tau1_pt'], include_weight=False,
        systematic=('TES_UP',))
down = z.merged_records(Category_VBF, 'OS', ['tau1_pt'], include_weight=False,
        systematic=('TES_DOWN',))

print nom['tau1_pt'].mean()
print up['tau1_pt'].mean()
print down['tau1_pt'].mean()

nom = z.merged_records(Category_VBF, 'OS', ['jet1_pt'], include_weight=False)
up = z.merged_records(Category_VBF, 'OS', ['jet1_pt'], include_weight=False,
        systematic=('JES_UP',))
down = z.merged_records(Category_VBF, 'OS', ['jet1_pt'], include_weight=False,
        systematic=('JES_DOWN',))

print nom['jet1_pt'].mean()
print up['jet1_pt'].mean()
print down['jet1_pt'].mean()
