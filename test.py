from mva.analysis import Analysis
from mva.categories import Category_VBF

#a = Data(2012)
#print a.records(Category_VBF, 'OS', ['tau1_pt'])
#print a.partitioned_records(Category_VBF, 'OS', ['tau1_pt'], num_partitions=3)

a = Analysis(2012, Category_VBF, systematics=True)

z = a.ztautau

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

nom = z.merged_records(Category_VBF, 'OS', ['tau1_pt'])
up = z.merged_records(Category_VBF, 'OS', ['tau1_pt'],
        systematic=('TAUID_UP',))
down = z.merged_records(Category_VBF, 'OS', ['tau1_pt'],
        systematic=('TAUID_DOWN',))

print nom['weight'].mean()
print up['weight'].mean()
print down['weight'].mean()

nom = z.merged_records(Category_VBF, 'OS', ['tau1_pt'])
up = z.merged_records(Category_VBF, 'OS', ['tau1_pt'],
        systematic=('TRIGGER_UP',))
down = z.merged_records(Category_VBF, 'OS', ['tau1_pt'],
        systematic=('TRIGGER_DOWN',))

print nom['weight'].mean()
print up['weight'].mean()
print down['weight'].mean()

nom = z.merged_records(Category_VBF, 'OS', ['tau1_pt'])
up = z.merged_records(Category_VBF, 'OS', ['tau1_pt'],
        systematic=('ZFIT_UP',))
down = z.merged_records(Category_VBF, 'OS', ['tau1_pt'],
        systematic=('ZFIT_DOWN',))

print nom['weight'].mean()
print up['weight'].mean()
print down['weight'].mean()

