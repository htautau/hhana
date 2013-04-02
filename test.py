import numpy as np
from mva.analysis import Analysis
from mva.categories import Category_VBF, Category_Boosted, Category_Preselection


a = Analysis(2012, Category_VBF, systematics=True)


def test_cache():

    for i in xrange(10):
        rec = a.data.records(Category_VBF, 'OS', ['tau1_pt'])
        rec[0]["weight"] *= 5
        print rec

    for i in xrange(10):
        rec = a.data.records(Category_VBF, 'OS', ['tau2_pt'])
        print rec

    for i in xrange(10):
        rec = a.data.records(Category_Boosted, 'OS', ['tau2_pt'])
        print rec

    print a.data.partitioned_records(Category_VBF, 'OS', ['tau1_pt'], num_partitions=3)


def test_systematics():
    z = a.ztautau

    print z.array(Category_VBF, 'OS', ['tau1_pt', 'tau1_charge'], include_weight=False)

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


def test_pileup():

    def show(sample):
        print sample.name
        weights = sample.array(Category_Preselection, 'OS', ['pileup_weight'],
                include_weight=False)
        print weights.mean()
        print len(weights[weights == 0]) / float(len(weights))

        mu = sample.array(Category_Preselection, 'OS', ['averageIntPerXing'],
                include_weight=False, cuts='pileup_weight==0')
        print np.unique(mu)

    show(a.ztautau)
    show(a.higgs_125)

test_pileup()
