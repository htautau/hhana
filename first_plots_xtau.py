from mva.samples import MC_Ztautau, Pythia_Ztautau
from hhdb.datasets import Database
from mva.variables import VARIABLES
from mva.categories.lephad import Category_VBF_lh, Category_Boosted_lh, Category_Preselection_lh
from mva.plotting import draw_ratio

DB = Database('datasets_lh')

s1 = MC_Ztautau(
    2014, db=DB, 
    channel='lephad', 
    ntuple_path='/afs/cern.ch/user/q/qbuat/work/lephad_test_datasets', 
    student='lhskim',
    trigger=False)

s2 = Pythia_Ztautau(
    2014, db=DB, 
    channel='lephad', 
    ntuple_path='/afs/cern.ch/user/q/qbuat/work/lephad_test_datasets', 
    student='lhskim',
    trigger=False)

fields = [
    'jet_0_pt',
    # 'jet_1_pt',
    'n_avg_int',
    'met_et',
    'dilep_mmc_1_resonance_m',
    'dilep_vis_mass',
    'tau_pt',
]

vars = {}
for f in fields:
    vars[f] =  VARIABLES[f]


for cat in (Category_Preselection_lh, Category_VBF_lh, Category_Boosted_lh):
    a1, b = s1.get_field_hist(vars, cat)
    s1.draw_array(a1, cat, 'ALL', field_scale=b)
    
    a2, _ = s2.get_field_hist(vars, cat)
    s2.draw_array(a2, cat, 'ALL', field_scale=b)

    for field in a1:
        h1 = a1[field]
        h1.title = 'Sherpa ' + s1.label
        h2 = a2[field] 
        h2.title = 'Pythia ' + s2.label
        plot = draw_ratio(h1, h2, field, cat, normalize=False)
        plot.SaveAs('plots/variables/blurp_{0}_{1}.png'.format(field, cat.name))
        plot_log = draw_ratio(h1, h2, field, cat, normalize=False, logy=True)
        plot_log.SaveAs('plots/variables/blurp_{0}_{1}_logy.png'.format(field, cat.name))
