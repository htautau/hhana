import ROOT
from mva.samples import MC_Ztautau, Pythia_Ztautau, Data
from mva.samples.others import EWK, Top
from hhdb.datasets import Database
from mva.variables import VARIABLES
from mva.categories.lephad import Category_VBF_lh, Category_Boosted_lh, Category_Preselection_lh
from mva.plotting import draw_ratio, draw

# Instantiate and load the database
DB = Database('datasets_lh')

# Ntuples path
NTUPLE_PATH = '/afs/cern.ch/user/q/qbuat/work/public/xtau_output/lephad/v1_1'


ztautau = Pythia_Ztautau(
    2015, db=DB, 
    channel='lephad', 
    ntuple_path=NTUPLE_PATH, 
    student='lhskim',
    trigger=False,
    color='#00A3FF')

top = Top(
    2015, db=DB, 
    channel='lephad', 
    ntuple_path=NTUPLE_PATH, 
    student='lhskim',
    trigger=False,
    color='lightskyblue')

ewk = EWK(
    2015, db=DB, 
    channel='lephad', 
    ntuple_path=NTUPLE_PATH, 
    student='lhskim',
    trigger=False,
    color='#8A0F0F')

data = Data(
    2015,
    ntuple_path=NTUPLE_PATH, 
    student='lhskim',
    label='Data 2015',
    trigger=False)


fields = [
    'jet_0_pt',
    'jet_1_pt',
    'n_avg_int',
    'met_et',
    'ditau_vis_mass',
    'ditau_coll_approx_m',
    'ditau_mmc_mlm_m',
    'tau_pt',
    'lep_pt',
]

vars = {}
for f in fields:
    vars[f] =  VARIABLES[f]

categories = [Category_Preselection_lh, Category_Boosted_lh]

# categories = [Category_VBF_lh]
for cat in categories:
    a1, b = data.get_field_hist(vars, cat)
    data.draw_array(a1, cat, 'ALL', field_scale=b)
    
    z_h, _ = ztautau.get_field_hist(vars, cat)
    ztautau.draw_array(z_h, cat, 'ALL', field_scale=b)

    t_h, _ = top.get_field_hist(vars, cat)
    top.draw_array(t_h, cat, 'ALL', field_scale=b)

    ewk_h, _ = ewk.get_field_hist(vars, cat)
    ewk.draw_array(ewk_h, cat, 'ALL', field_scale=b)


    for field in a1:
        # d = a1[field]
        draw(
            vars[field]['root'],
            cat,
            data=a1[field],
            model=[t_h[field], ewk_h[field], z_h[field]],
            units=vars[field]['units'] if 'units' in vars[field] else None, 
            logy=True,
            output_name='toto_{0}_{1}.png'.format(field, cat.name))
        # HACK: clear the list of canvases
        ROOT.gROOT.GetListOfCanvases().Clear()
