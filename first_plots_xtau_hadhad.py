import ROOT
from mva.samples import Pythia_Ztautau, Data
from mva.variables import VARIABLES
from mva.categories.hadhad import Category_Preselection
from mva.plotting import draw_ratio, draw
from tabulate import tabulate


ztautau = Pythia_Ztautau(
    2015,
    trigger=False,
    color='#00A3FF')

data = Data(
    2015,
    label='Data 2015',
    trigger=False)


fields = [
    'jet_0_pt',
    'jet_1_pt',
    'n_avg_int',
    'met_et',
    'tau_tau_vis_mass',
]

vars = {}
for f in fields:
    vars[f] =  VARIABLES[f]

categories = [Category_Preselection]
headers = [c.name for c in categories]
headers.insert(0, 'sample / category')
# categories = [Category_VBF_lh]
table = []

for sample in (ztautau, data):
    row = [sample.name]
    table.append(row)
    for category in categories:
        events = sample.events(category, region='OS_ISOL')
        row.append(
            "{0:.1f} +/- {1:.1f}".format(
                events[1].value, events[1].error))
       
    
print tabulate(table, headers=headers)
print


for cat in categories:
    a1, b = data.get_field_hist(vars, cat, )
    data.draw_array(a1, cat, 'OS_ISOL', field_scale=b)
    
    z_h, _ = ztautau.get_field_hist(vars, cat)
    ztautau.draw_array(z_h, cat, 'OS_ISOL', field_scale=b)


    for field in a1:
        # d = a1[field]
        draw(
            vars[field]['root'],
            cat,
           # data=a1[field],
            data=None if a1[field].Integral() == 0 else a1[field],
            model=[z_h[field]],
            units=vars[field]['units'] if 'units' in vars[field] else None, 
            logy=False,
            output_name='{0}_{1}.png'.format(field, cat.name))

        #print list(a1[field].y())
        #print a1[field].Integral()
        # HACK: clear the list of canvases
        ROOT.gROOT.GetListOfCanvases().Clear()
