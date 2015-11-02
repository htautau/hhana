import ROOT
from mva.samples import Pythia_Ztautau, Data
from mva.variables import VARIABLES
from mva.categories.hadhad import Category_Preselection
from mva.plotting import draw_ratio, draw
from tabulate import tabulate
from mva.samples.fakes import OS_SS
from mva.regions import OS, SS

ztautau = Pythia_Ztautau(
    2015,
    color='#00A3FF')

z_os_ss = OS_SS(ztautau, OS, SS)

data = Data(
    2015,
    label='Data 2015')


fields = [
    'jet_0_pt',
    'jet_1_pt',
    'n_avg_int',
    'met_et',
    'ditau_vis_mass',
]

vars = {}
for f in fields:
    vars[f] =  VARIABLES[f]

categories = [Category_Preselection]
headers = [c.name for c in categories]
headers.insert(0, 'sample / category')
# categories = [Category_VBF_lh]
table = []

# for sample in (ztautau, data):
#     row = [sample.name]
#     table.append(row)
#     for category in categories:
#         events = sample.events(category, region='OS_ISOL')
#         row.append(
#             "{0:.1f} +/- {1:.1f}".format(
#                 events[1].value, events[1].error))
       
    
# print tabulate(table, headers=headers)
# print


for cat in categories:
    a1, b = data.get_field_hist(vars, cat, )

    # data.draw_array(a1, cat, 'OS_ISOL', field_scale=b)
    
    z_h, _ = z_os_ss.get_field_hist(vars, cat)
    z_os_ss.draw_array(z_h, cat, 'OS', field_scale=b)
    for field in z_h:
        print list(z_h[field].y())
    # ztautau.draw_array(z_h, cat, 'OS_ISOL', field_scale=b)


#     for field in a1:
#         # d = a1[field]
#         draw(
#             vars[field]['root'],
#             cat,
#            # data=a1[field],
#             data=None if a1[field].Integral() == 0 else a1[field],
#             model=[z_h[field]],
#             units=vars[field]['units'] if 'units' in vars[field] else None, 
#             logy=False,
#             output_name='{0}_{1}.png'.format(field, cat.name))

#         #print list(a1[field].y())
#         #print a1[field].Integral()
#         # HACK: clear the list of canvases
#         ROOT.gROOT.GetListOfCanvases().Clear()
