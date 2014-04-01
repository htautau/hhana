from mva.cmd import get_parser

args = get_parser(actions=False).parse_args()

from mva.categories import Category_VBF
from mva.analysis import get_analysis
from mva.plotting import draw_channel_array
from mva.variables import VARIABLES

analysis = get_analysis(args)
analysis.normalize(Category_VBF)

draw_channel_array(analysis, {'dEta_jets': VARIABLES['dEta_jets']},
                   mass=125, mode=['gg', 'VBF'], signal_scale=100,
                   stack_signal=False,
                   signal_colors=['blue', 'red'],
                   signal_linestyles=['dashed', 'solid'],
                   category=Category_VBF, region='OS', show_ratio=False,
                   legend_leftmargin=0.28,
                   output_dir='plots/categories')
