from mva.cmd import get_parser

args = get_parser(actions=False).parse_args()

from mva.categories import Category_VBF, Category_Preselection, Category_Boosted
from mva.analysis import get_analysis
from mva.plotting import draw_channel_array
from mva.variables import VARIABLES
from rootpy.tree import Cut
from math import pi


for year in (2011, 2012):
    analysis = get_analysis(
        args, year=year,
        systematics=False,
        qcd_shape_region='nOS_NONISOL')
    analysis.normalize(Category_Preselection)

    # show justification of the dEta cut location in VBF
    draw_channel_array(
        analysis, {'dEta_jets': VARIABLES['dEta_jets']},
        mass=125, mode=['gg', 'VBF'], signal_scale=100,
        stack_signal=False,
        signal_colors=['blue', 'red'],
        signal_linestyles=['dashed', 'solid'],
        category=Category_VBF, region='OS', show_ratio=False,
        legend_leftmargin=0.28,
        output_dir='plots/categories',
        output_suffix='_{0}'.format(year % 1000),
        output_formats=['png', 'eps'],
        arrow_values=[2.])

    # show justification of the MET centrality cut at preselection and boosted
    for category, cut in ((Category_Preselection, pi / 4), (Category_Boosted, pi / 6)):
        draw_channel_array(
            analysis, {'MET_bisecting': VARIABLES['MET_bisecting']},
            mass=125, mode='combined', signal_scale=50,
            stack_signal=True,
            separate_legends=True,
            category=category, region='OS', show_ratio=False,
            output_dir='plots/categories',
            output_suffix='_{0}'.format(year % 1000),
            output_formats=['png', 'eps'],
            ypadding=(0.55, 0))

        draw_channel_array(
            analysis, {'dPhi_min_tau_MET': VARIABLES['dPhi_min_tau_MET']},
            mass=125, mode='combined', signal_scale=50,
            stack_signal=True,
            separate_legends=True,
            category=category, region='OS', show_ratio=False,
            output_dir='plots/categories',
            output_suffix='_{0}'.format(year % 1000),
            output_formats=['png', 'eps'],
            cuts=Cut('!MET_bisecting'),
            arrow_values=[cut])
