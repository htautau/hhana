#!/usr/bin/env python

# create an ntuple containing trees for data with classifier scores
# and event weights

from mva.cmd import get_parser

args = get_parser(actions=False).parse_args()

from mva.analysis import get_analysis

analysis = get_analysis(args)

from rootpy.plotting import Canvas, Hist
from rootpy.plotting.utils import draw
from root_numpy import fill_hist

target_region = args.target_region
signal = analysis.higgs_125
others = analysis.others

for category in analysis.iter_categories(
        args.categories, args.controls, names=args.category_names):

    if category.analysis_control:
        continue

    clf = analysis.get_clf(category, load=True)

    for sample, name in ((signal, 'signal'), (others, 'others')):
        # get the scores
        scores, weights = sample.scores(
            clf, category, target_region,
            systematics=False)['NOMINAL']

        canvas = Canvas()
        hist = Hist(50, -1, 1, drawstyle='hist', linestyle='dashed', linewidth=3)
        fill_hist(hist, scores, weights)
        draw(hist, pad=canvas, xtitle='BDT Score', ytitle='Events')
        canvas.SaveAs('trigger_sf_{0}_BDT_{1}.png'.format(name, category.name))

        canvas = Canvas()
        hist = Hist(25, 0, 250, drawstyle='hist', linestyle='dashed', linewidth=3)
        sample.draw_array({'mmc1_mass': hist}, category, target_region)
        draw(hist, pad=canvas, xtitle='MMC [GeV]', ytitle='Events')
        canvas.SaveAs('trigger_sf_{0}_MMC_{1}.png'.format(name, category.name))
