#!/usr/bin/env python

# create an ntuple containing trees for data with classifier scores
# and event weights

from mva.cmd import get_parser

args = get_parser(actions=False).parse_args()

from mva.analysis import get_analysis

analysis = get_analysis(args)

from rootpy.plotting import Canvas, Hist, Legend, Latex
from rootpy.plotting.utils import draw
from root_numpy import fill_hist

from mva.samples import Embedded_Ztautau, MC_Ztautau

target_region = args.target_region

eb_ztt = Embedded_Ztautau(year=args.year, systematics=False, title='Embedded Ztautau',
    fillstyle='hollow', color='black')
# emulate embedding treatment on MC Ztautau
mc_ztt = MC_Ztautau(year=args.year, systematics=False, trigger=False, title='Alpgen Ztautau',
    fillstyle='hollow', color='red', linestyle='dashed')

analysis.ztautau = eb_ztt

for category in analysis.iter_categories(
        args.categories, args.controls, names=args.category_names):

    if category.analysis_control:
        continue

    clf = analysis.get_clf(category, load=True)

    eb_ztt_events = eb_ztt.events(category, target_region)[1].value
    mc_ztt_events = mc_ztt.events(category, target_region)[1].value

    canvas = Canvas()
    hists = []
    for sample in (eb_ztt, mc_ztt):
        # get the scores
        scores, weights = sample.scores(
            clf, category, target_region,
            systematics=False)['NOMINAL']
        hist = Hist(25, -1, 1, drawstyle='hist E0', markersize=0, linewidth=3,
            legendstyle='L', **sample.hist_decor)
        fill_hist(hist, scores, weights)
        if sample is mc_ztt:
            hist *= eb_ztt_events / mc_ztt_events
        hists.append(hist)
    draw(hists, pad=canvas, xtitle='BDT Score', ytitle='Events', ypadding=(0.25, 0))
    leg = Legend(hists, pad=canvas, leftmargin=0.02, margin=0.1, topmargin=0.06)
    leg.Draw()
    label = Latex(canvas.leftmargin + 0.02, 0.9, category.name + ' ' + str(analysis.data.info), coord='NDC')
    label.SetTextSize(20)
    label.Draw()
    canvas.SaveAs('emb_BDT_{0}_{1}.png'.format(category.name, analysis.year))

    canvas = Canvas()
    hists = []
    for sample in (eb_ztt, mc_ztt):
        hist = Hist(25, 0, 250, drawstyle='hist E0', markersize=0, linewidth=3,
            legendstyle='L', **sample.hist_decor)
        sample.draw_array({'mmc1_mass': hist}, category, target_region)
        if sample is mc_ztt:
            hist *= eb_ztt_events / mc_ztt_events
        hists.append(hist)
    draw(hists, pad=canvas, xtitle='MMC [GeV]', ytitle='Events', ypadding=(0.25, 0))
    leg = Legend(hists, pad=canvas, leftmargin=0.02, margin=0.1, topmargin=0.06)
    leg.Draw()
    label = Latex(canvas.leftmargin + 0.02, 0.9, category.name + ' ' + str(analysis.data.info), coord='NDC')
    label.SetTextSize(20)
    label.Draw()
    canvas.SaveAs('emb_MMC_{0}_{1}.png'.format(category.name, analysis.year))
