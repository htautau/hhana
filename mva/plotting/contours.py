import ROOT
import numpy as np
from rootpy import asrootpy
from rootpy.plotting.base import linestyles_text2root
from rootpy.plotting.base import convert_color
from math import atan, pi, copysign
from itertools import cycle


def draw_contours(hist, n_contours=3, contours=None,
                  linecolors=None, linestyles=None, linewidths=None,
                  labelcontours=True, labelcolors=None,
                  labelsizes=None, labelformats=None, same=False,
                  min_points=5):
    if contours is None:
        contours = np.linspace(hist.min(), hist.max(), n_contours + 1,
                               endpoint=False)[1:]
    hist = hist.Clone()
    hist.SetContour(len(contours), np.asarray(contours, dtype=float))
    graphs = []
    levels = []
    with invisible_canvas() as c:
        hist.Draw('CONT LIST')
        c.Update()
        conts = asrootpy(ROOT.gROOT.GetListOfSpecials().FindObject('contours'))
        for i, cont in enumerate(conts):
            for curve in cont:
                if len(curve) < min_points:
                    continue
                graphs.append(curve.Clone())
                levels.append(contours[i])
    if not same:
        axes = hist.Clone()
        axes.Draw('AXIS')
    if linecolors is None:
        linecolors = ['black']
    elif not isinstance(linecolors, list):
        linecolors = [linecolors]
    if linestyles is None:
        linestyles = linestyles_text2root.keys()
    elif not isinstance(linestyles, list):
        linestyles = [linestyles]
    if linewidths is None:
        linewidths = [1]
    elif not isinstance(linewidths, list):
        linewidths = [linewidths]
    if labelsizes is not None:
        if not isinstance(labelsizes, list):
            labelsizes = [labelsizes]
        labelsizes = cycle(labelsizes)
    if labelcolors is not None:
        if not isinstance(labelcolors, list):
            labelcolors = [labelcolors]
        labelcolors = cycle(labelcolors)
    if labelformats is None:
        labelformats = ['%0.2g']
    elif not isinstance(labelformats, list):
        labelformats = [labelformats]
    linecolors = cycle(linecolors)
    linestyles = cycle(linestyles)
    linewidths = cycle(linewidths)
    labelformats = cycle(labelformats)
    label = ROOT.TLatex()
    xmin, xmax = hist.bounds(axis=0)
    ymin, ymax = hist.bounds(axis=1)
    stepx = (xmax - xmin) / 100
    stepy = (ymax - ymin) / 100
    for level, graph in zip(levels, graphs):
        graph.linecolor = linecolors.next()
        graph.linewidth = linewidths.next()
        graph.linestyle = linestyles.next()
        graph.Draw('C')
        if labelcontours:
            if labelsizes is not None:
                label.SetTextSize(labelsizes.next())
            if labelcolors is not None:
                label.SetTextColor(convert_color(labelcolors.next(), 'ROOT'))
            point_idx = len(graph) / 2
            point = graph[point_idx]
            padx, pady = 0, 0
            if len(graph) > 5:
                # use derivative to get text angle
                x1, y1 = graph[point_idx - 2]
                x2, y2 = graph[point_idx + 2]
                dx = (x2 - x1) / stepx
                dy = (y2 - y1) / stepy
                if dx == 0:
                    label.SetTextAngle(0)
                    label.SetTextAlign(12)
                else:
                    padx = copysign(1, -dy) * stepx
                    pady = copysign(1, dx) * stepy
                    if pady < 0:
                        align = 23
                    else:
                        align = 21
                    angle = atan(dy / dx) * 180 / pi
                    label.SetTextAngle(angle)
                    label.SetTextAlign(align)
            else:
                label.SetTextAngle(0)
                label.SetTextAlign(21)
            label.DrawLatex(point[0] + padx, point[1] + pady,
                            (labelformats.next()) % level)
