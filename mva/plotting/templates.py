import ROOT

from rootpy.context import preserve_current_style
from rootpy.plotting import Canvas, Pad, Hist
from rootpy.plotting.utils import draw, tick_length_pixels
from rootpy.plotting.shapes import Line


__all__ = [
    'SimplePlot',
    'RatioPlot',
]


class SimplePlot(Canvas):

    def __init__(self, width=None, height=None,
                 xtitle=None, ytitle=None,
                 tick_length=15,
                 logy=False):

        style = ROOT.gStyle

        # plot dimensions in pixels
        if height is None:
            height = style.GetCanvasDefH()
        if width is None:
            width = style.GetCanvasDefW()

        # margins
        left_margin = style.GetPadLeftMargin()
        bottom_margin = style.GetPadBottomMargin()
        top_margin = style.GetPadTopMargin()
        right_margin = style.GetPadRightMargin()

        super(SimplePlot, self).__init__(width=width, height=height)
        self.SetMargin(0, 0, 0, 0)

        # top pad for histograms
        with self:
            main = Pad(0., 0., 1., 1.)
            if logy:
                main.SetLogy()
            main.SetBottomMargin(bottom_margin)
            main.SetTopMargin(top_margin)
            main.SetLeftMargin(left_margin)
            main.SetRightMargin(right_margin)
            main.Draw()

        # draw axes
        with main:
            main_hist = Hist(1, 0, 1)
            main_hist.Draw('AXIS')

        if xtitle is not None:
            main_hist.xaxis.title = xtitle
        if ytitle is not None:
            main_hist.yaxis.title = ytitle

        # set the tick lengths
        tick_length_pixels(main, main_hist.xaxis, main_hist.yaxis,
                           tick_length)

        self.main = main
        self.main_hist = main_hist
        self.logy = logy

    def pad(self, region):
        if region == 'main':
            return self.main
        raise ValueError("SimplePlot region {0} does not exist".format(region))

    def cd(self, region=None):
        if region is not None:
            self.pad(region).cd()
        else:
            super(SimplePlot, self).cd()

    def axes(self, region):
        if region == 'main':
            return self.main_hist.xaxis, self.main_hist.yaxis
        raise ValueError("SimplePlot region {0} does not exist".format(region))

    def draw(self, region, objects, **kwargs):
        pad = self.pad(region)
        x, y = self.axes(region)
        if self.logy:
            kwargs['logy'] = True
        _, bounds = draw(objects, pad=pad,
                         xaxis=x, yaxis=y,
                         same=True, **kwargs)
        return bounds


class RatioPlot(Canvas):

    def __init__(self, width=None, height=None,
                 ratio_height=0.2, ratio_margin=0.05,
                 ratio_range=(0, 2), ratio_divisions=4,
                 ratio_line_values=(1,),
                 ratio_line_width=2,
                 ratio_line_style='dashed',
                 xtitle=None, ytitle=None, ratio_title=None,
                 tick_length=15,
                 logy=False):

        style = ROOT.gStyle

        # plot dimensions in pixels
        if height is not None:
            figheight = baseheight = height
        else:
            figheight = baseheight = style.GetCanvasDefH()
        if width is not None:
            figwidth = basewidth = width
        else:
            figwidth = basewidth = style.GetCanvasDefW()

        # margins
        left_margin = style.GetPadLeftMargin()
        bottom_margin = style.GetPadBottomMargin()
        top_margin = style.GetPadTopMargin()
        right_margin = style.GetPadRightMargin()

        figheight += (ratio_height + ratio_margin) * figheight
        ratio_height += bottom_margin + ratio_margin / 2.

        super(RatioPlot, self).__init__(
            width=int(figwidth), height=int(figheight))
        self.SetMargin(0, 0, 0, 0)

        # top pad for histograms
        with self:
            main = Pad(0., ratio_height, 1., 1.)
            if logy:
                main.SetLogy()
            main.SetBottomMargin(ratio_margin / 2.)
            main.SetTopMargin(top_margin)
            main.SetLeftMargin(left_margin)
            main.SetRightMargin(right_margin)
            main.Draw()

        # bottom pad for ratio plot
        with self:
            ratio = Pad(0, 0, 1, ratio_height)
            ratio.SetBottomMargin(bottom_margin / ratio_height)
            ratio.SetTopMargin(ratio_margin / (2. * ratio_height))
            ratio.SetLeftMargin(left_margin)
            ratio.SetRightMargin(right_margin)
            ratio.Draw()

        # draw main axes
        with main:
            main_hist = Hist(1, 0, 1)
            main_hist.Draw('AXIS')

        # hide x-axis labels and title on main pad
        xaxis, yaxis = main_hist.xaxis, main_hist.yaxis
        xaxis.SetLabelOffset(1000)
        xaxis.SetTitleOffset(1000)
        # adjust y-axis title spacing
        yaxis.SetTitleOffset(
            yaxis.GetTitleOffset() * figheight / baseheight)

        # draw ratio axes
        with ratio:
            ratio_hist = Hist(1, 0, 1)
            ratio_hist.Draw('AXIS')

        # adjust x-axis label and title spacing
        xaxis, yaxis = ratio_hist.xaxis, ratio_hist.yaxis
        xaxis.SetLabelOffset(
            xaxis.GetLabelOffset() / ratio_height)
        xaxis.SetTitleOffset(
            xaxis.GetTitleOffset() / ratio_height)
        # adjust y-axis title spacing
        yaxis.SetTitleOffset(
            yaxis.GetTitleOffset() * figheight / baseheight)

        if ratio_range is not None:
            yaxis.SetLimits(*ratio_range)
            yaxis.SetRangeUser(*ratio_range)
            yaxis.SetNdivisions(ratio_divisions)

        if xtitle is not None:
            ratio_hist.xaxis.title = xtitle
        if ytitle is not None:
            main_hist.yaxis.title = ytitle
        if ratio_title is not None:
            ratio_hist.yaxis.title = ratio_title

        # set the tick lengths
        tick_length_pixels(main, main_hist.xaxis, main_hist.yaxis,
                           tick_length)
        tick_length_pixels(ratio, ratio_hist.xaxis, ratio_hist.yaxis,
                           tick_length)

        # draw ratio lines
        lines = []
        if ratio_line_values:
            with ratio:
                for value in ratio_line_values:
                    line = Line(0, value, 1, value)
                    line.linestyle = ratio_line_style
                    line.linewidth = ratio_line_width
                    line.Draw()
                    lines.append(line)
        self.lines = lines

        self.main = main
        self.main_hist = main_hist
        self.ratio = ratio
        self.ratio_hist = ratio_hist
        self.ratio_range = ratio_range
        self.logy = logy

    def pad(self, region):
        if region == 'main':
            return self.main
        elif region == 'ratio':
            return self.ratio
        raise ValueError("RatioPlot region {0} does not exist".format(region))

    def cd(self, region=None):
        if region is not None:
            self.pad(region).cd()
        else:
            super(RatioPlot, self).cd()

    def axes(self, region):
        if region == 'main':
            return self.main_hist.xaxis, self.main_hist.yaxis
        elif region == 'ratio':
            return self.ratio_hist.xaxis, self.ratio_hist.yaxis
        raise ValueError("RatioPlot region {0} does not exist".format(region))

    def update_lines(self):
        x, y = self.axes('ratio')
        # update ratio line lengths
        for line in self.lines:
            line.SetX1(x.GetXmin())
            line.SetX2(x.GetXmax())

    def draw(self, region, objects, **kwargs):
        pad = self.pad(region)
        x, y = self.axes(region)
        if region == 'ratio' and self.ratio_range is not None:
            y = None
        if region == 'main' and self.logy:
            kwargs['logy'] = True
        _, bounds = draw(objects, pad=pad,
                         xaxis=x, yaxis=y,
                         same=True, **kwargs)
        if region == 'ratio':
            self.update_lines()
        return bounds
