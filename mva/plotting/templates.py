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

        super(SimplePlot, self).__init__(width=width, height=height)
        left, right, bottom, top = self.margin
        self.SetMargin(0, 0, 0, 0)

        # top pad for histograms
        with self:
            main = Pad(0., 0., 1., 1.)
            if logy:
                main.SetLogy()
            main.margin = (left, right, bottom, top)
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
                 offset=0,
                 ratio_height=None, ratio_margin=26,
                 ratio_limits=(0, 2), ratio_divisions=4,
                 prune_ratio_ticks=False,
                 ratio_line_values=(1,),
                 ratio_line_width=2,
                 ratio_line_style='dashed',
                 xtitle=None, ytitle=None, ratio_title=None,
                 tick_length=15,
                 logy=False):

        # first init as normal canvas
        super(RatioPlot, self).__init__(width=width, height=height)

        # get margins in pixels
        left, right, bottom, top = self.margin_pixels
        default_height = self.height
        default_frame_height = default_height - bottom - top

        if ratio_height is None:
            ratio_height = default_height / 4.

        self.height += ratio_height + ratio_margin + offset
        self.margin = (0, 0, 0, 0)

        main_height = default_frame_height + top + ratio_margin / 2. + offset
        ratio_height += ratio_margin / 2. + bottom

        # top pad for histograms
        with self:
            main = Pad(0., ratio_height / self.height, 1., 1.)
            if logy:
                main.SetLogy()
            main.margin_pixels = (left, right, ratio_margin / 2., top)
            main.Draw()

        # bottom pad for ratio plot
        with self:
            ratio = Pad(0, 0, 1, ratio_height / self.height)
            ratio.margin_pixels = (left, right, bottom, ratio_margin / 2.)
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
            yaxis.GetTitleOffset() * self.height / default_height)

        # draw ratio axes
        with ratio:
            ratio_hist = Hist(1, 0, 1)
            ratio_hist.Draw('AXIS')

        # adjust x-axis label and title spacing
        xaxis, yaxis = ratio_hist.xaxis, ratio_hist.yaxis

        xaxis.SetLabelOffset(
            xaxis.GetLabelOffset() * self.height / ratio_height)
        xaxis.SetTitleOffset(
            xaxis.GetTitleOffset() * self.height / ratio_height)
        # adjust y-axis title spacing
        yaxis.SetTitleOffset(
            yaxis.GetTitleOffset() * self.height / default_height)

        if ratio_limits is not None:
            low, high = ratio_limits
            if prune_ratio_ticks:
                delta = 0.01 * (high - low) / float(ratio_divisions % 100)
                low += delta
                high -= delta
            yaxis.SetLimits(low, high)
            yaxis.SetRangeUser(low, high)
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
        self.ratio_limits = ratio_limits
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
        if region == 'ratio' and self.ratio_limits is not None:
            y = None
        if region == 'main' and self.logy:
            kwargs['logy'] = True
        _, bounds = draw(objects, pad=pad,
                         xaxis=x, yaxis=y,
                         same=True, **kwargs)
        if region == 'ratio':
            self.update_lines()
        return bounds
