from rootpy.plotting.style import get_style, set_style
import ROOT


def set_hsg4_style(shape='square'):
    style = get_style('ATLAS', shape=shape)
    #style.SetFrameLineWidth(2)
    #style.SetLineWidth(2)
    #style.SetTitleYOffset(1.8)
    #style.SetTickLength(0.04, 'X')
    #style.SetTickLength(0.02, 'Y')

    # custom HSG4 modifications
    # style.SetPadTopMargin(0.06)
    style.SetPadLeftMargin(0.16)
    style.SetTitleYOffset(1.6)
    style.SetHistTopMargin(0.)
    style.SetHatchesLineWidth(1)
    style.SetHatchesSpacing(1)
    ROOT.TGaxis.SetMaxDigits(4)
    set_style(style)
