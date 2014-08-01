import os
from matplotlib import rc
from matplotlib import cm


def package_path(name):
    return os.path.splitext(os.path.abspath('latex/%s.sty' % name))[0]


LATEX_PREAMBLE = '''
\usepackage[EULERGREEK]{%s}
\sansmath
''' % package_path('sansmath')

"""
LATEX_PREAMBLE = '''
\usepackage[math-style=upright]{%s}
''' % package_path('unicode-math')
"""

#plt.rcParams['ps.useafm'] = True
#rc('text', usetex=True)
#rc('font', family='sans-serif')
rc('text.latex', preamble=LATEX_PREAMBLE)
#plt.rcParams['pdf.fonttype'] = 42


def set_colors(hists, colors='jet'):
    if isinstance(colors, basestring):
        colors = cm.get_cmap(colors, len(hists))
    if hasattr(colors, '__call__'):
        for i, h in enumerate(hists):
            color = colors((i + 1) / float(len(hists) + 1))
            h.SetColor(color)
    else:
        for h, color in izip(hists, colors):
            h.SetColor(color)


def format_legend(l):
    #frame = l.get_frame()
    #frame.set_alpha(.8)
    #frame.set_fill(False) # eps does not support alpha values
    #frame.set_linewidth(0)
    for t in l.get_texts():
        # left align all contents
        t.set_ha('left')
    l.get_title().set_ha("left")


def root_axes(ax,
              xtick_formatter=None,
              xtick_locator=None,
              xtick_rotation=None,
              logy=False, integer=False, no_xlabels=False,
              vscale=1.,
              bottom=None):
    #ax.patch.set_linewidth(2)
    if integer:
        ax.xaxis.set_major_locator(
            xtick_locator or MultipleLocator(1))
        ax.tick_params(axis='x', which='minor',
                       bottom='off', top='off')
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    if not logy:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    if no_xlabels:
        ax.xaxis.set_major_formatter(NullFormatter())
    elif xtick_formatter:
        ax.xaxis.set_major_formatter(xtick_formatter)

    if xtick_rotation is not None:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=xtick_rotation)

    ax.yaxis.set_label_coords(-0.13, 1.)
    ax.xaxis.set_label_coords(1., -0.15 / vscale)
