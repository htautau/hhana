import rootpy
from rootpy.extern import argparse

from .categories import CATEGORIES, CONTROLS
from .massregions import DEFAULT_LOW_MASS, DEFAULT_HIGH_MASS
from .variables import VARIABLES


class formatter_class(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawTextHelpFormatter):
    pass


def base_parser():

    return argparse.ArgumentParser(formatter_class=formatter_class)


def general_parser(parser=None):

    if parser is None:
        parser = base_parser()
    """
    General Options
    """
    parser.add_argument('--actions', nargs='*',
            choices=['validate', 'plot', 'train', 'evaluate', 'limits'],
            default=[],
            help='only perform these actions')
    parser.add_argument('--no-systematics', action='store_false',
            dest='systematics',
            help="turn off systematics",
            default=True)
    parser.add_argument('--categories', default='harmonize',
            choices=CATEGORIES.keys(),
            help='category definitions')
    parser.add_argument('--category-names', nargs="+", default=None,
            help='category names')
    """
    parser.add_argument('--controls', nargs='*', default=CONTROLS.keys(),
            help='which controls to draw plots in')
    parser.add_argument('--only-controls', action='store_true', default=False,
            help='only draw control plots. no category plots.')
    parser.add_argument('--enable-controls', action='store_true', default=False,
            help='plot in controls')
    """
    parser.add_argument('--unblind', action='store_true', default=False,
            help='plot the data in the signal region of the classifier output')
    parser.add_argument('--embedding', action='store_true', default=False,
            help='use embedding instead of ALPGEN')
    parser.add_argument('--year', type=int, default=2012, choices=(2011, 2012),
            help='the year')

    return parser


def mass_parser(parser=None):

    if parser is None:
        parser = base_parser()
    """
    Mass Regions Options
    """
    parser.add_argument('--low-mass-cut', type=int,
            default=DEFAULT_LOW_MASS,
            help='the low mass window cut. '
            'Norms of Z and QCD are fit below this and '
            'the signal region of the classifier output is above this')
    parser.add_argument('--high-mass-cut', type=int,
            default=DEFAULT_HIGH_MASS,
            help='the high mass window cut. '
            'Norms of Z and QCD are fit above this and '
            'the signal region of the classifier output is below this')
    parser.add_argument('--full-signal-region', action='store_true',
            default=False, help="use full mass range as signal region")
    parser.add_argument('--no-sideband-in-control',
            dest='high_sideband_in_control',
            action='store_false',
            default=True,
            help='Exclude the high mass sideband in the mass control and include '
            'it in the signal region')

    return parser


def fitting_parser(parser=None):

    if parser is None:
        parser = base_parser()
    """
    Fitting Options
    """
    parser.add_argument('--refit',
            action='store_false', dest='use_fit_cache',
            help="do not use cached background scale factors "
            "and instead recalculate them",
            default=True)
    parser.add_argument('--fit-param', choices=('bdt', 'track', 'track1d'),
            default='track',
            help='parameters used to determine normalization of QCD and Z')
    parser.add_argument('--draw-fit', action='store_true', default=False,
            help='draw the QCD/Z norm fit results')

    return parser


def training_parser(parser=None):

    if parser is None:
        parser = base_parser()
    """
    Training Options
    """
    parser.add_argument('--retrain',
            action='store_false', dest='use_clf_cache',
            help="do not use cached classifier "
            "and instead train a new one",
            default=True)
    parser.add_argument('--nfold', type=int, default=5,
            help='the number of folds in the cross-validation')
    parser.add_argument('--clf-bins', dest='bins', type=int, default=10,
            help='the number of bins to use in the plots of '
            'the classifier output')
    parser.add_argument('--train-fraction', type=float, default=.5,
            help='the fraction of events used for training and excluded from the '
            'final limit histograms')
    parser.add_argument('--train-categories', nargs='*', default=[],
            help='only train in these categories')
    parser.add_argument('--quick-train', action='store_true', default=False,
            help='perform a very small grid search for testing purposes')
    parser.add_argument('--quick-eval', action='store_true', default=False,
            help='do not make expensize validation plots')
    parser.add_argument('--grid-search', action='store_true', default=False,
            help='perform a grid-searched cross validation')
    parser.add_argument('--forest-feature-ranking',
            action='store_true', default=False,
            help='Use a random forest to perform a feature ranking.')
    parser.add_argument('--correlations', action='store_true', default=False,
            help='draw correlation plots')
    parser.add_argument('--ranking', action='store_true', default=False,
            help='only show the variable rankings')

    return parser


def plotting_parser(parser=None):

    if parser is None:
        parser = base_parser()
    """
    Plotting Options
    """
    parser.add_argument('--plots', nargs='*',
            help='only draw these plots. see the keys in variables.py')
    parser.add_argument('--plot-cut', default=None, nargs='?',
            help='extra cut to be applied on the plots, but excluded from the '
            'QCD/Z normaliation and training and classifier output')
    parser.add_argument('--plot-expr', default=None, nargs='?',
            help='expression to plot, instead of predefined ones in variables.py')
    parser.add_argument('--plot-name', default=None, nargs='?',
            help='name of expr')
    parser.add_argument('--plot-min', type=float, default=0, nargs='?',
            help='minimum of expr')
    parser.add_argument('--plot-max', type=float, default=1, nargs='?',
            help='maximum of expr')
    parser.add_argument('--plot-bins', type=int, default=20, nargs='?',
            help='number of bins to plot expr in')
    parser.add_argument('--root', action='store_true', default=False,
            help='draw plots with ROOT. default is matplotlib')
    parser.add_argument('--suffix', default=None, nargs='?',
            help='suffix to add to any output files or plots')
    parser.add_argument('--output-formats', default=['png'], nargs='+',
            choices=('png', 'eps', 'pdf'),
            help='output formats')

    return parser


def get_parser():

    parser = general_parser()
    mass_parser(parser)
    fitting_parser(parser)
    training_parser(parser)
    plotting_parser(parser)
    return parser
