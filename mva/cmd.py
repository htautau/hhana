import rootpy
from rootpy.extern import argparse

from .categories import CATEGORIES
from .massregions import DEFAULT_LOW_MASS, DEFAULT_HIGH_MASS
from .variables import VARIABLES
from .regions import QCD_SHAPE_REGIONS, TARGET_REGIONS


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
    parser.add_argument('actions', nargs='*',
            choices=[
                'norm',
                'stability',
                'validate',
                'weights',
                '2d',
                'plot',
                'train',
                'money',
                'scatter',
                'correlate',
                'evaluate',
                'workspace',
                'track-workspace',
                'deta-workspace',
                'bdt-workspace',
                'ntup',
                'ntuptruth',
                'massplot'],
            default=[],
            help='only perform these actions')
    parser.add_argument('--no-systematics', action='store_false',
            dest='systematics',
            help="turn off systematics",
            default=True)
    parser.add_argument('--categories', default='mva',
            choices=CATEGORIES.keys(),
            help='category definitions')
    parser.add_argument('--category-names', nargs="+", default=None,
            help='category names')
    parser.add_argument('--controls', default='mva_workspace_controls',
            help='control definitions')
    parser.add_argument('--unblind', action='store_true', default=False,
            help='plot the data in the signal region of the classifier output')
    parser.add_argument('--random-mu', action='store_true', default=False,
            help='set mu (signal strength) to a random number')
    parser.add_argument('--mu', default=1., type=float,
            help='set mu (signal strength)')
    parser.add_argument('--no-embedding', action='store_false', default=True,
            dest='embedding',
            help='use ALPGEN Z->tau+tau instead of embedding')
    parser.add_argument('--year', type=int, default=2012, choices=(2011, 2012),
            help='the year')
    parser.add_argument('--qcd-shape-region', choices=QCD_SHAPE_REGIONS,
            default='nOS',
            help='QCD shape region')
    parser.add_argument('--decouple-qcd-shape', action='store_true', default=False)
    parser.add_argument('--optimize-limits', default=False, action='store_true')
    parser.add_argument('--mass-points', default='125')
    parser.add_argument('--target-region', choices=TARGET_REGIONS,
            default='OS_TRK',
            help='target signal region')
    parser.add_argument('--suffix', default=None, nargs='?',
            help='suffix to add to any output files or plots')
    parser.add_argument('--workspace-suffix', default=None, nargs='?',
            help='suffix to add to workspace output files')
    parser.add_argument('--systematics-components', default=None,
            help='only include the following systematics in plots Example: '
                 'TES_TRUE_UP,QCD_SHAPE_UP')
    parser.add_argument('--constrain-norms',
            action='store_true', default=False)
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
    parser.add_argument('--no-sideband-in-control',
            dest='high_sideband_in_control',
            action='store_false',
            default=True,
            help='Exclude the high mass sideband in the mass control and include '
            'it in the signal region')
    parser.add_argument('--no-mmc', action='store_true', default=False,
            help="do not include the MMC in the trained classifier")
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
    parser.add_argument('--fit-param', choices=('track', 'mass'),
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
    parser.add_argument('--nfold', type=int, default=10,
            help='the number of folds in the cross-validation')
    parser.add_argument('--train-categories', nargs='*', default=[],
            help='only train in these categories')
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
    parser.add_argument('--raw-scores', action='store_true',
            default=False,
            help='use raw classifier scores instead of applying '
                 'a logistic transformation')
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
    parser.add_argument('--mpl', action='store_true', default=False,
            help='draw plots with matplotlib. Default is ROOT')
    parser.add_argument('--no-weight', action='store_true', default=False,
            help='do not apply correction weights')
    parser.add_argument('--output-formats', default=['png'], nargs='+',
            choices=('png', 'eps', 'pdf'),
            help='output formats')

    return parser


def fit_parser():

    parser = base_parser()
    parser.add_argument('categories', nargs='+', choices=CATEGORIES.keys(),
            help='category definitions')
    parser.add_argument('--plot', action='store_true', default=False,
            help='plot distributions before and after fit')
    parser.add_argument('--roofit', action='store_true', default=False,
            help='use RooFit instead of TrackFit')

    return parser


def get_parser():

    parser = general_parser()
    mass_parser(parser)
    fitting_parser(parser)
    training_parser(parser)
    plotting_parser(parser)
    return parser
