from rootpy.tree import Cut


class CategoryMeta(type):
    """
    Metaclass for all categories
    """
    CATEGORY_REGISTRY = {}
    def __new__(cls, name, bases, dct):

        if name in CategoryMeta.CATEGORY_REGISTRY:
            raise ValueError("Multiple categories with the same name: %s" % name)
        cat = type.__new__(cls, name, bases, dct)
        # register the category
        CategoryMeta.CATEGORY_REGISTRY[name] = cat
        return cat


class Category(object):

    __metaclass__ = CategoryMeta

    # common attrs for all categories. Override in subclasses
    analysis_control = False
    is_control = False
    # category used for normalization
    norm_category = None
    qcd_shape_region = 'nOS' # no track cut
    target_region = 'OS_TRK'
    cuts = Cut()
    common_cuts = Cut()
    from .. import samples
    train_signal_modes = samples.Higgs.MODES[:]
    clf_bins = 8
    # only unblind up to this number of bins in half-blind mode
    # flat, onebkg or constant (see mva/stats/utils.py)
    limitbinning = 'constant'
    plot_label = None

    @classmethod
    def get_cuts(cls, year, deta_cut=True):
        cuts = cls.cuts & cls.common_cuts
        if hasattr(cls, 'year_cuts') and year in cls.year_cuts:
            cuts &= cls.year_cuts[year]
        if 'DEta_Control' in cls.__name__:
            cuts &= Cut('dEta_tau1_tau2 >= 1.5')
        elif deta_cut:
            cuts &= Cut('dEta_tau1_tau2 < 1.5')
        # TODO
        #if 'ID_Control' in cls.__name__ or 'FF' in region:
        #    cuts &= TAUS_FAIL
        #else:
        #    cuts &= TAUS_PASS
        return cuts

    @classmethod
    def get_parent(cls):
        if cls.is_control:
            return cls.__bases__[0]
        return cls
