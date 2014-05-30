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
    target_region = 'OS_ISOL'
    cuts = Cut()
    common_cuts = Cut()
    from .. import samples
    # by default train with all modes
    train_signal_modes = samples.Higgs.MODES[:]
    plot_label = None

    @classmethod
    def get_cuts(cls, year, deta_cut=True):
        cuts = cls.cuts & cls.common_cuts
        if hasattr(cls, 'year_cuts') and year in cls.year_cuts:
            cuts &= cls.year_cuts[year]
        return cuts

    @classmethod
    def get_parent(cls):
        if cls.is_control:
            return cls.__bases__[0]
        return cls
