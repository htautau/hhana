from rootpy.tree import Cut

from .categories import BAD_MASS
from . import log; log = log[__name__]


DEFAULT_LOW_MASS = 110
DEFAULT_HIGH_MASS = 180


class MassRegions(object):

    def __init__(self,
            low=DEFAULT_LOW_MASS,
            high=DEFAULT_HIGH_MASS,
            high_sideband_in_control=False,
            full_signal_region=False):

        assert low > BAD_MASS

        # control region is low and high mass sidebands
        self.__control_region = Cut('mass_mmc_tau1_tau2 < %d' % low)
        if high_sideband_in_control:
            assert high > low
            self.__control_region |= Cut('mass_mmc_tau1_tau2 > %d' % high)

        if full_signal_region:
            self.__signal_region = Cut()
        else:
            # signal region is the negation of the control region
            self.__signal_region = -self.__control_region

        # train on everything
        self.__train_region = Cut()

        log.info("control region: %s" % self.__control_region)
        log.info("signal region: %s" % self.__signal_region)
        log.info("train region: %s" % self.__train_region)

    @property
    def control_region(self):

        # make a copy
        return Cut(self.__control_region)

    @property
    def signal_region(self):

        # make a copy
        return Cut(self.__signal_region)

    @property
    def train_region(self):

        # make a copy
        return Cut(self.__train_region)
