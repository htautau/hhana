from rootpy.tree import Cut
from . import log; log = log[__name__]


DEFAULT_LOW_MASS = 110
DEFAULT_HIGH_MASS = 180


class MassRegions(object):

    def __init__(self,
            low=DEFAULT_LOW_MASS,
            high=DEFAULT_HIGH_MASS,
            high_sideband_in_control=False,
            mass_window_signal_region=False,
            low_cutoff=None):

        # control region is low and high mass sidebands
        self.__control_region = Cut('mass_mmc_tau1_tau2 < %d' % low)
        if low_cutoff is not None:
            self.__control_region &= Cut('mass_mmc_tau1_tau2 > %d' % low_cutoff)
        if high_sideband_in_control:
            assert high > low
            self.__control_region |= Cut('mass_mmc_tau1_tau2 > %d' % high)

        if mass_window_signal_region:
            # signal region is the negation of the control region
            self.__signal_region = -self.__control_region
        else:
            self.__signal_region = Cut()

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
