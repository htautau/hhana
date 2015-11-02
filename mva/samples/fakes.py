# local imports
from .sample import SystematicsSample, Background
from . import log; log = log[__name__]


class OS_SS(SystematicsSample, Background):

    def __init__(self, base_sample, sr, cr, **kwargs):
        """
        Basic implementation of the OS - SS method
        Cannonically sr = 'OS', cr = 'SS' but the choice is up
        to the analyser
        """
        log.info(base_sample.year)
        super(SystematicsSample, self).__init__(year=base_sample.year, **kwargs)
        
        self.sample = base_sample
        self.sr = sr
        self.cr = cr
        log.info('OS_SS instantiation')


    def draw_array(self, field_hist, category, region, cuts=None, **kwargs):
        
        sr_region = region & self.sr
        field_hist_sr = dict([
                (expr, hist.Clone())
                for expr, hist in field_hist.items()])
        self.sample.draw_array(
            field_hist_sr,
            category,
            region,
            cuts=self.sr & cuts, **kwargs)

        field_hist_cr = dict([
                (expr, hist.Clone())
                for expr, hist in field_hist.items()])
        self.sample.draw_array(
            field_hist_cr,
            category,
            region,
            cuts=self.cr & cuts, **kwargs)
        
        for field in field_hist.keys():
            hist = field_hist_sr[field] - field_hist_cr[field]
            field_hist[field] = hist
