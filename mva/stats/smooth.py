from . import log; log = log[__name__]
from rootpy import asrootpy
from rootpy.utils.lock import lock
import os
import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))

with lock(HERE):
    ROOT.gSystem.CompileMacro(os.path.join(HERE, 'src', 'smooth.C'),
        'k',
        'smooth',
        '/tmp')

from ROOT import Smooth

__all__ = [
    'smooth',
    'smooth_alt',
]

def smooth(nom, sys, frac=0.5, **kwargs):
    log.info('smoothing {0}'.format(sys.name))
    return asrootpy(Smooth.EqualArea(nom, sys, frac), **kwargs)

def smooth_alt(nom, sys, **kwargs):
    log.info('smoothing {0}'.format(sys.name))
    return asrootpy(Smooth.EqualAreaGabriel(nom, sys), **kwargs)
