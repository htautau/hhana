from rootpy import asrootpy
import os
import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT.gSystem.CompileMacro(os.path.join(HERE, 'src', 'smooth.C'),
        'k',
        'smooth',
        '/tmp')

from ROOT import Smooth

def smooth(*args, **kwargs):
    return asrootpy(Smooth.EqualArea(*args), **kwargs)

def smooth_alt(*args, **kwargs):
    return asrootpy(Smooth.EqualAreaGabriel(*args), **kwargs)
