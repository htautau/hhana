import os
import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT.gSystem.CompileMacro(os.path.join(HERE, 'src', 'smooth.C'),
        'k',
        'smooth',
        '/tmp')

from ROOT import Smooth

EqualArea = Smooth.EqualArea
EqualAreaGabriel = Smooth.EqualAreaGabriel
