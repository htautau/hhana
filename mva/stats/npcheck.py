import os
import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT.gSystem.CompileMacro(os.path.join(HERE, 'src', 'FitCrossCheckForLimits.C'),
        'k',
        'NuisanceCheck',
        '/tmp')

from ROOT import LimitCrossCheck as NuisanceCheck

plot = NuisanceCheck.PlotFitCrossChecks
