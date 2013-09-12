import os
import ROOT
from rootpy.utils.lock import lock

HERE = os.path.dirname(os.path.abspath(__file__))

with lock(HERE):
    ROOT.gSystem.CompileMacro(os.path.join(HERE, 'src', 'AsymptoticsCLs.C'),
        'k',
        'AsymptoticsCLs',
        '/tmp')

from ROOT import AsymptoticsCLs
