import os
import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT.gSystem.CompileMacro(os.path.join(HERE, 'src', 'runSig.C'),
        'k',
        'runSig',
        '/tmp')

from ROOT import runSig
