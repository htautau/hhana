import os
import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT.gSystem.CompileMacro(os.path.join(HERE, 'runAsymptoticsCLs.C'),
        'k',
        'runAsymptoticsCLs',
        '/tmp')
