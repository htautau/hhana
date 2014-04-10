import rootpy.compiled as C
import os
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'src', 'runSig.C')
C.register_file(PATH, ['runSig'])
from rootpy.compiled import runSig
__all__ = ['runSig']
