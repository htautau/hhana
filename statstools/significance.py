import rootpy.compiled as C
import os
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'src', 'runSig.C')
C.register_file(PATH, ['runSig', 'makeAsimovData'])
from rootpy.compiled import runSig, makeAsimovData
__all__ = [
    'runSig',
    'makeAsimovData',
]
