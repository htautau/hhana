import rootpy.compiled as C
import os
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'src', 'AsymptoticsCLs.C')
C.register_file(PATH, ['AsymptoticsCLs'])
from rootpy.compiled import AsymptoticsCLs
__all__ = ['AsymptoticsCLs']
