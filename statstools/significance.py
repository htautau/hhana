import rootpy.compiled as C
import os
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'src', 'runSig.C')
C.register_file(PATH, ['significance', 'make_asimov_data'])
from rootpy.compiled import significance, make_asimov_data
__all__ = [
    'significance',
    'make_asimov_data',
]
