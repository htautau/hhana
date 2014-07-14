import rootpy.compiled as C
import os
HERE = os.path.dirname(os.path.abspath(__file__))
C.register_file(os.path.join(HERE, 'runSig.C'),
                ['significance', 'make_asimov_data'])
C.register_file(os.path.join(HERE, 'AsymptoticsCLs.C'),
                ['AsymptoticsCLs'])
from rootpy.compiled import AsymptoticsCLs
from rootpy.compiled import significance, make_asimov_data

__all__ = [
    'AsymptoticsCLs',
    'significance',
    'make_asimov_data',
]
