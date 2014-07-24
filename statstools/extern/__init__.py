import rootpy.compiled as C
import os
HERE = os.path.dirname(os.path.abspath(__file__))
if os.getenv('OLD', False):
    C.register_file(os.path.join(HERE, 'old_runSig.C'),
                    ['significance'])
    def make_asimov_data(*args, **kwargs):
        return None
    from rootpy.compiled import significance
else:
    C.register_file(os.path.join(HERE, 'runSig.C'),
                    ['significance', 'make_asimov_data'])
    from rootpy.compiled import significance, make_asimov_data

C.register_file(os.path.join(HERE, 'AsymptoticsCLs.C'),
                ['AsymptoticsCLs'])
from rootpy.compiled import AsymptoticsCLs

__all__ = [
    'AsymptoticsCLs',
    'significance',
    'make_asimov_data',
]
