import os
import atexit
import tables
from rootpy.io import root_open, TemporaryFile
from higgstautau import datasets
from .. import NTUPLE_PATH, DEFAULT_STUDENT
from . import log; log = log[__name__]

VERBOSE = False

DB = datasets.Database(name='datasets_hh', verbose=VERBOSE)
FILES = {}


TEMPFILE = TemporaryFile()


def get_file(student=DEFAULT_STUDENT, hdf=False, suffix=''):

    if hdf:
        ext = '.h5'
    else:
        ext = '.root'
    filename = student + ext
    if filename in FILES:
        return FILES[filename]
    file_path = os.path.join(NTUPLE_PATH, student + suffix, filename)
    log.info("opening %s ..." % file_path)
    if hdf:
        student_file = tables.openFile(file_path)#, driver="H5FD_CORE")
    else:
        student_file = root_open(file_path, 'READ')
    FILES[filename] = student_file
    return student_file


@atexit.register
def cleanup():

    if TEMPFILE:
        TEMPFILE.close()
    for filehandle in FILES.values():
        if filehandle:
            filehandle.close()
