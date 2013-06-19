#!/usr/bin/env python

import sys

from mva.stats import hypotests
from rootpy.io import root_open

fname, wname = sys.argv[1:]

with root_open(fname) as f:
    h = hypotests.get_significance_workspace(f[wname], verbose=True)
    print list(h)
