#!/usr/bin/env python

from rootpy.extern.argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--unblind', action='store_false', dest='blind', default=True)
parser.add_argument('file')
parser.add_argument('workspace')
args = parser.parse_args()

from mva.stats import hypotests
from rootpy.io import root_open

with root_open(args.file) as f:
    if args.workspace not in f:
        f.ls()
    else:
        h = hypotests.get_significance_workspace(f[args.workspace], verbose=True, blind=args.blind)
        print list(h.y())
