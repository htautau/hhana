#!/usr/bin/env python

from rootpy.extern.argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--unblind', action='store_false', dest='blind', default=True)
parser.add_argument('--profile-mu', default='1')
parser.add_argument('file')
parser.add_argument('workspace')
args = parser.parse_args()

from statstools import get_significance_workspace
from rootpy.io import root_open

with root_open(args.file) as f:
    if args.workspace not in f:
        f.ls()
    else:
        h = get_significance_workspace(
            f[args.workspace], blind=args.blind,
            mu_profile_value=args.profile_mu, verbose=True)
        print list(h.y())
