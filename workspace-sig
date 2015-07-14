#!/usr/bin/env python

from rootpy.extern.argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--unblind', action='store_true', default=False)
parser.add_argument('--profile', default=None)
parser.add_argument('file')
parser.add_argument('workspace')
args = parser.parse_args()

from statstools.significance import significance
from rootpy.io import root_open

with root_open(args.file) as f:
    if args.workspace not in f:
        f.ls()
    else:
        sig, mu, mu_error = significance(
            f[args.workspace],
            observed=args.unblind,
            profile=args.profile,
            verbose=True)
        print sig
