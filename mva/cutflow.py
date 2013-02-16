import math
import os
import sys

from argparse import ArgumentParser

from higgstautau import datasets
from rootpy.extern.tabulartext import TextTable
from rootpy.io import open as ropen
from .periods import LUMI
from . import log; log = log[__name__]


def get_parser():

    parser = ArgumentParser()
    parser.add_argument('--format', choices=('latex', 'text'), default='text')
    parser.add_argument('--errors', action='store_true', default=False)
    parser.add_argument('--precision', default=2)
    parser.add_argument('--proc', default='HHProcessor')
    parser.add_argument('--db', default='datasets_hh')
    parser.add_argument('--year', type=int, default=2012)
    parser.add_argument('--no-weight', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--rst', action='store_true', default=False)
    parser.add_argument('--rst-class', default='cutflow')
    parser.add_argument('--total', action='store_true', default=False)
    parser.add_argument('--landscape', action='store_true', default=False)
    parser.add_argument('--warn-missing', action='store_true', default=False)
    return parser


def tab(s, tabstr=4 * ' '):

    return '\n'.join(((tabstr) + x if x else x) for x in s.splitlines())


def make_cutflow(samples, args):

    used_samples, filters, cutflows, data_index = make_cutflow_table(samples, args)
    print_cutflow(used_samples, filters, cutflows, data_index, args)


def make_cutflow_table(samples, args):

    filters = None
    db = datasets.Database(args.db)

    # build table
    cutflow_table = {}
    data_index = []
    used_samples = []
    i = 0
    for latex_name, text_name, sample in samples:
        matched_samples = db.search(sample)
        if not matched_samples:
            if args.warn_missing:
                log.warning("sample %s not found" % sample)
                continue
            raise datasets.NoMatchingDatasetsFound(sample)
        total_cutflow = None
        for ds in matched_samples:
            log.info(ds.name)
            for filename in ds.files:
                log.debug(filename)
                with ropen(filename) as rfile:
                    cutflow = rfile.cutflow_event
                    cutflow.SetDirectory(0)

                    if filters is None:
                        filters = [cutflow.GetXaxis().GetBinLabel(j + 1) for j in xrange(len(cutflow))]

                    # scale MC by lumi and xsec
                    if ds.datatype != datasets.DATA and not args.no_weight:
                        lumi = LUMI[args.year]
                        events = cutflow[0]
                        xsec, kfact, effic = ds.xsec_kfact_effic
                        weight = 1E3 * lumi * xsec * ds.xsec_factor / (effic * events)
                        if args.verbose:
                            print '-' * 30
                            print ds.name
                            print "xsec: %f [nb]" % xsec
                            print "effic: %f" % effic
                            print "events: %d" % events
                            print "lumi: %f [1/pb]" % lumi
                            print "weight (1E3 * lumi * xsec / (effic * events)): %f" % weight
                        cutflow *= weight

                    if ds.datatype == datasets.DATA:
                        data_index.append(i)

                    if total_cutflow is None:
                        total_cutflow = cutflow
                    else:
                        total_cutflow += cutflow
        if args.errors:
            cutflow_table[i] = list(zip(total_cutflow, total_cutflow.yerrh()))
        else:
            cutflow_table[i] = list(total_cutflow)
        used_samples.append((latex_name, text_name, sample))
        i += 1

    cutflows = [cutflow_table[idx] for idx in xrange(i)]
    return used_samples, filters, cutflows, data_index


def print_cutflow(samples, filters, cutflows, data_index, args, stream=None):

    if args.landscape:
        cutflows = zip(*cutflows)
    if stream is None:
        stream = sys.stdout
    print >> stream
    if args.format == 'text':
        sample_names = [sample[1] for sample in samples]
        table = TextTable(max_width=-1)
        if args.no_weight:
            dtypes = ['t'] + ['i'] * len(cutflows)
        else:
            dtypes = ['t'] + ['f'] * len(cutflows)
            if not args.landscape:
                for i in data_index:
                    dtypes[i + 1] = 'i'
        table.set_cols_dtype(dtypes)
        table.set_precision(args.precision)
        if args.landscape:
            if args.errors:
                table.add_row(['Sample'] + filters)
                for i, row in enumerate(zip(*cutflows)):
                    table.add_row([sample_names[i]] +
                                  ["%s(%s)" % (num_format % passing,
                                               num_format % error)
                                               for passing, error in row])
            else:
                table.add_rows([['Sample'] + filters] +
                               zip(*([sample_names] + cutflows)))
        else:
            if args.errors:
                table.add_row(['Filter'] + sample_names)
                for i, row in enumerate(zip(*cutflows)):
                    table.add_row([filters[i]] +
                                  ["%s(%s)" % (num_format % passing,
                                               num_format % error)
                                               for passing, error in row])
            else:
                table.add_rows([['Filter'] + sample_names] +
                               zip(*([filters] + cutflows)))
        table_str = table.draw()
        if args.rst:
            print >> stream, ".. table::"
            print >> stream, "   :class: %s" % args.rst_class
            print >> stream
            print >> stream, tab(table_str, 3 * ' ')
        else:
            print >> stream, table_str
    else:
        sample_names = [sample[0] for sample in samples]
        print >> stream, r'\begin{center}'
        print >> stream, r'\begin{scriptsize}'
        print >> stream, r'\begin{tabular}{%s}' % ('|'.join(['c' for i in xrange(len(samples) + 1)]))
        print >> stream, " & ".join(['Filter'] + sample_names) + '\\\\'
        print >> stream, r'\hline\hline'
        if args.short:
            for i, row in enumerate(zip(*cutflows)):
                print >> stream, " & ".join([filters[i]] + [num_format % passing
                                  for passing, error in row]) + '\\\\'
        else:
            for i, row in enumerate(zip(*cutflows)):
                print >> stream, " & ".join([filters[i]] + ["$%s\pm%s$" % (num_format % passing,
                                                                num_format % error)
                                  for passing, error in row]) + '\\\\'
        print >> stream, r'\end{tabular}'
        print >> stream, r'\end{scriptsize}'
        print >> stream, r'\end{center}'
    print >> stream
