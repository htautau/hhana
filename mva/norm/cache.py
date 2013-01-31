#!/usr/bin/env python
from .. import log, CACHE_DIR
import os
import cPickle as pickle

if __name__ == '__main__':
    log = log[os.path.splitext(os.path.basename(__file__))[0]]
else:
    log = log[__name__]

SCALES_FILE = None
SCALES = {}
MODIFIED = False


def read_scales(name='norm.cache'):

    global SCALES_FILE
    global SCALES

    SCALES_FILE = os.path.join(CACHE_DIR, name)
    log.info("reading background scale factors from %s" % SCALES_FILE)

    if os.path.isfile(SCALES_FILE):
        with open(SCALES_FILE) as cache:
            SCALES = pickle.load(cache)


def get_scales(year, category, embedded, param, verbose=True):

    year %= 1E3
    category = category.upper()
    param = param.upper()
    if has_category(year, category, embedded, param):
        qcd_scale, qcd_scale_error, \
        ztautau_scale, ztautau_scale_error = SCALES[year][category][embedded][param]
        if verbose:
            log.info("background normalization for year %d" % year)
            log.info("using the embedding scale factors: %s" % str(embedded))
            log.info("scale factors for %s category" % category)
            log.info("fits were derived via %s parameters" % param)
            log.info("    qcd scale: %.3f +/- %.4f" % (qcd_scale,
                qcd_scale_error))
            log.info("    ztautau scale: %.3f +/- %.4f" % (
                    ztautau_scale, ztautau_scale_error))
        return qcd_scale, qcd_scale_error, ztautau_scale, ztautau_scale_error
    else:
        return None


def has_category(year, category, embedded, param):

    year %= 1E3
    category = category.upper()
    param = param.upper()
    return (year in SCALES and category in SCALES[year] and
            embedded in SCALES[year][category] and
            param in SCALES[year][category][embedded])


def set_scales(year, category, embedded, param,
        qcd_scale, qcd_scale_error,
        ztautau_scale, ztautau_scale_error):

    global MODIFIED
    year %= 1E3
    param = param.upper()
    category = category.upper()
    log.info("background normalization for year %d" % year)
    log.info("setting the embedding scale factors: %s" % str(embedded))
    log.info("setting scale factors for %s category" % category)
    log.info("fits were derived via %s parameters" % param)
    log.info("    qcd scale: %.3f +/- %.4f" % (qcd_scale, qcd_scale_error))
    log.info("    ztautau scale: %.3f +/- %.4f" % (ztautau_scale,
        ztautau_scale_error))
    if has_category(year, category, embedded, param):
        qcd_scale_old, qcd_scale_error_old, \
        ztautau_scale_old, ztautau_scale_error_old = get_scales(
                year, category, embedded, param, verbose=False)
        log.info("scale factors were previously:")
        log.info("    qcd scale: %.3f +/- %.4f" % (
                qcd_scale_old,
                qcd_scale_error_old))
        log.info("    ztautau scale: %.3f +/- %.4f" % (
                ztautau_scale_old,
                ztautau_scale_error_old))
    if year not in SCALES:
        SCALES[year] = {}
    if category not in SCALES:
        SCALES[year][category] = {}
    if embedded not in SCALES[year][category]:
        SCALES[year][category][embedded] = {}
    SCALES[year][category][embedded][param] = (
            qcd_scale, qcd_scale_error,
            ztautau_scale, ztautau_scale_error)
    MODIFIED = True


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--embedding', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('cache', default='background_scales.cache', nargs='?')
    args = parser.parse_args()

    z_norms = {}
    qcd_norms = {}
    read_scales(args.cache)
    for year in SCALES.keys():
        for category in sorted(SCALES[year].keys()):
            for embedding in SCALES[category].keys():
                if embedding != args.embedding:
                        continue
                params = sorted(SCALES[category][embedding].keys())
                for param in params:
                    qcd_scale, qcd_scale_error, \
                    ztautau_scale, ztautau_scale_error = \
                    SCALES[category][embedding][param]
                    if param not in z_norms:
                        z_norms[param] = []
                        qcd_norms[param] = []
                    z_norms[param].append((ztautau_scale, ztautau_scale_error))
                    qcd_norms[param].append((qcd_scale, qcd_scale_error))
                    log.info("scale factors for embedding: %s" % str(embedding))
                    log.info("scale factors for %s category" % category)
                    log.info("fits were derived via %s parameters" % param)
                    log.info("    qcd scale: %.3f +/- %.4f" % (qcd_scale,
                        qcd_scale_error))
                    log.info("    ztautau scale: %.3f +/- %.4f" %
                        (ztautau_scale, ztautau_scale_error))
    if args.plot:
        from matplotlib import pyplot as plt

        plt.figure()
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

        for param in params:
            z_n, z_e = zip(*z_norms[param])
            q_n, q_e = zip(*qcd_norms[param])
            x = range(len(z_n))

            ax = axs[0]
            ax.errorbar(x, z_n, z_e, fmt='o',
                    markersize=5, label='%s Fit' % param)
            ax = axs[1]
            ax.errorbar(x, q_n, q_e, fmt='o',
                    markersize=5, label='%s Fit' % param)

        axs[0].set_ylim(.5, 2)
        axs[0].set_ylabel('Z Scale Factor')

        axs[1].set_ylim(.5, 2.5)
        axs[1].set_xticklabels([''] + categories)
        axs[1].set_xlim(-0.5, len(z_norms[params[0]]) - .5)
        axs[1].set_ylabel('QCD Scale Factor')

        l1 = axs[0].legend(numpoints=1)
        l2 = axs[1].legend(numpoints=1)

        l1.get_frame().set_fill(False)
        l1.get_frame().set_linewidth(0)

        l2.get_frame().set_fill(False)
        l2.get_frame().set_linewidth(0)

        out_name = 'bkg_norms'
        if args.embedding:
            out_name += '_embedding'

        for f in ('png', 'eps'):
            plt.savefig('%s.%s' % (out_name, f))

else:
    import atexit

    read_scales()

    @atexit.register
    def write_scales():

        if MODIFIED:
            with open(SCALES_FILE, 'w') as cache:
                pickle.dump(SCALES, cache)
