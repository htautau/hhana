from . import log; log = log[__name__]
from .. import samples, CACHE_DIR
import os
import cPickle as pickle
import atexit


MODIFIED = False
SCALES = {}
SCALES_FILE = os.path.join(CACHE_DIR, 'norm.cache')


def print_scales():

    for year in SCALES.keys():
        for category in sorted(SCALES[year].keys()):
            for embedding in SCALES[year][category].keys():
                params = sorted(SCALES[year][category][embedding].keys())
                for param in params:
                    for shape_region in SCALES[year][category][embedding][param].keys():
                        qcd_scale, qcd_scale_error, \
                        ztautau_scale, ztautau_scale_error = \
                        SCALES[year][category][embedding][param][shape_region]
                        log.info("%d scale factors for %s category" % (year, category))
                        log.info("embedding: %s" % str(embedding))
                        log.info("QCD shape region: %s" % shape_region)
                        log.info("fits were derived via %s parameters" % param)
                        log.info("    qcd scale: %.3f +/- %.4f" % (qcd_scale,
                            qcd_scale_error))
                        log.info("    ztautau scale: %.3f +/- %.4f" %
                            (ztautau_scale, ztautau_scale_error))


if os.path.isfile(SCALES_FILE):
    log.info("reading background scale factors from %s" % SCALES_FILE)
    with open(SCALES_FILE) as cache:
        SCALES = pickle.load(cache)


@atexit.register
def write_scales():

    if MODIFIED:
        with open(SCALES_FILE, 'w') as cache:
            pickle.dump(SCALES, cache)


def qcd_ztautau_norm(ztautau,
                     qcd,
                     category,
                     param):

    # if this is a control region then use the name of the parent category
    if category.is_control:
        category = category.__bases__[0].name
    else:
        category = category.name

    is_embedded = isinstance(ztautau, samples.Embedded_Ztautau)
    param = param.upper()

    scales = get_scales(
        ztautau.year, category, is_embedded, param, qcd.shape_region)

    qcd.data_scale = scales['qcd_data_scale']
    qcd.scale_error = scales['qcd_data_scale_error']

    # assume that the MC order is Z, Others
    qcd.mc_scales = [
        scales['qcd_z_scale'],
        scales['qcd_others_scale']
        ]

    ztautau.scale = scales['z_scale']
    ztautau.scale_error = scales['z_scale_error']


def get_scales(year, category, embedded, param, shape_region, verbose=True):

    year %= 1E3
    category = category.upper()
    param = param.upper()
    if has_category(year, category, embedded, param, shape_region):
        scales = SCALES[year][category][embedded][param][shape_region]
        if verbose:
            log.info("%d scale factors for %s category" % (year, category))
            log.info("embedding: %s" % str(embedded))
            log.info("QCD shape region: %s" % shape_region)
            log.info("scale factors were derived via fit using %s parameters" % param)
            log.info("   QCD data scale: %.3f +/- %.4f" % (
                scales['qcd_data_scale'], scales['qcd_data_scale_error']))
            log.info("    QCD ztt scale: %.3f" % scales['qcd_z_scale'])
            log.info(" QCD others scale: %.3f" % scales['qcd_others_scale'])
            log.info("        ztt scale: %.3f +/- %.4f" % (
                scales['z_scale'], scales['z_scale_error']))

        return scales
    raise ValueError(
        "No scale factors for %d, %s, embedding: %s, param: %s, shape: %s" %
        (year, category, embedded, param, shape_region))


def has_category(year, category, embedded, param, shape_region):

    year %= 1E3
    category = category.upper()
    param = param.upper()
    return (year in SCALES and category in SCALES[year] and
            embedded in SCALES[year][category] and
            param in SCALES[year][category][embedded] and
            shape_region in SCALES[year][category][embedded][param])


def set_scales(year, category, embedded, param, shape_region,
               qcd_data_scale, qcd_data_scale_error,
               qcd_z_scale,
               qcd_others_scale,
               z_scale, z_scale_error):

    global MODIFIED
    year %= 1E3
    param = param.upper()
    category = category.upper()

    log.info("setting %d scale factors for %s category" % (year, category))
    log.info("embedding: %s" % str(embedded))
    log.info("QCD shape region: %s" % shape_region)
    log.info("new scale factors derived via fit using %s parameters" % param)
    log.info("   QCD data scale: %.3f +/- %.4f" % (
        qcd_data_scale, qcd_data_scale_error))
    log.info("    QCD ztt scale: %.3f" % qcd_z_scale)
    log.info(" QCD others scale: %.3f" % qcd_others_scale)
    log.info("        ztt scale: %.3f +/- %.4f" % (z_scale, z_scale_error))

    """
    if has_category(year, category, embedded, param, shape_region):
        scales = get_scales(year, category, embedded, param, shape_region,
                            verbose=False)
        log.info("scale factors were previously:")
        log.info("    qcd scale: %.3f +/- %.4f" % (
                qcd_scale_old,
                qcd_scale_error_old))
        log.info("    ztt scale: %.3f +/- %.4f" % (
                ztautau_scale_old,
                ztautau_scale_error_old))
    """
    if year not in SCALES:
        SCALES[year] = {}
    if category not in SCALES[year]:
        SCALES[year][category] = {}
    if embedded not in SCALES[year][category]:
        SCALES[year][category][embedded] = {}
    if param not in SCALES[year][category][embedded]:
        SCALES[year][category][embedded][param] = {}

    SCALES[year][category][embedded][param][shape_region] = {
        'qcd_data_scale': qcd_data_scale,
        'qcd_data_scale_error': qcd_data_scale_error,
        'qcd_z_scale': qcd_z_scale,
        'qcd_others_scale': qcd_others_scale,
        'z_scale': z_scale,
        'z_scale_error': z_scale_error,
        }
    MODIFIED = True
