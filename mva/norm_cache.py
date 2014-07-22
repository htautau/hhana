from . import log; log = log[__name__]
from . import samples, CACHE_DIR

import os
import yaml
import atexit

from rootpy.utils.lock import lock


MODIFIED = False
UPDATED = set()
SCALES_FILE = os.path.join(CACHE_DIR, 'norm.yml')


def read_scales():
    scales = {}
    if os.path.isfile(SCALES_FILE):
        log.info("reading background scale factors from %s" % SCALES_FILE)
        with lock(SCALES_FILE):
            with open(SCALES_FILE) as cache:
                scales = yaml.load(cache)
    return scales


SCALES = read_scales()


@atexit.register
def write_scales():
    if not MODIFIED:
        return
    with lock(SCALES_FILE):
        # merge with possible changes made by another process
        scales = {}
        if os.path.isfile(SCALES_FILE):
            with open(SCALES_FILE) as cache:
                scales = yaml.load(cache)
        for year, category, embedded, param, shape_region, target_region in UPDATED:
            if year not in scales:
                scales[year] = {}
            if category not in scales[year]:
                scales[year][category] = {}
            if embedded not in scales[year][category]:
                scales[year][category][embedded] = {}
            if param not in scales[year][category][embedded]:
                scales[year][category][embedded][param] = {}
            if shape_region not in scales[year][category][embedded][param]:
                scales[year][category][embedded][param][shape_region] = {}
            scales[year][category][embedded][param][shape_region][target_region] = \
                SCALES[year][category][embedded][param][shape_region][target_region]
        with open(SCALES_FILE, 'w') as cache:
            yaml.dump(scales, cache, default_flow_style=False)


def qcd_ztautau_norm(ztautau, qcd,
                     category, param, target_region):
    norm_category = getattr(category, 'norm_category', None)
    if norm_category is not None:
        category = norm_category.name
    elif category.is_control:
        # if this is a control region then use the name of the parent category
        category = category.__bases__[0].name
    else:
        category = category.name

    is_embedded = isinstance(ztautau, samples.Embedded_Ztautau)
    param = param.upper()

    scales = get_scales(
        ztautau.year, category, is_embedded, param,
        qcd.shape_region, target_region)

    qcd.scale = scales['qcd_scale']
    qcd.scale_error = scales['qcd_scale_error']
    qcd.data_scale = scales['qcd_data_scale']
    # assume that the MC order is Z, Others
    assert(isinstance(qcd.mc[0], samples.Ztautau))
    qcd.mc_scales = [
        scales['qcd_z_scale'],
        scales['qcd_others_scale']
        ]
    ztautau.scale = scales['z_scale']
    ztautau.scale_error = scales['z_scale_error']


def get_scales(year, category, embedded, param,
               shape_region, target_region,
               verbose=True):
    year %= 1000
    category = category.upper()
    param = param.upper()
    if has_category(year, category, embedded, param, shape_region, target_region):
        scales = SCALES[year][category][embedded][param][shape_region][target_region]
        if verbose:
            log.info("%d scale factors for %s category" % (year, category))
            log.info("embedding: %s" % str(embedded))
            log.info("fakes shape region: %s" % shape_region)
            log.info("target region: %s" % target_region)
            log.info("derived from fit of %s" % param)
            log.info("        QCD scale: %.3f +/- %.4f" % (
                scales['qcd_scale'], scales['qcd_scale_error']))
            log.info("   QCD data scale: %.3f" % scales['qcd_data_scale'])
            log.info("    QCD ztt scale: %.3f" % scales['qcd_z_scale'])
            log.info(" QCD others scale: %.3f" % scales['qcd_others_scale'])
            log.info("        ztt scale: %.3f +/- %.4f" % (
                scales['z_scale'], scales['z_scale_error']))

        return scales
    raise ValueError(
        "No scale factors for %d, %s, embedding: %s, param: %s, shape: %s" %
        (year, category, embedded, param, shape_region))


def has_category(year, category, embedded, param, shape_region, target_region):
    year %= 1000
    category = category.upper()
    param = param.upper()
    try:
        SCALES[year][category][embedded][param][shape_region][target_region]
        return True
    except KeyError:
        return False


def set_scales(year, category, embedded, param,
               shape_region, target_region,
               qcd_scale, qcd_scale_error,
               qcd_data_scale,
               qcd_z_scale,
               qcd_others_scale,
               z_scale, z_scale_error):
    global MODIFIED
    if shape_region == target_region:
        raise ValueError(
            "fakes shape region cannot equal "
            "the target region: {0}".format(target_region))
    year %= 1000
    param = param.upper()
    category = category.upper()

    log.info("setting %d scale factors for %s category" % (year, category))
    log.info("embedding: %s" % str(embedded))
    log.info("fakes shape region: %s" % shape_region)
    log.info("target region: %s" % target_region)
    log.info("derived from fit of %s" % param)
    log.info("        QCD scale: %.3f +/- %.4f" % (qcd_scale, qcd_scale_error))
    log.info("   QCD data scale: %.3f" % qcd_data_scale)
    log.info("    QCD ztt scale: %.3f" % qcd_z_scale)
    log.info(" QCD others scale: %.3f" % qcd_others_scale)
    log.info("        ztt scale: %.3f +/- %.4f" % (z_scale, z_scale_error))

    if year not in SCALES:
        SCALES[year] = {}
    if category not in SCALES[year]:
        SCALES[year][category] = {}
    if embedded not in SCALES[year][category]:
        SCALES[year][category][embedded] = {}
    if param not in SCALES[year][category][embedded]:
        SCALES[year][category][embedded][param] = {}
    if shape_region not in SCALES[year][category][embedded][param]:
        SCALES[year][category][embedded][param][shape_region] = {}

    SCALES[year][category][embedded][param][shape_region][target_region] = {
        'qcd_scale': qcd_scale,
        'qcd_scale_error': qcd_scale_error,
        'qcd_data_scale': qcd_data_scale,
        'qcd_z_scale': qcd_z_scale,
        'qcd_others_scale': qcd_others_scale,
        'z_scale': z_scale,
        'z_scale_error': z_scale_error,
        }
    UPDATED.add((year, category, embedded, param, shape_region, target_region))
    MODIFIED = True
