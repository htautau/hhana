from rootpy.tree import Cut

class SYSTEMATICS_CATEGORIES:
    TAUS, \
    JETS, \
    WEIGHTS, \
    NORMALIZATION = range(4)

# WIP:
class Systematic(object):

    def __init__(self, name, variations=None):

        if variations is None:
            self.variations = ('UP', 'DOWN')
        else:
            if not isinstance(variations, (list, tuple)):
                variations = (variations,)
            self.variations = variations
        self.name = name

    def __iter__(self):

        for var in self.variations:
            yield '%s_%s' % (self.name, var)


SYSTEMATICS_2011 = {
    'JES': (('JES_UP',), ('JES_DOWN',)),
    'JER': (('JER_UP',),),
    'TES': (('TES_UP',), ('TES_DOWN',)),
    'MFS': (('MFS_UP',), ('MFS_DOWN',)),
    'ISOL': (('ISOL_UP',), ('ISOL_DOWN',)),
    'TRIGGER': (('TRIGGER_UP',), ('TRIGGER_DOWN',)),
    'FAKERATE': (('FAKERATE_UP',), ('FAKERATE_DOWN',)),
    'TAUID': (('TAUID_UP',), ('TAUID_DOWN',)),
    'QCD_FIT': (('QCDFIT_UP',), ('QCDFIT_DOWN',)),
    'Z_FIT': (('ZFIT_UP',), ('ZFIT_DOWN',)),
}

SYSTEMATICS_2012 = {

    'MET_RESOSOFTTERMS': (('MET_RESOSOFTTERMS_UP',), ('MET_RESOSOFTTERMS_DOWN',)),
    'MET_SCALESOFTTERMS': (('MET_SCALESOFTTERMS_UP',), ('MET_SCALESOFTTERMS_DOWN',)),

    #'JES': (('JES_UP',), ('JES_DOWN',)),
    'JES_Modelling': (('JES_Modelling_UP',), ('JES_Modelling_DOWN',)),
    'JES_Detector': (('JES_Detector_UP',), ('JES_Detector_DOWN',)),
    'JES_EtaModelling': (('JES_EtaModelling_UP',), ('JES_EtaModelling_DOWN',)),
    'JES_EtaMethod': (('JES_EtaMethod_UP',), ('JES_EtaMethod_DOWN',)),
    'JES_PURho': (('JES_PURho_UP',), ('JES_PURho_DOWN',)),
    'JES_FlavComp': (('JES_FlavComp_UP',), ('JES_FlavComp_DOWN',)),
    'JES_FlavResp': (('JES_FlavResp_UP',), ('JES_FlavResp_DOWN',)),

    'JVF': (('JVF_UP',), ('JVF_DOWN',)),
    'JER': (('JER_UP',),),

    #'TES': (('TES_UP',), ('TES_DOWN',)),
    'TES_TRUE': (('TES_TRUE_UP',), ('TES_TRUE_DOWN',)),
    'TES_FAKE': (('TES_FAKE_UP',), ('TES_FAKE_DOWN',)),
    #'TES_EOP': (('TES_EOP_UP',), ('TES_EOP_DOWN',)),
    #'TES_CTB': (('TES_CTB_UP',), ('TES_CTB_DOWN',)),
    #'TES_Bias': (('TES_Bias_UP',), ('TES_Bias_DOWN',)),
    #'TES_EM': (('TES_EM_UP',), ('TES_EM_DOWN',)),
    #'TES_LCW': (('TES_LCW_UP',), ('TES_LCW_DOWN',)),
    #'TES_PU': (('TES_PU_UP',), ('TES_PU_DOWN',)),
    #'TES_OTHERS': (('TES_OTHERS_UP',), ('TES_OTHERS_DOWN',)),

    'MFS': (('MFS_UP',), ('MFS_DOWN',)),
    'ISOL': (('ISOL_UP',), ('ISOL_DOWN',)),

    'TRIGGER': (('TRIGGER_UP',), ('TRIGGER_DOWN',)),
    'FAKERATE': (('FAKERATE_UP',), ('FAKERATE_DOWN',)),
    'TAUID': (('TAUID_UP',), ('TAUID_DOWN',)),

    'QCD_FIT': (('QCDFIT_UP',), ('QCDFIT_DOWN',)),
    'Z_FIT': (('ZFIT_UP',), ('ZFIT_DOWN',)),

    'QCD_SHAPE': (('QCDSHAPE_UP',), ('QCDSHAPE_DOWN',)),
}

SYSTEMATICS_BY_WEIGHT = [
    ('TRIGGER_UP',),
    ('TRIGGER_DOWN',),
    ('FAKERATE_UP',),
    ('FAKERATE_DOWN',),
    ('TAUID_UP',),
    ('TAUID_DOWN',),
]

WEIGHT_SYSTEMATICS = {
    'TRIGGER': {
        'UP': [
            'tau1_trigger_scale_factor_high',
            'tau2_trigger_scale_factor_high'],
        'DOWN': [
            'tau1_trigger_scale_factor_low',
            'tau2_trigger_scale_factor_low'],
        'NOMINAL': [
            'tau1_trigger_scale_factor',
            'tau2_trigger_scale_factor']},
    'FAKERATE': {
        'UP': [
            'tau1_fakerate_scale_factor_high',
            'tau2_fakerate_scale_factor_high'],
        'DOWN': [
            'tau1_fakerate_scale_factor_low',
            'tau2_fakerate_scale_factor_low'],
        'NOMINAL': [
            'tau1_fakerate_scale_factor',
            'tau2_fakerate_scale_factor']},
    'TAUID': {
        'UP': [
            'tau1_efficiency_scale_factor_high',
            'tau2_efficiency_scale_factor_high'],
        'DOWN': [
            'tau1_efficiency_scale_factor_low',
            'tau2_efficiency_scale_factor_low'],
        'NOMINAL': [
            'tau1_efficiency_scale_factor',
            'tau2_efficiency_scale_factor']},
}

EMBEDDING_SYSTEMATICS = {
    'ISOL': { # MUON ISOLATION
        'UP': Cut('(embedding_isolation == 2)'),
        'DOWN': Cut(),
        'NOMINAL': Cut('(embedding_isolation >= 1)'),
    }
}


def iter_systematics(include_nominal=False, year=2012, components=None):

    syst = get_systematics(year)
    if include_nominal:
        yield 'NOMINAL'
    terms = components if components is not None else syst.keys()
    for term in terms:
        try:
            variations = syst[term]
        except KeyError:
            raise ValueError("systematic term {0} is not defined".format(term))
        for var in variations:
            yield var


def get_systematics(year=2012):

    if year == 2012:
        return SYSTEMATICS_2012
    elif year == 2011:
        return SYSTEMATICS_2011
    else:
        raise ValueError("No systematics defined for year %d" % year)


def systematic_name(systematic):

    if isinstance(systematic, basestring):
        return systematic
    return '_'.join(systematic)


def parse_systematics(string):

    if not string:
        return None
    return [tuple(token.split('+')) for token in string.split(',')]
