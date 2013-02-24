from rootpy.tree import Cut

SYSTEMATICS_TERMS = [
    ('JES_UP',),
    ('JES_DOWN',),
    ('TES_UP',),
    ('TES_DOWN',),
    ('JER_UP',),
    ('MFS_UP',),
    ('MFS_DOWN',),
    ('ISOL_UP',),
    ('ISOL_DOWN',),
]

SYSTEMATICS_BY_WEIGHT = [
    ('TRIGGER_UP',),
    ('TRIGGER_DOWN',),
    ('FAKERATE_UP',),
    ('FAKERATE_DOWN',),
    ('TAUID_UP',),
    ('TAUID_DOWN',),
]

SYSTEMATICS = {
    'JES': (('JES_UP',), ('JES_DOWN',)),
    'TES': (('TES_UP',), ('TES_DOWN',)),
    'JER': (('JER_UP',),),
    #(('MFS_UP',), ('MFS_DOWN',)),
    #(('ISOL_UP',), ('ISOL_DOWN',)),
    #(('TRIGGER_UP',), ('TRIGGER_DOWN',)),
    #'FAKERATE': (('FAKERATE_UP',), ('FAKERATE_DOWN',)),
    'TAUID': (('TAUID_UP',), ('TAUID_DOWN',)),
    #(('QCDFIT_UP',), ('QCDFIT_DOWN',)),
    #(('ZFIT_UP',), ('ZFIT_DOWN',)),
}

WEIGHT_SYSTEMATICS = {
    #'TRIGGER': {
    #    'UP': [
    #        'tau1_trigger_scale_factor_high',
    #        'tau2_trigger_scale_factor_high'],
    #    'DOWN': [
    #        'tau1_trigger_scale_factor_low',
    #        'tau2_trigger_scale_factor_low'],
    #    'NOMINAL': [
    #        'tau1_trigger_scale_factor',
    #        'tau2_trigger_scale_factor']},
    #'FAKERATE': {
    #    'UP': [
    #        'tau1_fakerate_scale_factor_high',
    #        'tau2_fakerate_scale_factor_high'],
    #    'DOWN': [
    #        'tau1_fakerate_scale_factor_low',
    #        'tau2_fakerate_scale_factor_low'],
    #    'NOMINAL': [
    #        'tau1_fakerate_scale_factor',
    #        'tau2_fakerate_scale_factor']},
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


def iter_systematics(include_nominal=False):

    if include_nominal:
        yield 'NOMINAL'
    for term, variations in SYSTEMATICS.items():
        for var in variations:
            yield var
