SYSTEMATICS = [
    (('JES_UP',), ('JES_DOWN',)),
    (('TES_UP',), ('TES_DOWN',)),
    (('JER_UP',),),
    (('MFS_UP',), ('MFS_DOWN',)),
    (('ISOL_UP',), ('ISOL_DOWN',)),
    #(('TRIGGER_UP',), ('TRIGGER_DOWN',)),
    (('FAKERATE_UP',), ('FAKERATE_DOWN',)),
    (('TAUID_UP',), ('TAUID_DOWN',)),
    (('QCDFIT_UP',), ('QCDFIT_DOWN',)),
    (('ZFIT_UP',), ('ZFIT_DOWN',)),
]


def iter_systematics(include_nominal=False):

    if include_nominal:
        yield 'NOMINAL'
    for variations in SYSTEMATICS:
        for var in variations:
            yield var


if __name__ == '__main__':

    for variations in SYSTEMATICS:
        if len(variations) == 2:
            print "high: %s low: %s" % (
                    '+'.join(variations[0]),
                    '+'.join(variations[1]))
        else:
            print "high: %s low: NOMINAL" % (
                    '+'.join(variations[0]))
