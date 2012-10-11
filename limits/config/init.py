#!/usr/bin/env python

categories = ['ggf', 'boosted', 'vbf']
fitmethod = 'trackfit'
masses = range(100, 155, 5)
lumi = '1.'
lumi_rel_err = '0.039'

channels = ('hh', 'elh', 'mulh')

channel_templates = {}
comb_templates = {}
for channel in channels:
    channel_templates[channel] = ''.join(
            open('template_channel_%s.xml' % channel, 'r').readlines())
    comb_templates[channel] = ''.join(
            open('template_combination_%s.xml' % channel, 'r').readlines())

#template_combination_all = ''.join(open('template_combination_all_hh.xml', 'r').readlines())
full_comb_template = ''.join(
        open('template_combination_all.xml', 'r').readlines())

for mass in masses:
    for channel in channels:
        for category in categories:
            with open('%(channel)s_channel_%(category)s_%(mass)d.xml' % locals(), 'w') as f:
                f.write(channel_templates[channel] % locals())
            #with open('hh_combination_%(category)s_%(mass)d.xml' % locals(), 'w') as f:
            #    f.write(template_combination % locals())
        with open('%(channel)s_combination_%(mass)d.xml' % locals(), 'w') as f:
            f.write(comb_templates[channel] % locals())
    with open('all_combination_%(mass)d.xml' % locals(), 'w') as f:
            f.write(full_comb_template % locals())
