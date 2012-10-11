#!/usr/bin/env python

categories = ['ggf', 'boosted', 'vbf']
fitmethod = 'trackfit'
masses = range(100, 155, 5)
lumi = '1.'
lumi_rel_err = '0.039'
workspace = 'SYSTEMATICS'

channels = ('hh', 'elh', 'mulh')
channel_templates = {}
for channel in channels:
    channel_templates[channel] = ''.join(
            open('template_channel_%s.xml' % channel, 'r').readlines())

comb_channels = ('hh', 'lh')
comb_templates = {}
comb_category_templates = {}
for channel in comb_channels:
    comb_templates[channel] = ''.join(
            open('template_combination_%s.xml' % channel, 'r').readlines())
    comb_category_templates[channel] = ''.join(
            open('template_combination_category_%s.xml' % channel, 'r').readlines())

full_comb_template = ''.join(
        open('template_combination_all.xml', 'r').readlines())
full_comb_category_template = ''.join(
        open('template_combination_category_all.xml', 'r').readlines())

for mass in masses:
    for channel in channels:
        for category in categories:
            with open('%(channel)s_channel_%(category)s_%(mass)d.xml' % locals(), 'w') as f:
                f.write(channel_templates[channel] % locals())

    for channel in comb_channels:
        with open('%(channel)s_combination_%(mass)d.xml' % locals(), 'w') as f:
            f.write(comb_templates[channel] % locals())
        for category in categories:
            with open('%(channel)s_combination_%(category)s_%(mass)d.xml' % locals(), 'w') as f:
                f.write(comb_category_templates[channel] % locals())

    with open('all_combination_%(mass)d.xml' % locals(), 'w') as f:
            f.write(full_comb_template % locals())

    for category in categories:
        with open('all_combination_%(category)s_%(mass)d.xml' % locals(), 'w') as f:
                f.write(full_comb_category_template % locals())
