#!/usr/bin/env python

from rootpy.io import root_open
from rootpy.utils.path import mkdir_p
from rootpy.interactive import wait
import os
import sys
import ROOT


def draw_category():
    pass

def draw_workspace(workspace, output_path='.'):
    if not os.path.exists(output_path):
        mkdir_p(output_path)
    data = workspace.data('obsData')
    config = workspace.obj('ModelConfig')
    simpdf = config.pdf
    if isinstance(simpdf, ROOT.RooSimultaneous):
        index_category = simpdf.index_category
        for category in simpdf:
            pdf = simpdf.pdf(category)
            # get first observable
            obs = pdf.observables(config.observables).first()
            # total model histogram
            model_hist = pdf.createHistogram(
                'cat_{0}'.format(category.name), obs)
            # create the data histogram
            data_category = data.reduce('{0}=={1}::{2}'.format(
                index_category.name, index_category.name, category.name))
            data_hist = data_category.createHistogram(
                'hdata_cat_{0}'.format(category.name), obs)
            # get the background components
            comp_hists = []
            components = pdf.components()
            for component in components:
                if 'nominal' not in component.GetName():
                    continue
                comp_hists.append(
                    component.createHistogram(
                        '{0}_{1}'.format(
                            category.name, component.GetName()), obs))


def draw_file(filename, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(filename)[0]
    if os.path.isfile(output_path):
        raise ValueError("a file already exists at {0}".format(output_path))
    with root_open(filename) as file:
        for dirpath, dirnames, objectnames in file.walk(class_pattern='RooWorkspace'):
            for name in objectnames:
                workspace = file['{0}/{1}'.format(dirpath, name)]
                draw_workspace(workspace, os.path.join(output_path, dirpath, name))

if __name__ == '__main__':
    from rootpy.extern.argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    for filename in args.files:
        draw_file(filename, args.output)
