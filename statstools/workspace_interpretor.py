# stdlib imports
import os, sys
import shutil
import math

# root/rootpy imports
import rootpy
rootpy.log.basic_config_colorized()
import ROOT
from rootpy.plotting import Hist
from rootpy.stats.histfactory import HistoSys, split_norm_shape

# local imports
from .import log; log = log[__name__]
from mva.categories import CATEGORIES


class workspaceinterpretor:
    """A class to read and retrieve HSG4-type WS components"""
    def __init__(self,ws):
        self.hist_nuis = {}
        self.hist_comp_cat = {'data':{}, 'total':{}, 'Z':{}, 'Fakes':{}, 'Others':{}}

        obsData = ws.data('obsData')
        #         obsData.Print()
        # --> Get the Model Config object
        mc = ws.obj("ModelConfig")
        #         mc.Print()
        # --> Get the simultaneous PDF
        simPdf = mc.GetPdf()
        # --> get the list of categories index and iterate over
        catIter = simPdf.indexCat().typeIterator()
        while True:
            cat = catIter.Next()
            if not cat:
                break
            log.info("Scanning category {0}".format(cat.GetName()))

            # --> Get the total (signal+bkg) model pdf
            pdftmp = simPdf.getPdf(cat.GetName())
            # --> Get the list of observables
            obstmp  = pdftmp.getObservables(mc.GetObservables())
            # --> Get the first (and only) observable (mmc mass for cut based)
            obs = obstmp.first()

            # --> Create the total model histogram
            self.hist_comp_cat["total"][cat.GetName()] = pdftmp.createHistogram("cat_"+cat.GetName(),obs)
            log.info("Retrieve the total model (signal+background)")

            # --> Create the data histogram
            datatmp = obsData.reduce( "{0}=={1}::{2}".format(simPdf.indexCat().GetName(),simPdf.indexCat().GetName(),cat.GetName()) )
            datatmp.__class__=ROOT.RooAbsData # --> Ugly fix !!!
            log.info("Retrieve the data histogram")
            self.hist_comp_cat["data"][cat.GetName()] = datatmp.createHistogram("hdata_%s"%cat.GetName(),obs)
            self.hist_comp_cat["data"][cat.GetName()].Print('all')
            comps = pdftmp.getComponents()
            compsIter = comps.createIterator()
            while True:
                comp = compsIter.Next()
                if not comp:
                    break
                # ---> loop only over the nominal histograms
                if 'nominal' not in comp.GetName():
                    continue

                log.info('Scanning component {0}'.format(comp.GetName()))
                if ('Ztautau' in comp.GetName()) or ('Ztt' in comp.GetName()):
                    log.info('Retrieve the Z component')
                    self.hist_comp_cat["Z"][cat.GetName()] = comp.createHistogram('%s_%s'%(cat.GetName(),comp.GetName()),obs)
                    self.hist_comp_cat["Z"][cat.GetName()].Print('all')
                if ('Fakes' in comp.GetName()):
                    log.info('Retrieve the Fakes component')
                    self.hist_comp_cat["Fakes"][cat.GetName()] = comp.createHistogram("%s_%s"%(cat.GetName(),comp.GetName()),obs)
                    self.hist_comp_cat["Fakes"][cat.GetName()].Print('all')
                if ('Others' in comp.GetName()):
                    log.info('Retrieve the others background component')
                    self.hist_comp_cat["Others"][cat.GetName()] = comp.createHistogram("%s_%s"%(cat.GetName(),comp.GetName()),obs)
                    self.hist_comp_cat["Others"][cat.GetName()].Print('all')


            self.PrintYields(cat.GetName())

    def PrintHistsContents(self,cat):
        log.info(cat)


    def PrintYields(self,cat):
        log.info('data:   {0}'.format(self.hist_comp_cat["data"]  [cat].Integral()))
        log.info('total:  {0}'.format(self.hist_comp_cat["total"] [cat].Integral())) 
        log.info('Z:      {0}'.format(self.hist_comp_cat["Z"]     [cat].Integral()))     
        log.info('Fakes:  {0}'.format(self.hist_comp_cat["Fakes"] [cat].Integral()))     
        log.info('Others: {0}'.format(self.hist_comp_cat["Others"][cat].Integral()))     



#             # --> Iterate over nuisance params
#             nuisIter = mc.GetNuisanceParameters().createIterator()
#             while True:
#                 nuis = nuisIter.Next()
#                 if not nuis: break
#                 #                 print 'Nuisance parameter: '+nuis.GetName()
#                 #                 nuis.Print()
#                 if "gamma_stat" in nuis.GetName(): continue
#                 if "ATLAS_norm" in nuis.GetName(): continue
#                 if "ATLAS_sampleNorm" in nuis.GetName(): continue
#                 #                 print '--> plot this nuisance parameter: '+nuis.GetName()
#                 self.hist_nuis[cat.GetName()+"_"+nuis.GetName()] = pdftmp.createHistogram( "channel_"+cat.GetName()+"_nuis_"+nuis.GetName(),obs,ROOT.RooFit.YVar(nuis) )

 
    #  Get the components amd the shape nuissance parameters
#     hist_comp_nom = {}
#     hist_nuis     = {}

#     args = pdftmp.getComponents()
#     argIter = args.createIterator()
#     while True:
#         comp = argIter.Next()
#         if not comp: break
#         comp.Print()
#         if 'nominal' in comp.GetName():
#             hist_comp_nom[comp.GetName()] = comp.createHistogram(cat.GetName()+"_"+comp.GetName(),obs)
#         if "alpha" in comp.GetName()[0:5]:
#             comp.Print(),comp.getVariables().Print()
#         comp.Print()
#     canvas_hist[ cat.GetName() ] = ROOT.TCanvas()
#     canvas_hist[ cat.GetName() ] .cd()
#     hist.Draw("")
#     for h in hist_comp_nom:
#         print h
#         hist_comp_nom[h].Draw("same")





