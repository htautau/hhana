# stdlib imports
from cStringIO import StringIO
from collections import OrderedDict
# root/rootpy imports
import ROOT
from rootpy import asrootpy
from rootpy.plotting import Hist
from rootpy.stats.histfactory import HistoSys, split_norm_shape
from rootpy.extern.tabulartext import PrettyTable

# local imports
from .import log; log = log[__name__]

class prettyfloat(float):
    def __repr__(self):
        #         if self<0:
        #             return "\033[93m%0.2f\033[0m" % self
        #         elif self==0:
        #             return "\033[91m%0.2f\033[0m" % self
        #         else:
        return "%1.1f" % self
    def __str__(self):
        return repr(self)


            
class workspaceinterpretor:
    """A class to read and retrieve HSG4-type WS components"""
    def __init__(self,ws):
        obsData = ws.data('obsData')
        # --> Get the Model Config object
        mc = ws.obj("ModelConfig")
        # --> Get the simultaneous PDF
        simPdf = mc.GetPdf()
        # --> Get the dictionary of nominal hists
        self.hists = self.get_nominal_hists_array(obsData, mc, simPdf)
        # --> Start the log 
        log.info('\n')
        log.info('------- START OF THE SANITY CHECK -----------')
        log.info('------- START OF THE SANITY CHECK -----------')
        log.info('------- START OF THE SANITY CHECK -----------')
        log.info('\n')
        for cat,hlist in self.hists.items():
            self.PrintHistsContents(cat,hlist)
        self.get_nuisance_params(mc, simPdf)

    def get_nominal_hists_array(self, obsData, mc, simPdf):
        hists_array={}
        # --> get the list of categories index and iterate over
        catIter = simPdf.indexCat().typeIterator()
        while True:
            cat = catIter.Next()
            if not cat:
                break
            log.info("Scanning category {0}".format(cat.GetName()))
            hists_comp = []

            # --> Get the total (signal+bkg) model pdf
            pdftmp = simPdf.getPdf(cat.GetName())
            # --> Get the list of observables
            obstmp  = pdftmp.getObservables(mc.GetObservables())
            # --> Get the first (and only) observable (mmc mass for cut based)
            obs = obstmp.first()
            # --> Get the parameter of interest
            poi =  mc.GetParametersOfInterest().first()

            # --> Create the data histogram
            datatmp = obsData.reduce( "{0}=={1}::{2}".format(simPdf.indexCat().GetName(),simPdf.indexCat().GetName(),cat.GetName()) )
            datatmp.__class__=ROOT.RooAbsData # --> Ugly fix !!!
            log.info("Retrieve the data histogram")
            hists_comp.append( ('data', asrootpy(datatmp.createHistogram('',obs))) )

            # --> Create the total model histogram
            log.info("Retrieve the total background")
            poi.setVal(0.0)
            hists_comp.append( ('background', asrootpy(pdftmp.createHistogram("cat_%s"%cat.GetName(),obs))) )

            # --> Create the total model histogram
            log.info("Retrieve the total model (signal+background)")
            poi.setVal(1.0)
            hists_comp.append( ('background+signal', asrootpy(pdftmp.createHistogram("model_cat_%s"%cat.GetName(),obs))) )
            poi.setVal(0.0)

            comps = pdftmp.getComponents()
            compsIter = comps.createIterator()
            while True:
                comp = compsIter.Next()
                if not comp:
                    break
                # ---> loop only over the nominal histograms
                if 'nominal' not in comp.GetName():
                    continue

                log.info('Retrieve component {0}'.format(comp.GetName()))
                hists_comp.append( (comp.GetName()[:14], asrootpy(comp.createHistogram('%s_%s'%(cat.GetName(),comp.GetName()),obs))) )
            hists_array[cat.GetName()]=hists_comp
        return hists_array

    def get_nuisance_params(self, mc, simPdf):
        nuisIter = mc.GetNuisanceParameters().createIterator()
        while True:
            nuis = nuisIter.Next()
            if not nuis:
                break
            log.info( '%s: %1.2f<%1.2f<%1.2f'%(nuis.GetName(),nuis.getAsymErrorLo(),nuis.getVal(),nuis.getAsymErrorHi()))

        comps = simPdf.getComponents()
        compIter = comps.createIterator()
        while True:
            comp = compIter.Next()
            if not comp:
                break
            log.info( comp.GetName() )


        vars = simPdf.getVariables()
        varIter = vars.createIterator()
        while True:
            var = varIter.Next()
            if not var:
                break
            log.info( '%s: %1.2f < %1.2f'%(var.GetName()) )
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



    def PrintHistsContents(self,cat,hlist):
        log.info(cat)
        row_template = [cat]+list(hlist[0][1].bins_range())
        out = StringIO()
        table = PrettyTable(row_template)
        for pair in hlist:
            pretty_bin_contents=map(prettyfloat,pair[1].y())
            table.add_row( [pair[0]]+pretty_bin_contents ) 
        print >> out, '\n'
        print >> out, table.get_string(hrules=1)
        log.info(out.getvalue())




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





