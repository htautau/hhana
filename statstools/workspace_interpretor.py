# stdlib imports
from cStringIO import StringIO
from collections import OrderedDict

# root/rootpy imports
import ROOT
import rootpy
from rootpy import asrootpy
from rootpy.plotting import Hist
from rootpy.stats.histfactory import HistoSys, split_norm_shape
from rootpy.stats import Workspace
from rootpy.extern.tabulartext import PrettyTable
from rootpy.extern.pyparsing import (Literal, Word, Combine,
                                     Optional, delimitedList,
                                     oneOf, alphas, nums, Suppress)
# local imports
from .import log; log = log[__name__]

class highlighted_string(unicode):
    def __new__(self, content):
        if isinstance(content, basestring):
            return unicode.__new__(self, "\033[93m%s\033[0m"%str(content))
        else:
            return unicode.__new__(self, "\033[93m%0.2f\033[0m"%content)
            
    def __len__(self):
        ESC = Literal('\x1b')
        integer = Word(nums)
        escapeSeq = Combine(ESC + '[' + Optional(delimitedList(integer,';')) + oneOf(list(alphas)))
        return unicode.__len__(Suppress(escapeSeq).transformString(str(self)))        

class prettyfloat(float):
    def __repr__(self):
        #         if self<0:
        #             return stripped_str("\033[93m%0.2f\033[0m"%self)
        #         elif self==0:
        #             return stripped_str("\033[91m%0.2f\033[0m" % self)
        #         else:
        return "%1.1f" % self
    def __str__(self):
        return repr(self)


            
class workspaceinterpretor:
    """
    A class to read and retrieve HSG4-type WS components
    - Parameters:
    - A HSG4 workspace
    """
    # ---------------------------------------
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
        self.get_nuisance_checks(mc, simPdf, obsData, ws)

    # ---------------------------------------
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

    # ------------------------------------------------
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

    # ------------------------------------------------
    def get_nuisance_checks(self, mc, simPdf, obsData, ws):

        poi =  mc.GetParametersOfInterest().first()
        poi.setRange(0., 2.)
        poi.setVal(0.)
        roo_min = asrootpy(ws).fit()
        fitres = roo_min.save()
        minNLL_hat = fitres.minNll()
        log.info( 'minimized NLL: %f'%minNLL_hat)
        ws.saveSnapshot("StartingPoint", simPdf.getParameters(obsData))


        nuisance_params = mc.GetNuisanceParameters()
        params_list = self.nuisance_params(mc)

        minNlls_dict = {}
        for key,_ in params_list.items():
            if 'alpha_ATLAS_JES_Eta_Modelling' not in key:
                continue
            log.info('Scanning parameter %s'%key)
            params_list[key]=False
            nuis_par = nuisance_params.find(key)
            minNlls = []
            for val in xrange(-5, 5, 2):
                log.info( 'Fit with %s = %d'%(key, val) )
                ws.loadSnapshot("StartingPoint")
                nuis_par.setVal(val)
                roo_min = asrootpy(ws).fit(param_const=params_list,print_level=-1)
                nuisance_params.Print('v')
                fitres = roo_min.save()
                minNlls.append(fitres.minNll()-minNLL_hat)
                
            minNlls_dict[key] = minNlls
            nuisance_params[key] = True

        out = StringIO()
        row_template = ['NLL -Nll_hat'] + [val for val in xrange(-5, 5, 2)] 
        table = PrettyTable(row_template)
        for key,list_nuis in minNlls_dict.items():
            pretty_bin_contents=map(prettyfloat,list_nuis)
            table.add_row( [key]+pretty_bin_contents ) 
        print >> out, '\n'
        print >> out, table.get_string(hrules=1)
        log.info(out.getvalue())
        log.info( str(minNlls_dict) )    


    def nuisance_params(self,mc,constant=False):
        nuisIter = mc.GetNuisanceParameters().createIterator()
        params_list = {}
        while True:
            nuis = nuisIter.Next()
            if not nuis:
                break
            params_list[nuis.GetName()] = constant
        return params_list



