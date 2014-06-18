# python imports
import os
import pickle
from multiprocessing import Process

# root/rootpy imports
import ROOT
from ROOT import RooArgSet, RooAddition, RooPlot
from rootpy.memory.keepalive import keepalive
from rootpy import asrootpy
from rootpy.io import root_open
from rootpy.stats.collection import ArgList
from rootpy.utils.lock import lock

# local imports
from . import log; log=log[__name__]

ERROR_BAND_STRATEGY = 1


class Component(object):
    """
    Class to decorate a component pdf of a fit model
    (add a name, integral, integral_err and an histo)

    Parameter
    ---------
    pdf: RooAbsPdf object
    """
    def __init__(self, pdf):
        self.name = pdf.GetName()
        self.pdf = pdf
        self.integral = 0
        self.integral_err = 0
        self.hist = None

class FitModel(object):
    """
    Class to retrieve and compute relevant
    information from a category in a workspace

    Parameters
    ----------
    - workspace: rootpy Workspace)
    - category: rootpy Category
    """
    def __init__(self, workspace, category):
        self.ws = workspace
        self.cat = category
        self.mc = self.ws.obj('ModelConfig')
        self.index_cat = self.mc.GetPdf().index_category
        self.obsData = self.ws.data('obsData')
        self.pdf = self.mc.GetPdf().pdf(self.cat)
        self._obs = self.pdf.getObservables(self.mc.GetObservables()).first()
        self._frame = self._obs.frame()
        self._frame.SetName('{0}'.format(self.cat.name))
        keepalive(self._frame, self._frame.GetName())
        self._components = [Component(comp) for comp in self.iter_pdf_components()]
        self._signal = Component(RooAddition('sum_sig_{0}'.format(self.cat.name),
                                             'sum_sig_{0}'.format(self.cat.name),
                                             self.signal_pdf_components()))
        self._background = Component(RooAddition('sum_bkg_{0}'.format(self.cat.name),
                                                 'sum_bkg_{0}'.format(self.cat.name),
                                                 self.background_pdf_components()))

    @property
    def pdfmodel(self):
        return self.pdf.getComponents().find(self.cat.name+'_model')

    @property
    def data(self):
        return self.obsData.reduce("{0}=={0}::{1}".format(self.index_cat.name, self.cat.name))

    @property
    def data_hist(self):
        hist_data = asrootpy(self.data.createHistogram("h_data_"+self.cat.name, self.obs))
        hist_data.name = "h_data_{0}".format(self.cat.name)
        hist_data.title = ''
        return hist_data

    @property
    def obs(self):
        return self._obs

    @property
    def binwidth(self):
        return self.pdf.getVariables().find('binWidth_obs_x_{0}_0'.format(self.cat.name))

    @property
    def frame(self):
        return self._frame

    def iter_pdf_components(self):
        iterator = self.pdfmodel.funcList().createIterator()
        component = iterator.Next()
        while component:
            yield component
            component = iterator.Next()

    def signal_pdf_components(self):
        comps = ArgList('signal_{0}'.format(self.cat.name))
        for comp in self.components:
            if 'Signal' in comp.name:
                comps.add(comp.pdf)
        return comps

    def background_pdf_components(self):
        comps = ArgList('background_{0}'.format(self.cat.name))
        for comp in self.components:
            if not 'Signal' in comp.name:
                comps.add(comp.pdf)
        return comps

    @property
    def components(self):
        return self._components

    @property
    def signal(self):
        return self._signal

    @property
    def background(self):
        return self._background

def process_fitmodel(model, fit_res):
    """
    Compute histograms and frame of the FitModel
    according to a given RooFitResult
    """
    model.data.plotOn(model.frame,
                      ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson),
                      ROOT.RooFit.Name("Data"), ROOT.RooFit.MarkerSize(1))
    components = [comp for comp in model.components]+[model.signal, model.background]
    for comp in components:
        log.info('Scan component {0}'.format(comp.name))
        name = comp.name.replace('L_x_', '').split('_')[0]
        if name == 'Signal':
            name += '_'+comp.name.replace('L_x_', '').split('_')[1]
        name += '_'+model.cat.name
        if 'sum' in comp.name:
            name = comp.name

        comp.hist = asrootpy(comp.pdf.createHistogram('h_{0}'.format(name), model.obs,
                                                      ROOT.RooFit.Extended(False)))
        comp.hist.name = 'h_{0}'.format(name)
        comp.hist.title = ''
        Integral_comp = comp.pdf.createIntegral(RooArgSet(model.obs))
        comp.integral = Integral_comp.getVal() * model.binwidth.getVal()
        if fit_res:
            comp.integral_err = Integral_comp.getPropagatedError(fit_res)*model.binwidth.getVal()
            # --> Add the components uncertainty band
            comp.pdf.plotOn(model.frame,
                            ROOT.RooFit.Normalization(1, ROOT.RooAbsReal.RelativeExpected),
                            ROOT.RooFit.VisualizeError(fit_res, 1, ERROR_BAND_STRATEGY),
                            ROOT.RooFit.Name('FitError_AfterFit_{0}'.format(name)))
        log.info('{0}: Integral = {1}+/-{2}'.format(comp.hist.name, comp.integral, comp.integral_err))

class ModelCalculator(Process):
    """
    Class to compute several FitModel in parallel.
    This is relevant when running over a ws with several categories

    Parameters
    ----------
    model: FitModel
    fit_res: RooFitResult to be applied
    root_name: Name of the rootfile where histograms and frames are stored
    pickle_name: Name of the pickle file where yields are stored
    """
    def __init__(self, 
                 file, 
                 workspace, 
                 cat, 
                 fit_res, 
                 root_name, 
                 pickle_name): 
        super(ModelCalculator, self).__init__()
        self.file = file
        self.ws = workspace
        self.cat = cat
        self.fit_res = fit_res
        self.root_name = root_name
        self.pickle_name = pickle_name

    def run(self):
        model = FitModel(self.ws, self.cat)
        process_fitmodel(model, self.fit_res)
        components = [
            comp for comp in model.components] + [
            model.signal, model.background]
        with lock(self.root_name):
            with root_open(self.root_name, 'update') as fout:
                log.info('{0}'.format(model.frame))
                model.frame.Write()
                for comp in components:
                    log.info('{0}: {1}'.format(comp.hist, comp.hist.Integral()))
                    comp.hist.Write()
        with lock(self.pickle_name):
            with open(self.pickle_name) as pickle_file:
                yields = pickle.load(pickle_file)
                yields_cat = {}
                for comp in components:
                    yields_cat[comp.name] = (comp.integral, comp.integral_err)
                yields_cat['Data'] = (model.data_hist.Integral(),)
                yields[model.cat.name] = yields_cat
            with open(self.pickle_name, 'w') as pickle_file:
                pickle.dump(yields, pickle_file)
