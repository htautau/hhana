# root/rootpy imports
from rootpy import ROOT
from ROOT import RooArgList, RooArgSet, RooAddition
from rootpy import asrootpy
from rootpy.plotting import Graph, Hist
# local imports
from . import log; log=log[__name__]

# -------------------------------------
def UncertGraph(hnom, curve_uncert):
    """
    Convert an histogram and a RooCurve
    into a TGraphAsymmError

    Parameters
    ----------
    hnom: TH1F, TH1D,...
        The histogram of nominal values
    curve_uncert: RooCurve
        The uncertainty band around the nominal value
    curve_uncert: RooCurve
    TODO: Improve the handling of the underflow and overflow bins
    """

    graph = Graph(hnom.GetNbinsX())
    # ---------------------------------------------
    for ibin in xrange(1, hnom.GetNbinsX()+1):
        uncerts = []
        for ip in xrange(3, curve_uncert.GetN()-3):
            x, y = ROOT.Double(0.), ROOT.Double(0.)
            curve_uncert.GetPoint(ip, x, y)
            if int(x)==int(hnom.GetBinLowEdge(ibin)):
                uncerts.append(y)
        uncerts.sort()
        for val in uncerts:
            if val in uncerts:
                uncerts.remove(val)
        if len(uncerts)!=2:
            raise RuntimeError('Need exactly two error values and got '+len(uncerts))

        bin_center = 0.5*(hnom.GetBinLowEdge(ibin+1)+hnom.GetBinLowEdge(ibin))
        e_x_low = bin_center-hnom.GetBinLowEdge(ibin)
        e_x_high = hnom.GetBinLowEdge(ibin+1) - bin_center
        bin_content = hnom.GetBinContent(ibin)
        e_y_low = hnom.GetBinContent(ibin)-uncerts[0]
        e_y_high = uncerts[1]-hnom.GetBinContent(ibin) 
        graph.SetPoint( ibin-1, bin_center, bin_content)
        graph.SetPointError(ibin-1, e_x_low, e_x_high, e_y_low, e_y_high)
    # ---------------------------------------------
    return graph


# ----------------------------------------------------
def getPostFitPlottingObjects(mc, obsData, simPdf, fit_res):
    """
    Return a list of histograms and frames
    containing the output plotting objects
    from the workspace post-fit

    parameters:

    mc: ModelConfig
    obsData: RooDataHist object (either real data or asimov or PE)
    simPdf: RooSimultaneousPdf
    fit_res: RooFitResult
    """
    plotting_objects = []
    # --> get the list of categories and iterate over (VBF, Boosted and Rest for HSG4 hadhad)
    catIter = simPdf.indexCat().typeIterator()
    yields = {}
    while True:
        cat = catIter.Next()
        if not cat:
            break
        log.info('============================================')
        log.info('retrieve plotting objects of {0}'.format(cat.GetName()))
        frame, hlist, yields_cat = getFrame(cat, obsData, simPdf, mc, fit_res, compute_yields=True)
        yields[cat.GetName()] = yields_cat
        plotting_objects += [frame]+hlist
    return plotting_objects, yields

# ------------------------------------------------------------------------------
def getFrame(cat, obsData, simPdf, mc, fit_res, error_band_strategy=1, compute_yields=False):
    """
    Build a frame with the different fit components and their uncertainties

    Parameters
    ----------
    cat    : RooCategory
        Category of the simultaneous PDF we are interested in
    obsData: RooDataHist object (either real data or asimov or PE)
    simPdf : RooSimultaneous PDF
    mc : ModelConfig object
    fit_res: RooFitResult
        Result of the fit (covariance matrix, NLL value, ...)
    error_band_strategy : True/False
        True: Use the linear approximation to extract the error band
        False: Use a sampling method
        See http://root.cern.ch/root/html/RooAbsReal.html#RooAbsReal:plotOnWithErrorBand
    verbose: True/False

    TODO: implement a more generic way to retrieve the binning.
    Currently it has to be put in the workspace with the name binWidth_obs_x_{channel}_0
    """

    hlist = []
    if compute_yields:
        yields = {}
    # --> Get the total (signal+bkg) model pdf
    pdftmp = simPdf.getPdf(cat.GetName())
    if not pdftmp:
        raise RuntimeError('Could not retrieve the total pdf ')
    # --> Get the list of observables
    obstmp  = pdftmp.getObservables(mc.GetObservables())
    if not obstmp:
        raise RuntimeError('Could not retrieve the list of observable')
    # --> Get the first (and only) observable (mmc mass for cut based)
    obs  = obstmp.first()
    # --> Get the RooDataHist of the given category
    datatmp = obsData.reduce("{0}=={1}::{2}".format(simPdf.indexCat().GetName(), simPdf.indexCat().GetName(), cat.GetName()))
    datatmp.__class__=ROOT.RooAbsData # --> Ugly fix !!!
    # --> Get the binning width of the category (stored as a workspace variable)
    binWidth = pdftmp.getVariables().find('binWidth_obs_x_{0}_0'.format(cat.GetName()))
    if not binWidth:
        raise RuntimeError('Could not retrieve the binWidth')
    # --> parameter of interest (mu=sigma/sigma_sm)
    poi =  mc.GetParametersOfInterest().first()
    poi_fitted_val = poi.getVal()
    log.info('POI: {0} = {1} +/- {2}'.format(poi.GetName(), poi.getVal(), poi.getError()))
            
    # --> Create the data histogram
    hist_data = asrootpy(datatmp.createHistogram("hdata_"+cat.GetName(), obs))
    hist_data.name = "hdata_{0}".format(cat.GetName())
    hist_data.title = ""
    hlist.append(hist_data)
    #--> compute the data yields
    if compute_yields:
        Yield_data = hist_data.Integral()
        yields['Data'] = (Yield_data, 0)

    # --> Create the frame structure from the observable
    frame = obs.frame()
    frame.SetName(cat.GetName())
    datatmp.plotOn(frame,
                   ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson),
                   ROOT.RooFit.Name("Data"),
                   ROOT.RooFit.MarkerSize(1))

    # --> Create the signal histogram template
    hist_sig = hist_data.Clone('h_TotalSignal_{0}'.format(cat.GetName()))
    hist_sig.Reset()
    signal_comps = RooArgList()
    bkg_comps = RooArgList()
    # --> get the list of components (hadhad HSG4: QCD,Other,Ztautau, Signal_Z, Signal_W, Signal_gg, Signal_VBF)
    # --> and iterate over 
    pdfmodel = pdftmp.getComponents().find(cat.GetName()+'_model')
    funcListIter = pdfmodel.funcList().iterator()
    while True:
        comp = funcListIter.Next()
        if not comp:
            break

        name = comp.GetName().replace('L_x_', '').split('_')[0]
        log.info('Scan component {0}'.format(comp.GetName()))
        hist_comp = asrootpy(comp.createHistogram(cat.GetName()+"_"+comp.GetName(), obs, ROOT.RooFit.Extended(False)))
        hist_comp.name = 'h_{0}_{1}'.format(name, cat.GetName())
        hist_comp.title = ''
        hlist.append(hist_comp)

        if 'Signal' in comp.GetName():
            signal_comps.add(comp)
            hist_sig.Add(hist_comp)
        else:
            bkg_comps.add(comp)
        Integral_comp = comp.createIntegral(RooArgSet(obs))
        Yield_comp = Integral_comp.getVal() * binWidth.getVal()
        log.info('Integral = {0}'.format(Yield_comp))
        if Yield_comp==0:
            raise RuntimeError('Yield integral is wrong !!')


        # --> Add the components to the frame but in an invisible way
        pdfmodel.plotOn(frame,
                        ROOT.RooFit.Components(comp.GetName()),
                        ROOT.RooFit.Normalization(Yield_comp, ROOT.RooAbsReal.NumEvent),
                        ROOT.RooFit.Name("NoStacked_"+comp.GetName()),
                        ROOT.RooFit.Invisible())
        if fit_res:
            # --> Add the components uncertainty band 
            comp.plotOn(frame,
                        ROOT.RooFit.Normalization(1, ROOT.RooAbsReal.RelativeExpected),
                        ROOT.RooFit.VisualizeError(fit_res, 1, error_band_strategy),
                        ROOT.RooFit.Name("FitError_AfterFit_"+comp.GetName()),
                        ROOT.RooFit.Invisible())
            if compute_yields:
                Yield_comp_err = Integral_comp.getPropagatedError(fit_res)* binWidth.getVal()
                yields[comp.GetName()] = (Yield_comp, Yield_comp_err)

    hlist.append(hist_sig)
    # --> total signal yields
    if compute_yields:
        signal_tot_comp = RooAddition('pdf_sum_sig', 'pdf_sum_sig', signal_comps)
        Integral_signal = signal_tot_comp.createIntegral(RooArgSet(obs))
        yields_sig_tot = Integral_signal.getVal()*binWidth.getVal()
        yields_sig_tot_err = Integral_signal.getPropagatedError(fit_res)* binWidth.getVal()
        yields['TotalSignal'] = (yields_sig_tot, yields_sig_tot_err)
        bkg_tot_comp = RooAddition('pdf_sum_bkg', 'pdf_sum_bkg', bkg_comps)
        Integral_bkg = bkg_tot_comp.createIntegral(RooArgSet(obs))
        yields_bkg_tot = Integral_bkg.getVal()*binWidth.getVal()
        yields_bkg_tot_err = Integral_bkg.getPropagatedError(fit_res)* binWidth.getVal()
        yields['bkg'] = (yields_bkg_tot, yields_bkg_tot_err)
        log.info('Bkg Total Yield: {0}'.format(yields_bkg_tot))

    # --> bkg+signal PDF central value and error
    Integral_total = pdfmodel.createIntegral(RooArgSet(obs))
    Yield_total = Integral_total.getVal() * binWidth.getVal()
    log.info('Postfit Total Yield: {0}'.format(Yield_total))
    hist_bkg_plus_sig = asrootpy(pdfmodel.createHistogram("hbkg_plus_sig_"+cat.GetName(), obs, ROOT.RooFit.Extended(False)))
    hist_bkg_plus_sig.name = 'hbkg_plus_sig_'+cat.GetName()
    hist_bkg_plus_sig.title = ''
    hist_bkg_plus_sig.Scale(Yield_total)
    hlist.append(hist_bkg_plus_sig)

    # --> Add the components to the frame but in an invisible way
    pdftmp.plotOn(frame,
                   ROOT.RooFit.Normalization(Yield_total, ROOT.RooAbsReal.NumEvent),
                   ROOT.RooFit.Name("Bkg_plus_sig"))

    if fit_res:
        pdftmp.plotOn(frame,
                      ROOT.RooFit.VisualizeError(fit_res, 1, error_band_strategy),
                      ROOT.RooFit.Normalization(1, ROOT.RooAbsReal.RelativeExpected),
                      ROOT.RooFit.Name("FitError_AfterFit"),
                      ROOT.RooFit.FillColor(ROOT.kOrange),
                      ROOT.RooFit.LineWidth(2),
                      ROOT.RooFit.LineColor(ROOT.kBlue))

    # --> bkg only PDF central value and error
    poi.setVal(0.)
    Integral_bkg_total = pdfmodel.createIntegral(RooArgSet(obs))
    Yield_bkg_total = Integral_bkg_total.getVal() * binWidth.getVal()

    hist_bkg = pdfmodel.createHistogram("hbkg_"+cat.GetName(), obs, ROOT.RooFit.Extended(False))
    hist_bkg.Scale(Yield_bkg_total)
    hist_bkg.SetName("hbkg_"+cat.GetName())
    hist_bkg.SetTitle("")
    hlist.append(hist_bkg)
    pdftmp.plotOn(frame,
                  ROOT.RooFit.Normalization(Yield_bkg_total, ROOT.RooAbsReal.NumEvent),
                  ROOT.RooFit.Name("Bkg"),
                  ROOT.RooFit.LineStyle(ROOT.kDashed))
    if fit_res:
        pdftmp.plotOn(frame,
                      ROOT.RooFit.VisualizeError(fit_res, 1, error_band_strategy),
                      ROOT.RooFit.Normalization(1, ROOT.RooAbsReal.RelativeExpected),
                      ROOT.RooFit.Name("FitError_AfterFit_Mu0"),
                      ROOT.RooFit.FillColor(ROOT.kOrange),
                      ROOT.RooFit.LineWidth(2),
                      ROOT.RooFit.LineColor(ROOT.kBlue))
        #if compute_yields:
            # Yield_bkg_total_err = Integral_bkg_total.getPropagatedError(fit_res)*binWidth.getVal()
            # yields['bkg'] = (Yield_bkg_total, Yield_bkg_total_err)

    poi.setVal(poi_fitted_val)
    if compute_yields:
        return frame, hlist, yields
    else:
        return frame, hlist
