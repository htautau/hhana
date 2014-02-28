
from array import array

# root/rootpy imports
import rootpy
import ROOT
from rootpy.plotting import Graph

# -------------------------------------
def UncertGraph( hnom, curve_uncert ):
    """
    Convert an histogram and a RooCurve
    into a TGraphAsymmError

    Parameters
    ----------
    hnom: TH1F,TH1D,...
        The histogram of nominal values
    curve_uncert: RooCurve
        The uncertainty band around the nominal value
    """

    graph = Graph( hnom.GetNbinsX() )
    # ---------------------------------------------
    for ibin in xrange(1,hnom.GetNbinsX()+1):
        uncerts = []
        for ip in xrange(3,curve_uncert.GetN()-3):
            x,y = ROOT.Double(0.),ROOT.Double(0.)
            curve_uncert.GetPoint(ip,x,y)
            if int(x)==int(hnom.GetBinLowEdge(ibin)):
                #                 print ip,int(x),y
                uncerts.append(y)
        uncerts.sort()
        for val in uncerts:
            if val in uncerts:
                uncerts.remove(val)
        print uncerts
        if len(uncerts)!=2:
            raise RuntimeError( 'Need exactly two error values and got '+len(uncerts) )

        bin_center = 0.5*(hnom.GetBinLowEdge(ibin+1)+hnom.GetBinLowEdge(ibin))
        e_x_low = bin_center-hnom.GetBinLowEdge(ibin)
        e_x_high = hnom.GetBinLowEdge(ibin+1) - bin_center
        bin_content = hnom.GetBinContent(ibin)
        e_y_low = hnom.GetBinContent(ibin)-uncerts[0]
        e_y_high = uncerts[1]-hnom.GetBinContent(ibin) 
        graph.SetPoint( ibin-1, bin_center, bin_content)
        graph.SetPointError( ibin-1,e_x_low,e_x_high,e_y_low,e_y_high)
    # ---------------------------------------------
    return graph



# ------------------------------
def Fit_WS(workspace):
    """
    Fit the WS and compute the histograms and TGraphAssymErrors
    for the final plot drawing

    Parameters
    ----------

    workspace : RooWorkspace
        HSG4 like workspace
    """

    # --> Get the Model Config object
    mc = workspace.obj("ModelConfig")
    if not mc:
        raise RuntimeError('Could not retrieve the ModelConfig object')
    mc.GetParametersOfInterest().first().setVal(1)
#     fit_res = 0
    roo_min = workspace.fit()
    fit_res = roo_min.save()
    fit_res.Print()


    out_file = ROOT.TFile('frames.root','RECREATE')

    # --> Get the data distribution
    obsData = workspace.data('obsData')
    if not obsData:
        raise RuntimeError('Could not retrieve the data histograms')
    # --> Get the simultaneous PDF
    simPdf = mc.GetPdf()
    if not simPdf:
        raise RuntimeError('Could not retrieve the simultaneous pdf')
        
    # --> get the list of categories and iterate over (VBF,Boosted and Rest for HSG4 hadhad)
    catIter = simPdf.indexCat().typeIterator()
    while True:
        cat = catIter.Next()
        if not cat: break
        frame,hlist = getFrame(cat,obsData,simPdf,mc,fit_res)
        frame.Write( cat.GetName() )
        for h in hlist:
            h.Write()
    out_file.Close()




def getFrame(cat,obsData,simPdf,mc,fit_res,error_band_strategy=1,verbose=False):
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
        Result of the fit (covariance matrix, NLL value,...)
    error_band_strategy : True/False
        True: Use the linear approximation to extract the error band
        False: Use a sampling method
        See http://root.cern.ch/root/html/RooAbsReal.html#RooAbsReal:plotOnWithErrorBand
    verbose: True/False
    """
    
    hlist = []

    # --> Get the total (signal+bkg) model pdf
    pdftmp = simPdf.getPdf( cat.GetName() )
    if not pdftmp:
        raise RuntimeError('Could not retrieve the total pdf ')
    # --> Get the list of observables
    obstmp  = pdftmp.getObservables( mc.GetObservables() )
    if not obstmp:
        raise RuntimeError('Could not retrieve the list of observable')
    # --> Get the first (and only) observable (mmc mass for cut based)
    obs  = obstmp.first()
    # --> Get the RooDataHist of the given category
    datatmp = obsData.reduce( "{0}=={1}::{2}".format(simPdf.indexCat().GetName(),simPdf.indexCat().GetName(),cat.GetName()) )
    datatmp.__class__=ROOT.RooAbsData # --> Ugly fix !!!
    # --> Get the binning width of the category (stored as a workspace variable)
    binWidth = pdftmp.getVariables().find( 'binWidth_obs_x_{0}_0'.format( cat.GetName() ) )
    if not binWidth:
        raise RuntimeError('Could not retrieve the binWidth')
            
    # --> Create the data histogram
    hist_data = datatmp.createHistogram("hdata_"+cat.GetName(),obs)
    hlist.append(hist_data)

    # --> Create the frame structure from the observable
    frame = obs.frame()
    datatmp.plotOn( frame,
                    ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson),
                    ROOT.RooFit.Name("Data"),
                    ROOT.RooFit.MarkerSize(1)
                    )
    # --> get the list of components (hadhad HSG4: QCD,Other,Ztautau, Signal_Z, Signal_W, Signal_gg, Signal_VBF)
    # --> and iterate over 
    pdfmodel = pdftmp.getComponents().find( cat.GetName()+'_model' )
    funcListIter =  pdfmodel.funcList().iterator()
    while True:
        comp = funcListIter.Next()
        if not comp:
            break
        hist_comp = comp.createHistogram( cat.GetName()+"_"+comp.GetName(), obs, ROOT.RooFit.Extended(False) )
        hlist.append(hist_comp)

        Integral_comp = comp.createIntegral(ROOT.RooArgSet(obs))
        Yield_comp = Integral_comp.getVal() * binWidth.getVal()

        # ------------------------------------------------
        # ------------------------------------------------
#         components_set = comp.components()
#         components_set.Print('v')
#         components_set_it = components_set.createIterator()
#         while True:
#             current = components_set_it.Next()
#             if not components_set_it:
#                 break
#             if isinstance(current,ROOT.RooProduct): 
#                 current.Print('v')
        #         comp.Print('v')
        # ------------------------------------------------
        # ------------------------------------------------
        print comp.GetName()
        print 'YIELD=',Yield_comp,Integral_comp.getVal(),binWidth.getVal()
        if Yield_comp==0:
            raise RuntimeError( 'Yield integral is wrong !!')



        # --> Add the components to the frame but in an invisible way
        pdfmodel.plotOn( frame,
                         ROOT.RooFit.Components(comp.GetName()),
                         ROOT.RooFit.Normalization(Yield_comp,ROOT.RooAbsReal.NumEvent),
                         ROOT.RooFit.Name("NoStacked_"+comp.GetName()),
                         ROOT.RooFit.Invisible()
                         )
        if fit_res:
            # --> Add the components uncertainty band 
            comp.plotOn( frame,
                         ROOT.RooFit.Normalization(1,ROOT.RooAbsReal.RelativeExpected ),
                         ROOT.RooFit.VisualizeError(fit_res,1,error_band_strategy),
                         ROOT.RooFit.Name("FitError_AfterFit_"+comp.GetName()),
                         ROOT.RooFit.Invisible()
                         )
            if verbose:
                Yield_comp_err = Integral_comp.getPropagatedError( fit_res )* binWidth.getVal()
                print comp.GetName(),':\t',Yield_comp,' +/- ',Yield_comp_err




    # --> parameter of interest (mu=sigma/sigma_sm)
    poi =  mc.GetParametersOfInterest().first()

    # --> bkg+signal PDF error
    hist_bkg_plus_sig = pdftmp.createHistogram("hbkg_plus_sig_"+cat.GetName(),obs,ROOT.RooFit.Extended(False))
    hlist.append(hist_bkg_plus_sig)
    Integral_total = pdftmp.createIntegral(ROOT.RooArgSet(obs))
    Yield_total = Integral_total.getVal() * binWidth.getVal()
    # --> Add the components to the frame but in an invisible way
    pdftmp.plotOn( frame,
                   ROOT.RooFit.Normalization(Yield_total,ROOT.RooAbsReal.NumEvent),
                   ROOT.RooFit.Name("Bkg_plus_sig"),
                   )

    if fit_res:
        pdftmp.plotOn( frame,
                       ROOT.RooFit.VisualizeError( fit_res,1, error_band_strategy ),
                       ROOT.RooFit.Normalization( 1,ROOT.RooAbsReal.RelativeExpected),
                       ROOT.RooFit.Name("FitError_AfterFit"),
                       ROOT.RooFit.FillColor(ROOT.kOrange),
                       ROOT.RooFit.LineWidth(2),
                       ROOT.RooFit.LineColor(ROOT.kBlue)
                       )

    # --> bkg only PDF error
    poi.setVal(0.)
    hist_bkg = pdftmp.createHistogram("hbkg_"+cat.GetName(),obs,ROOT.RooFit.Extended(False))
    hlist.append(hist_bkg)
    Integral_bkg_total = pdftmp.createIntegral(ROOT.RooArgSet(obs))
    Yield_bkg_total = Integral_bkg_total.getVal() * binWidth.getVal()
    pdftmp.plotOn( frame,
                   ROOT.RooFit.Normalization(Yield_bkg_total,ROOT.RooAbsReal.NumEvent),
                   ROOT.RooFit.Name("Bkg"),
                   ROOT.RooFit.LineStyle(ROOT.kDashed)
                   )
    if fit_res:
        pdftmp.plotOn( frame,
                       ROOT.RooFit.VisualizeError( fit_res,1, error_band_strategy),
                       ROOT.RooFit.Normalization( 1,ROOT.RooAbsReal.RelativeExpected),
                       ROOT.RooFit.Name("FitError_AfterFit_Mu0"),
                       ROOT.RooFit.FillColor(ROOT.kOrange),
                       ROOT.RooFit.LineWidth(2),
                       ROOT.RooFit.LineColor(ROOT.kBlue)
                       )
        if verbose:
            Yield_bkg_total_err = Integral_bkg_total.getPropagatedError( fit_res )* binWidth.getVal()
            print 'Total bkg yiedl:\t',Yield_bkg_total,' +/- ',Yield_bkg_total_err
    poi.setVal(1)
    return frame,hlist



