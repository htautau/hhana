/*                                                                                                                                                                           
Author: Romain Madar
Date:   2012-02-16
Email:  romain.madar@cern.ch

Description : This code allows the check quality of fit performed in the limit derivation.
              It works on a generic workspace produced by hist2workspace command. It performs
	      a global fit and a fit per subchannel automatically.

              Various control plots (pull distribution, correlation matrix, distribution
	      before and after fit, ...) are stored in a the rootfile FitCrossChecks.root.
*/

// C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

// Root
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TCanvas.h"
#include "TList.h"
#include "TList.h"
#include "TMath.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"

// RooFit
#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooPlot.h"
#include "RooAbsData.h"
#include "RooSimultaneous.h"
#include "RooCategory.h"
#include "RooFitResult.h"
#include "RooAbsData.h"
#include "Roo1DTable.h"
#include "RooConstVar.h"
#include <RooMsgService.h>

// RooStat
#include "RooStats/ModelConfig.h"
#include "RooStats/ProfileInspector.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodInterval.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/SamplingDistPlot.h"
#include "RooStats/ToyMCSampler.h"

using namespace std;
using namespace RooFit;
using namespace RooStats;


namespace LimitCrossCheck{
  

  // Global variables;
  RooWorkspace *w         ;
  ModelConfig  *mc        ;
  RooAbsData   *data      ;
  TFile        *outputfile; 
  double        LumiRelError;
  TDirectory   *MainDirSyst;
  TDirectory   *MainDirFitEachSubChannel;
  TDirectory   *MainDirFitGlobal;
  TDirectory   *MainDirModelInspector;
  map <string,double> MapNuisanceParamNom;
  TString OutputDir;

  //Global functions
  void     PlotHistosBeforeFit(double nSigmaToVary, double mu);
  void     PlotHistosAfterFitEachSubChannel(bool IsConditionnal , double mu);
  void     PlotHistosAfterFitGlobal(bool IsConditionnal , double mu);
  void     PlotsNuisanceParametersVSmu();
  double   FindMuUpperLimit();
  void     PrintModelObservables();
  void     PrintNuisanceParameters();
  void     PrintAllParametersAndValues(RooArgSet para);
  void     PrintNumberOfEvents(RooAbsPdf *pdf);
  void     PrintSubChannels();
  bool     IsSimultaneousPdfOK();
  bool     IsChannelNameOK();
  void     SetAllNuisanceParaToSigma(double Nsigma);
  void     SetAllStatErrorToSigma(double Nsigma);
  void     SetNuisanceParaToSigma(RooRealVar *var, double Nsigma);
  void     GetNominalValueNuisancePara();
  void     SetNominalValueNuisancePara();
  void     SetPOI(double mu);
  void     SetStyle();
  

  //======================================================
  // ================= Main function =====================
  //======================================================
  void PlotFitCrossChecks(const char* infile          = "WorkspaceForTest1.root",
			  const char* outputdir       = "./results/",
			  const char* workspaceName   = "combined",
			  const char* modelConfigName = "ModelConfig",
			  const char* ObsDataName     = "obsData"){
    
    // Cosmetics
    SetStyle();

    // Lumi error hard-coded;
    LumiRelError = 0.037;
    
    // Container for the plots
    OutputDir = (TString) outputdir;
    gSystem->Exec("mkdir -p " + OutputDir);
    gSystem->Exec("mkdir -p " + OutputDir + "/LatexFileNPs");
    gSystem->Exec("mkdir -p " + OutputDir + "/TextFileFitResult");
    outputfile = new TFile(OutputDir+"/FitCrossChecks.root","RECREATE");
    
    // Load workspace, model and data
    TFile *file = TFile::Open(infile);
    if (!file) {
      cout << "The file " << infile << " is not found/created, will stop here." << endl;
      return;
    }
    if(!(RooWorkspace*) file->Get(workspaceName)){
      cout <<"workspace not found" << endl;
      return;
    }
    
    w      = (RooWorkspace*) file->Get(workspaceName);
    mc     = (ModelConfig*) w->obj(modelConfigName);
    data   = w->data(ObsDataName);
    
    if(!data || !mc){
      w->Print();
      cout << "data or ModelConfig was not found" <<endl;
      return;
    }
      
    // Some sanity checks on the workspace
    if ( !IsSimultaneousPdfOK() ) return;
    if (   !IsChannelNameOK()   ) return;
    GetNominalValueNuisancePara();

    // Print some information
    PrintModelObservables();
    PrintNuisanceParameters();
    PrintSubChannels();
    
    // Prepare the directory structure of the outputfile
    MainDirSyst              = (TDirectory*) outputfile->mkdir("PlotsBeforeFit");
    MainDirFitEachSubChannel = (TDirectory*) outputfile->mkdir("PlotsAfterFitOnSubChannel");
    MainDirFitGlobal         = (TDirectory*) outputfile->mkdir("PlotsAfterGlobalFit");
    MainDirModelInspector    = (TDirectory*) outputfile->mkdir("PlotsNuisanceParamVSmu");
    gROOT->cd();


    // -----------------------------------------------------------------------------------
    // 1 - Plot nominal and +/- Nsigma (for each nuisance paramater) for Data, signal+bkg
    // -----------------------------------------------------------------------------------
    double nsigma = 1.0;
    double MuUpperLimit = 10; //FindMuUpperLimit();
    RooRealVar * firstPOI = dynamic_cast<RooRealVar*>(mc->GetParametersOfInterest()->first());
    double mumax = firstPOI->getMax();
    if (MuUpperLimit>mumax) MuUpperLimit=mumax;
    PlotHistosBeforeFit(nsigma,0.0);
    

    // ----------------------------------------------------------------------------------
    // 2 - Plot histograms after unconditional fit (theta and mu fitted at the same time)
    // ----------------------------------------------------------------------------------    
    bool IsConditional = false;
    //PlotHistosAfterFitEachSubChannel(IsConditional,0.0);
    //PlotHistosAfterFitGlobal(IsConditional,0.0);
        

    // --------------------------------------------------------------------------------------------
    // 3 - Plot the unconditionnal fitted nuisance paramters value (theta fitted while mu is fixed)
    // --------------------------------------------------------------------------------------------
    IsConditional = true;
    PlotHistosAfterFitEachSubChannel(IsConditional, 0.0);
    PlotHistosAfterFitGlobal(IsConditional,0.0);


    // -------------------------------------------
    // 4 - Plot the nuisance parameters versus mu
    // -------------------------------------------
    //PlotsNuisanceParametersVSmu(); // This can take time


    outputfile->Close();
    return;
  
  }
  










  // ============================================================
  // ============ Definition of all the functions ===============
  // ============================================================


  void PlotHistosBeforeFit(double nSigmaToVary, double mu){

    RooMsgService::instance().setGlobalKillBelow(ERROR);

    ostringstream MaindirName;
    MaindirName << "MuIsEqualTo_" << mu;
    TDirectory *MainDir =  (TDirectory*) MainDirSyst->mkdir(MaindirName.str().c_str());
    gROOT->cd();

    // Get the RooSimultaneous PDF
    RooSimultaneous *simPdf = (RooSimultaneous*)(mc->GetPdf());
    RooRealVar * firstPOI = dynamic_cast<RooRealVar*>(mc->GetParametersOfInterest()->first());
    firstPOI->setVal(mu);

    RooCategory* channelCat = (RooCategory*) (&simPdf->indexCat());
    TIterator* iter = channelCat->typeIterator() ;
    RooCatType* tt = NULL;    
    int  NumberOfChannelDone = 0;
    while((tt=(RooCatType*) iter->Next()) ){
      
      NumberOfChannelDone++;
      //if (NumberOfChannelDone>1) break;

      cout << endl;
      cout << endl;
      cout << " -- On category " << tt->GetName() << " " << endl;
      ostringstream SubdirName;
      SubdirName << tt->GetName();
      TDirectory *SubDirChannel = (TDirectory*) MainDir->mkdir(SubdirName.str().c_str());
      gROOT->cd();
    
      // Get pdf associated with state from simpdf
      RooAbsPdf  *pdftmp  = simPdf->getPdf(tt->GetName()) ;
      RooArgSet  *obstmp  = pdftmp->getObservables( *mc->GetObservables() ) ;
      RooAbsData *datatmp = data->reduce(Form("%s==%s::%s",channelCat->GetName(),channelCat->GetName(),tt->GetName()));
      RooRealVar *obs     = ((RooRealVar*) obstmp->first());

      // Loop over nuisance params
      TIterator* it = mc->GetNuisanceParameters()->createIterator();
      RooRealVar* var = NULL;
      bool IsAllStatDone = false; 
      while( (var = (RooRealVar*) it->Next()) ){

	string varname = (string) var->GetName();
	if ( varname.find("gamma_stat")!=string::npos ){
	  continue;
	}

	// Not consider nuisance parameter being not assocaited to systematics
	if (MapNuisanceParamNom[varname]!=0.0 &&
	    MapNuisanceParamNom[varname]!=1.0 ) continue;

	cout << endl;
	cout << "  -- On nuisance parameter : " << var->GetName() << endl; 
	
	TString cname = "can_" + (TString)tt->GetName() + "_" +  (TString)var->GetName() + "_mu";
	cname += mu;
	TCanvas* c2 = new TCanvas( cname );
        RooPlot* frame = obs->frame();
	TString FrameName = "Plot_" + (TString)tt->GetName() + "_" +  (TString)var->GetName() + "_mu";
	FrameName += mu;
	frame->SetName( FrameName );
        frame->SetYTitle(var->GetName());
	datatmp->plotOn(frame,MarkerSize(1),Name("Mydata"));
	cout << "N(observed) = " << datatmp->sumEntries() << endl;	

	// Fisrt be sure that all nuisance parameters are nominal
	SetAllStatErrorToSigma(0.0);
	SetAllNuisanceParaToSigma(0.0);
	SetPOI(mu);	
	
	// -1 sigma
	SetNuisanceParaToSigma(var,-nSigmaToVary);
	SetPOI(mu);
	pdftmp->plotOn(frame,LineWidth(2), LineColor(kGreen), LineStyle(kDashed), Name("m1sigma"),
		       Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));
	cout << "N(-sigma) = " << pdftmp->expectedEvents(*obs) << endl;

	// +1 sigma
	SetNuisanceParaToSigma(var,+nSigmaToVary);
	SetPOI(mu);	
	pdftmp->plotOn(frame,Name("p1sigma"),LineWidth(2),LineColor(kRed),LineStyle(kDashed),
		       Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));
		       cout << "N(+sigma) = " << pdftmp->expectedEvents(*obs) << endl;

	// Nominal
	SetNuisanceParaToSigma(var,0.0);
	SetPOI(mu);
	pdftmp->plotOn(frame,LineWidth(2), Name("Mynominal"),
		       Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));
	cout << "N(nominal) = " << pdftmp->expectedEvents(*obs) << endl;
	
	TLegend *leg = new TLegend(0.43,0.68,0.70,0.86);
	leg->SetBorderSize(0);
	leg->SetFillColor(0);
	leg->SetTextFont(62);
	leg->SetTextSize(0.050);
	for (int i=0; i<frame->numItems(); i++) {
	  TString obj_name=frame->nameOf(i); 
	  if (obj_name=="") continue;
	  TObject *obj = frame->findObject(obj_name.Data());
	  if (((string)obj_name).find("data")   !=string::npos) leg->AddEntry( obj , "Data" , "p");
	  if (((string)obj_name).find("m1sigma")!=string::npos) leg->AddEntry( obj , (TString)var->GetName() + " down" , "l");
	  if (((string)obj_name).find("nominal")!=string::npos) leg->AddEntry( obj , "Nominal" , "l");
	  if (((string)obj_name).find("p1sigma")!=string::npos) leg->AddEntry( obj , (TString)var->GetName() + " up" , "l");
	}	  
	leg->Draw();
	
	SubDirChannel->cd();
	c2->cd();
	frame->Draw();
	leg->Draw();
	c2->Write();
	c2->Close();
	gROOT->cd();

	// Put everything back to the nominal
	SetAllNuisanceParaToSigma(0.0);
	SetPOI(mu);

	// Stat uncertainty
	if (!IsAllStatDone){
	  
	  cname = "can_" + (TString)tt->GetName() + "_Stat_mu";
	  cname += mu;
	  TCanvas* c4 = new TCanvas( cname );
	  RooPlot* frame3 = obs->frame();
	  FrameName = "Plot_" + (TString)tt->GetName() + "_Stat_mu";
	  FrameName += mu;
	  frame3->SetName( FrameName );
	  frame3->SetYTitle(var->GetName());
	  datatmp->plotOn(frame3,MarkerSize(1),Name("Mydata"));

	  SetAllStatErrorToSigma(-nSigmaToVary);
	  SetAllNuisanceParaToSigma(0.0);
	  SetPOI(mu);
	  pdftmp->plotOn(frame3,Name("m1sigma"),LineWidth(2), LineColor(kGreen), LineStyle(kDashed),
			 Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));
	  
	  SetAllStatErrorToSigma(+nSigmaToVary);
	  SetAllNuisanceParaToSigma(0.0);
	  SetPOI(mu);
	  pdftmp->plotOn(frame3, Name("p1sigma") ,LineWidth(2), LineColor(kRed), LineStyle(kDashed), 
			 Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));
	  
	  SetAllStatErrorToSigma(0.0);
	  SetAllNuisanceParaToSigma(0.0);
	  SetPOI(mu);
	  pdftmp->plotOn(frame3,Name("nominal"),LineWidth(2), 
			 Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));

	  leg = new TLegend(0.43,0.68,0.70,0.86);
	  leg->SetBorderSize(0);
	  leg->SetFillColor(0);
	  leg->SetTextFont(62);
	  leg->SetTextSize(0.050);
	  for (int i=0; i<frame3->numItems(); i++) {
	    TString obj_name=frame3->nameOf(i); 
	    if (obj_name=="") continue;
	    TObject *obj = frame3->findObject(obj_name.Data());
	    if (((string)obj_name).find("data")   !=string::npos) leg->AddEntry( obj , "Data" , "p");
	    if (((string)obj_name).find("m1sigma")!=string::npos) leg->AddEntry( obj , "Stat down" , "l");
	    if (((string)obj_name).find("nominal")!=string::npos) leg->AddEntry( obj , "Nominal" , "l");
	    if (((string)obj_name).find("p1sigma")!=string::npos) leg->AddEntry( obj , "Stat up" , "l");
	  }	  
	  leg->Draw();

	  SubDirChannel->cd();
	  c4->cd();
	  frame3->Draw();
	  leg->Draw();
	  c4->Write();
	  c4->Close();
	  gROOT->cd();
	  
	  IsAllStatDone=true;
	}

      }
      
    } 
    
    return;
  }
  


  void PlotHistosAfterFitEachSubChannel(bool IsConditionnal, double mu){
    
    // Conditionnal or unconditional fit
    TString TS_IsConditionnal;
    if (IsConditionnal) TS_IsConditionnal="conditionnal";
    else                TS_IsConditionnal="unconditionnal";
    RooRealVar * firstPOI = dynamic_cast<RooRealVar*>(mc->GetParametersOfInterest()->first());
    firstPOI->setVal(mu);

    ostringstream MaindirName;
    if (IsConditionnal) MaindirName << TS_IsConditionnal << "_MuIsEqualTo_" << mu;
    else                MaindirName << TS_IsConditionnal;
    TDirectory *MainDir =  (TDirectory*) MainDirFitEachSubChannel->mkdir(MaindirName.str().c_str());
    gROOT->cd();
    
    // Get the RooSimultaneous PDF
    RooSimultaneous *simPdf = (RooSimultaneous*)(mc->GetPdf());
    
    RooCategory* channelCat = (RooCategory*) (&simPdf->indexCat());
    TIterator *iter = channelCat->typeIterator() ;
    RooCatType *tt  = NULL;
    while((tt=(RooCatType*) iter->Next()) ){
     
      cout << endl;
      cout << endl;
      cout << " -- On category " << tt->GetName() << " " << endl;
      ostringstream SubdirName;
      SubdirName << tt->GetName();
      TDirectory *SubDirChannel = (TDirectory*) MainDir->mkdir(SubdirName.str().c_str());
      gROOT->cd();
         
      // Get pdf and datset associated to the studied channel
      RooAbsPdf  *pdftmp  = simPdf->getPdf( tt->GetName() );
      RooAbsData *datatmp = data->reduce(Form("%s==%s::%s",channelCat->GetName(),channelCat->GetName(),tt->GetName()));
      RooArgSet  *obstmp  = pdftmp->getObservables( *mc->GetObservables() ) ;
      RooRealVar *obs = ((RooRealVar*) obstmp->first());
      
      // Fit 
      if (IsConditionnal) firstPOI->setConstant();
      ROOT::Math::MinimizerOptions::SetDefaultStrategy(2);
      RooFitResult *fitres = pdftmp->fitTo( *datatmp , Save() );
      cout << endl;
      cout << endl;
      if (IsConditionnal) cout << "Conditionnal fit : mu is fixed at " << mu << endl;
      else                cout << "Unconditionnal fit : mu is fitted" << endl;
      firstPOI->setConstant(kFALSE);
  
      // Plotting the nuisance paramaters correlations during the fit
      TString cname = "can_NuisPara_" + (TString)tt->GetName() + "_" + TS_IsConditionnal + "_mu";
      cname += mu;
      TCanvas* c1 = new TCanvas( cname, cname, 1260, 500);
      c1->Divide(2,1);
      TH2D *h2Dcorrelation = (TH2D*) fitres->correlationHist(cname);
      TString hname = "Corr_NuisPara_"+ (TString)tt->GetName() + "_"  + TS_IsConditionnal + "_mu";
      h2Dcorrelation->SetName(hname);
      c1->cd(1); h2Dcorrelation->Draw("colz");

      // Plotting the nuisance paramaters after fit
      TString h1name = "h_NuisParaPull_" + (TString)tt->GetName() + "_" + TS_IsConditionnal + "_mu";
      h1name += mu;
      TIterator* it1 = mc->GetNuisanceParameters()->createIterator();
      RooRealVar* var = NULL;
      int Npar=0;
      int NparNotStat=0;
      while( (var = (RooRealVar*) it1->Next()) ) {
	Npar++;
	string varname = (string) var->GetName();      
	if (varname.find("gamma_stat")==string::npos) NparNotStat++;
      }
      NparNotStat=NparNotStat;
      TH1F * h1Dpull = new TH1F(h1name,h1name,NparNotStat,0,NparNotStat);
      h1Dpull->GetYaxis()->SetRangeUser(-5.5,5.5);
      h1Dpull->GetXaxis()->SetTitle("#theta");

      // Create a latex table of NPs after fit
      TString fname = OutputDir + "/LatexFileNPs/Fit"+(TString)tt->GetName()+"_nuisPar_"+TS_IsConditionnal+"_mu";
      fname += mu;
      fname += ".tex";
      ofstream fnuisPar(fname.Data());
      TString fnuiscorr = OutputDir + "/TextFileFitResult/Fit"+(TString)tt->GetName()+"_fitres_"+TS_IsConditionnal+"_mu";
      fnuiscorr += mu;
      fnuiscorr += ".txt";
      ofstream fnuisParAndCorr(fnuiscorr.Data());
      fnuisParAndCorr << "NUISANCE_PARAMETERS" << endl;
      
      fnuisPar << endl;
      fnuisPar << "\\begin{tabular}{|l|c|}" << endl;
      fnuisPar << "\\hline" << endl;
      fnuisPar << "Nuisance parameter & postfit value (in $\\sigma$ unit) \\\\\\hline" << endl;

      int ib=0;
      TIterator* it2 = mc->GetNuisanceParameters()->createIterator();
      while( (var = (RooRealVar*) it2->Next()) ){

	// Not consider nuisance parameter being not associated to syst
	string varname = (string) var->GetName();
	if ( (varname.find("gamma_stat")!=string::npos) ) continue;
	
	double pull  = var->getVal() / 1.0 ; // GetValue() return value in unit of sigma
	double error = var->getError() / 1.0; 
	
	if(strcmp(var->GetName(),"Lumi")==0){
	  pull  = (var->getVal() - w->var("nominalLumi")->getVal() ) / (w->var("nominalLumi")->getVal() * LumiRelError );
	  error = var->getError() / (w->var("nominalLumi")->getVal() * LumiRelError); 
	}
	
	TString vname=var->GetName();
	vname.ReplaceAll("alpha_","");
	vname.ReplaceAll("gamma_","");
	vname.ReplaceAll("Lumi","Luminosity");
	vname.ReplaceAll("_","\\_");
	fnuisPar.precision(3);
	fnuisPar << vname << " & " << pull << " $\\pm$ " << error << "\\\\" << endl;
	fnuisParAndCorr << vname << "  " << pull << "  " << error << "   -" << error << endl;
	
	ib++;
	h1Dpull->SetBinContent(ib,pull);
	h1Dpull->SetBinError(ib,error);
	h1Dpull->GetXaxis()->SetBinLabel(ib,var->GetName());
      }

      fnuisPar << "\\hline" << endl;
      fnuisPar << "\\end{tabular}" << endl;
      fnuisPar.close();
      
      fnuisParAndCorr << endl << endl << "CORRELATION_MATRIX" << endl;
      fnuisParAndCorr << h2Dcorrelation->GetNbinsX() << "   " << h2Dcorrelation->GetNbinsY() << endl;
      for(int kk=1; kk < h2Dcorrelation->GetNbinsX()+1; kk++) {
	for(int ll=1; ll < h2Dcorrelation->GetNbinsY()+1; ll++) {
	  fnuisParAndCorr << h2Dcorrelation->GetBinContent(kk,ll) << "   ";
	}
	fnuisParAndCorr << endl;
      }
      fnuisParAndCorr << endl;
      fnuisParAndCorr.close();
      
      
      double _1SigmaValue[1000];
      double _2SigmaValue[1000];
      double NuisParamValue[1000];
      for (int i=0 ; i<NparNotStat+1 ; i++){
	_1SigmaValue[i] = 1.0;
	_1SigmaValue[2*NparNotStat-i] = -1;
	_2SigmaValue[i] = 2;
	_2SigmaValue[2*NparNotStat-i] = -2;
	NuisParamValue[i] = i;
	NuisParamValue[2*NparNotStat-1-i] = i;
      }

      TGraph *_1sigma = new TGraph(2*NparNotStat,NuisParamValue,_1SigmaValue);
      TGraph *_2sigma = new TGraph(2*NparNotStat,NuisParamValue,_2SigmaValue);
      c1->cd(2); 
      h1Dpull->SetLineWidth(2);
      h1Dpull->SetLineColor(1);
      h1Dpull->SetMarkerColor(1);
      h1Dpull->SetMarkerStyle(21);
      h1Dpull->SetMarkerSize(0.11);      
      _2sigma->SetFillColor(5);
      _2sigma->SetLineColor(5);
      _2sigma->SetMarkerColor(5);
      _1sigma->SetFillColor(3);
      _1sigma->SetLineColor(3);
      _1sigma->SetMarkerColor(3);
      h1Dpull->Draw("histE");
      _2sigma->Draw("F"); 
      _1sigma->Draw("F"); 
      h1Dpull->Draw("histEsame");
      
      TLatex text;
      text.SetNDC();
      text.SetTextSize( 0.054);
      text.SetTextAlign(31);
      TString WritDownMuValue;
      if(!IsConditionnal) WritDownMuValue = "#mu_{best} = ";
      else                WritDownMuValue = "#mu_{fixed} = ";
      WritDownMuValue += Form("%2.2f",firstPOI->getVal());
      c1->cd(2); 
      text.DrawLatex( 0.87,0.81, WritDownMuValue );
     

      // Plotting the distributions
      cname = "can_DistriAfterFit_" + (TString)tt->GetName() + "_" + TS_IsConditionnal + "_mu";
      cname += mu;
      TCanvas* c2 = new TCanvas( cname );
      RooPlot* frame = obs->frame();
      TString FrameName = "Plot_" + (TString)tt->GetName() + "_FitIsconditional" + (TString) IsConditionnal;
      frame->SetName( FrameName );
      frame->SetYTitle("EVENTS");
      pdftmp->plotOn(frame,FillColor(kOrange),LineWidth(2),LineColor(kBlue),VisualizeError(*fitres,1),
		     Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent),Name("AfterFit_error"),Name("AfterFit"));
      pdftmp->plotOn(frame,LineWidth(2),Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));
      datatmp->plotOn(frame,MarkerSize(1),Name("Data"));
      c2->cd();
      frame->Draw();
      double chi2 = frame->chiSquare();

      // Putting nuisance parameter at the central value and draw the nominal distri
      SetAllStatErrorToSigma(0.0);
      SetAllNuisanceParaToSigma(0.0);
      if (!IsConditionnal) SetPOI(0.0);
      pdftmp->plotOn(frame,LineWidth(2),Name("BeforeFit"),LineStyle(kDashed),Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));      
      c2->cd();
      frame->Draw();
      text.DrawLatex( 0.84,0.81, WritDownMuValue );
      TString ts_chi2 = Form("#chi^{2}=%1.1f",chi2);
      text.DrawLatex( 0.22, 0.83, ts_chi2 );
                  
      TLegend *leg = new TLegend(0.64,0.56,0.87,0.74);
      leg->SetBorderSize(0);
      leg->SetFillColor(0);
      leg->SetTextFont(62);
      leg->SetTextSize(0.050);
      for (int i=0; i<frame->numItems(); i++) {
	TString obj_name=frame->nameOf(i); 
	if (obj_name=="") continue;
	TObject *obj = frame->findObject(obj_name.Data());
	if (((string)obj_name).find("Data")   !=string::npos) leg->AddEntry( obj , "Data" , "p");
	if (((string)obj_name).find("AfterFit")!=string::npos) leg->AddEntry( obj , "After fit" , "lf");
	TString legname = Form("Before fit (#mu=%2.2f)",firstPOI->getVal());
	if (((string)obj_name).find("BeforeFit")!=string::npos) leg->AddEntry( obj ,legname , "l");
      }	  
      leg->Draw();
            

      // Save plots in outputfile
      SubDirChannel->cd();
      c1->Write();
      c2->Write();
      c1->Close();
      c2->Close();
      gROOT->cd();

    }

    return;
  }


  void PlotHistosAfterFitGlobal(bool IsConditionnal, double mu){
    
    // Conditionnal or unconditional fit
    TString TS_IsConditionnal;
    if (IsConditionnal) TS_IsConditionnal="conditionnal";
    else                TS_IsConditionnal="unconditionnal";

    RooRealVar * firstPOI = dynamic_cast<RooRealVar*>(mc->GetParametersOfInterest()->first());
    firstPOI->setVal(mu);
    
    ostringstream MaindirName;
    if (IsConditionnal) MaindirName << TS_IsConditionnal << "_MuIsEqualTo_" << mu;
    else                MaindirName << TS_IsConditionnal;
    TDirectory *MainDir =  (TDirectory*) MainDirFitGlobal->mkdir(MaindirName.str().c_str());
    gROOT->cd();

    // Get the RooSimultaneous PDF
    RooSimultaneous *simPdf = (RooSimultaneous*)(mc->GetPdf());

    // Fit 
    if (IsConditionnal) firstPOI->setConstant();
    ROOT::Math::MinimizerOptions::SetDefaultStrategy(2);
    RooFitResult    *fitresGlobal  = simPdf->fitTo( *data , Save() );    
    const RooArgSet *ParaGlobalFit = mc->GetNuisanceParameters();
    mc->SetSnapshot(*ParaGlobalFit);
    firstPOI->setConstant(kFALSE);

    
    if (IsConditionnal) cout << "Conditionnal fit : mu is fixed at " << mu << endl;
    else                cout << "Unconditionnal fit : mu is fitted" << endl;
    fitresGlobal->Print("v");
    
    // PLotting the nuisance paramaters correlations during the fit
    TString cname = "can_NuisPara_GlobalFit_" + TS_IsConditionnal + "_mu";
    cname += mu;

    TCanvas* c1 = new TCanvas( cname, cname, 1260, 500);
    c1->Divide(2,1);
    TH2D *h2Dcorrelation = (TH2D*) fitresGlobal->correlationHist();
    TString hname = "Corr_NuisPara_GlobalFit_" + TS_IsConditionnal + "_mu";
    h2Dcorrelation->SetName(hname);
    c1->cd(1); h2Dcorrelation->Draw("colz");

    // PLotting the nuisance paramaters correlations during the fit
    TString h1name = "h_NuisParaPull_GlobalFit_" + TS_IsConditionnal + "_mu";
    h1name += mu;
    TIterator* it1 = mc->GetNuisanceParameters()->createIterator();
    RooRealVar* var = NULL;
    int Npar=0;
    int NparNotStat=0;
    while( (var = (RooRealVar*) it1->Next()) ) {
      Npar++;
      string varname = (string) var->GetName();      
      if (varname.find("gamma_stat")==string::npos) NparNotStat++;
    }
    TH1F * h1Dpull = new TH1F(h1name,h1name,NparNotStat,0,NparNotStat);
    h1Dpull->GetYaxis()->SetRangeUser(-5.5,5.5);
    h1Dpull->GetYaxis()->SetTitle("(#theta_{fit} - #theta_{0}) / #Delta#theta");
    h1Dpull->GetXaxis()->SetTitle("#theta");
    
    // Create a latex table of NPs after fit
    TString fname = OutputDir + "/LatexFileNPs/GlobalFit_nuisPar_"+TS_IsConditionnal+"_mu";
    fname += mu;
    fname += ".tex";
    ofstream fnuisPar(fname.Data());
    TString fnuiscorr = OutputDir + "/TextFileFitResult/GlobalFit_fitres_"+TS_IsConditionnal+"_mu";
    fnuiscorr += mu;
    fnuiscorr += ".txt";
    ofstream fnuisParAndCorr(fnuiscorr.Data());
    fnuisParAndCorr << "NUISANCE_PARAMETERS" << endl;

    fnuisPar << endl;
    fnuisPar << "\\begin{tabular}{|l|c|}" << endl;
    fnuisPar << "\\hline" << endl;
    fnuisPar << "Nuisance parameter & postfit value (in $\\sigma$ unit) \\\\\\hline" << endl;

    int ib=0;
    TIterator* it2 = mc->GetNuisanceParameters()->createIterator();
    while( (var = (RooRealVar*) it2->Next()) ){
      
      // Not consider nuisance parameter being not associated to syst
      string varname = (string) var->GetName();
      if ((varname.find("gamma_stat")!=string::npos)) continue;

      double pull  = var->getVal() / 1.0 ; // GetValue() return value in unit of sigma
      double error = var->getError() / 1.0; 
      
      if(strcmp(var->GetName(),"Lumi")==0){
	pull  = (var->getVal() - w->var("nominalLumi")->getVal() ) / (w->var("nominalLumi")->getVal() * LumiRelError );
	error = var->getError() / (w->var("nominalLumi")->getVal() * LumiRelError); 
      }
      
      TString vname=var->GetName();
      vname.ReplaceAll("alpha_","");
      vname.ReplaceAll("Lumi","Luminosity");
      vname.ReplaceAll("_","\\_");
      fnuisPar.precision(3);
      fnuisPar << vname << " & " << pull << " $\\pm$ " << error << "\\\\" << endl;
      fnuisParAndCorr << vname << "  " << pull << "  " << error << "   -" << error << endl;

      ib++;
      h1Dpull->SetBinContent(ib,pull);
      h1Dpull->SetBinError(ib,error);
      h1Dpull->GetXaxis()->SetBinLabel(ib,var->GetName());
    }

    fnuisPar << "\\hline" << endl;
    fnuisPar << "\\end{tabular}" << endl;
    fnuisPar.close();

    fnuisParAndCorr << endl << endl << "CORRELATION_MATRIX" << endl;
    fnuisParAndCorr << h2Dcorrelation->GetNbinsX() << "   " << h2Dcorrelation->GetNbinsY() << endl;
    for(int kk=1; kk < h2Dcorrelation->GetNbinsX()+1; kk++) {
      for(int ll=1; ll < h2Dcorrelation->GetNbinsY()+1; ll++) {
	fnuisParAndCorr << h2Dcorrelation->GetBinContent(kk,ll) << "   ";
      }
      fnuisParAndCorr << endl;
    }
    fnuisParAndCorr << endl;
    fnuisParAndCorr.close();
    
    double _1SigmaValue[1000];
    double _2SigmaValue[1000];
    double NuisParamValue[1000];
    for (int i=0 ; i<NparNotStat+1 ; i++){
      _1SigmaValue[i] = 1.0;
      _1SigmaValue[2*NparNotStat-i] = -1;
      _2SigmaValue[i] = 2;
      _2SigmaValue[2*NparNotStat-i] = -2;
      NuisParamValue[i] = i;
      NuisParamValue[2*NparNotStat-1-i] = i;
    }
    TGraph *_1sigma = new TGraph(2*NparNotStat,NuisParamValue,_1SigmaValue);
    TGraph *_2sigma = new TGraph(2*NparNotStat,NuisParamValue,_2SigmaValue);
    c1->cd(2); 
    h1Dpull->SetLineWidth(2);
    h1Dpull->SetLineColor(1);
    h1Dpull->SetMarkerColor(1);
    h1Dpull->SetMarkerStyle(21);
    h1Dpull->SetMarkerSize(0.11);      
    _2sigma->SetFillColor(5);
    _2sigma->SetLineColor(5);
    _2sigma->SetMarkerColor(5);
    _1sigma->SetFillColor(3);
    _1sigma->SetLineColor(3);
    _1sigma->SetMarkerColor(3);
    h1Dpull->Draw("histE");
    _2sigma->Draw("F"); 
    _1sigma->Draw("F"); 
    h1Dpull->Draw("histEsame");
    h1Dpull->GetYaxis()->DrawClone();
    
    TLatex text;
    text.SetNDC();
    text.SetTextSize( 0.054);
    text.SetTextAlign(31);
    TString WritDownMuValue;
    if(!IsConditionnal) WritDownMuValue = "#mu_{best} = ";
    else                WritDownMuValue = "#mu_{fixed} = ";
    WritDownMuValue += Form("%2.2f",firstPOI->getVal());
    c1->cd(2); 
    text.DrawLatex( 0.87,0.81, WritDownMuValue );

    MainDir->cd();
    c1->Write();
    gROOT->cd();

    // PLotting the distributions for each subchannel
    RooCategory* channelCat = (RooCategory*) (&simPdf->indexCat());
    TIterator *iter = channelCat->typeIterator() ;
    RooCatType *tt  = NULL;
    vector<double> Chi2Channel;Chi2Channel.clear();
    vector<TString> NameChannel;NameChannel.clear();
    while((tt=(RooCatType*) iter->Next()) ){
                
      RooAbsPdf  *pdftmp  = simPdf->getPdf( tt->GetName() );
      RooAbsData *datatmp = data->reduce(Form("%s==%s::%s",channelCat->GetName(),channelCat->GetName(),tt->GetName()));
      RooArgSet  *obstmp  = pdftmp->getObservables( *mc->GetObservables() ) ;
      RooRealVar *obs     = ((RooRealVar*) obstmp->first());
      
      // Load the value from the global fit
      mc->GetSnapshot();

      cname = "can_DistriAfterFit_"+ (TString) tt->GetName() +"_GlobalFit_" + TS_IsConditionnal + "_mu";
      cname += mu;
      TCanvas* c2 = new TCanvas( cname );
      RooPlot* frame = obs->frame();
      TString FrameName = "Plot_DistriGlobal_" + (TString) IsConditionnal;
      frame->SetName( FrameName );
      frame->SetYTitle("EVENTS");
      pdftmp->plotOn(frame,FillColor(kOrange),LineWidth(2),LineColor(kBlue),VisualizeError(*fitresGlobal,1),
		     Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent),Name("AfterFit"));
      pdftmp->plotOn(frame,LineWidth(2),Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));
      datatmp->plotOn(frame,MarkerSize(1),Name("Data"));
      c2->cd();
      frame->Draw();
      double chi2 = frame->chiSquare();
      Chi2Channel.push_back( chi2 );
      NameChannel.push_back( (TString)tt->GetName() );
	
      // Putting nuisance parameter at the central value and draw the nominal distri
      SetAllStatErrorToSigma(0.0);
      SetAllNuisanceParaToSigma(0.0);
      if (!IsConditionnal) SetPOI(0.0);
      pdftmp->plotOn(frame,LineWidth(2),Name("BeforeFit"),LineStyle(kDashed),Normalization(pdftmp->expectedEvents(*obs),RooAbsReal::NumEvent));      
      c2->cd();
      frame->Draw();
      c2->cd(); 
      text.DrawLatex( 0.84,0.81, WritDownMuValue );
      TString ts_chi2 = Form("#chi^{2}=%1.1f", chi2 );
      text.DrawLatex( 0.22, 0.83, ts_chi2 );

      TLegend *leg = new TLegend(0.64,0.56,0.87,0.74);
      leg->SetBorderSize(0);
      leg->SetFillColor(0);
      leg->SetTextFont(62);
      leg->SetTextSize(0.050);
      for (int i=0; i<frame->numItems(); i++) {
	TString obj_name=frame->nameOf(i); 
	if (obj_name=="") continue;
	TObject *obj = frame->findObject(obj_name.Data());
	if (((string)obj_name).find("Data")   !=string::npos) leg->AddEntry( obj , "Data" , "p");
	if (((string)obj_name).find("AfterFit")!=string::npos) leg->AddEntry( obj , "After fit" , "lf");
	TString legname = Form("Before fit (#mu=%2.2f)",firstPOI->getVal());
	if (((string)obj_name).find("BeforeFit")!=string::npos) leg->AddEntry( obj ,legname , "l");
      }	  
      leg->Draw();


      // Save the plots
      MainDir->cd();
      c2->Write();
      c2->Close();
      gROOT->cd();
    }

    int Nchannel = NameChannel.size();
    TH1F *hChi2 = new TH1F("Chi2PerChannel","Chi2PerChannel",Nchannel,0,Nchannel);
    for (int jb=0 ; jb<Nchannel ; jb++){
      hChi2->SetBinContent(jb+1,Chi2Channel[jb]);
      hChi2->GetXaxis()->SetBinLabel(jb+1,NameChannel[jb]);
    }
    sort( Chi2Channel.begin(), Chi2Channel.end());
    hChi2->GetYaxis()->SetRangeUser(0.0,Chi2Channel[Chi2Channel.size()-1]*1.50);
    hChi2->SetTitle("#chi^{2} overview among channels");
    hChi2->SetLineColor(1);
    hChi2->SetMarkerColor(1);
    hChi2->SetLineWidth(2);


    MainDir->cd();
    hChi2->Write();
    gROOT->cd();

    return;
  }



  void PlotsNuisanceParametersVSmu(){
    cout << endl;
    cout << endl;
    cout << "Performing a global fit for mu : can take time ..." << endl;
    cout << endl;

    ProfileInspector p;
    TList* list = p.GetListOfProfilePlots(*data,mc);

    for(int i=0; i<list->GetSize(); ++i){

      TString cname = "ProfileInspector_" + (TString) list->At(i)->GetName();
      TCanvas* c1 = new TCanvas(cname);
      c1->cd();
      list->At(i)->Draw("al");

      MainDirModelInspector->cd();
      c1->Write();
      c1->Close();
      gROOT->cd();
    }
    return;
  }


  double FindMuUpperLimit(){

    //RooMsgService::instance().setGlobalKillBelow(ERROR);

    RooRealVar* firstPOI = (RooRealVar*) mc->GetParametersOfInterest()->first();
    ProfileLikelihoodCalculator plc(*data,*mc);
    LikelihoodInterval* interval = plc.GetInterval();    
    double UpperLimit = interval->UpperLimit(*firstPOI);

    TCanvas* c2 = new TCanvas( "Likelihood_vs_mu" );
    LikelihoodIntervalPlot plot(interval);
    plot.SetNPoints(50);
    c2->cd();
    plot.Draw("");
    delete interval;
    
    SetAllStatErrorToSigma(0.0);
    SetAllNuisanceParaToSigma(0.0);

    outputfile->cd();
    c2->Write();
    gROOT->cd();

    return UpperLimit;

  }


    void GetNominalValueNuisancePara(){
    TIterator *it = mc->GetNuisanceParameters()->createIterator();
    RooRealVar *var = NULL;
    if (MapNuisanceParamNom.size() > 0) MapNuisanceParamNom.clear();
    std::cout << "Nuisance parameter names and values" << std::endl;
    while ((var = (RooRealVar*)it->Next()) != NULL){
      const double val = var->getVal();
      MapNuisanceParamNom[(string)var->GetName()] = val;
    }
    return;
  }
  

  void SetNominalValueNuisancePara(){
    TIterator *it = mc->GetNuisanceParameters()->createIterator();
    RooRealVar *var = NULL;
    while ((var = (RooRealVar*)it->Next()) != NULL){
      const double val =  MapNuisanceParamNom[(string)var->GetName()];
      var->setVal(val);
    }
    return;
  }
  
  
  void SetAllStatErrorToSigma(double Nsigma){
    
    TIterator* it = mc->GetNuisanceParameters()->createIterator();
    RooRealVar* var = NULL;
    while( (var = (RooRealVar*) it->Next()) ){
      string varname = (string) var->GetName();
      if ( varname.find("gamma_stat")!=string::npos ){
	RooAbsReal* nom_gamma = (RooConstVar*) w->obj( ("nom_" + varname).c_str() );
	double nom_gamma_val = nom_gamma->getVal();
	double sigma = 1/TMath::Sqrt( nom_gamma_val );
	var->setVal(1 + Nsigma*sigma);
      }
    }

    return;
  }



  void SetAllNuisanceParaToSigma(double Nsigma){
    
    if (fabs(Nsigma)<0.1){
      SetNominalValueNuisancePara();
      return;
    }

    TIterator* it = mc->GetNuisanceParameters()->createIterator();
    RooRealVar* var = NULL;
    while( (var = (RooRealVar*) it->Next()) ){
      string varname = (string) var->GetName();
      if ( varname.find("gamma_stat")!=string::npos ) continue;
      if(strcmp(var->GetName(),"Lumi")==0){
	var->setVal(w->var("nominalLumi")->getVal()*(1+Nsigma*LumiRelError));
      } else{
	var->setVal(Nsigma);
      }	
    }
    
    return;
  }
  
  
  void SetNuisanceParaToSigma(RooRealVar *var, double Nsigma){
    
    string varname = (string) var->GetName();
    if ( varname.find("gamma_stat")!=string::npos ) return;

    if(strcmp(var->GetName(),"Lumi")==0){
      var->setVal(w->var("nominalLumi")->getVal()*(1+Nsigma*LumiRelError));
    } else{
      var->setVal(Nsigma);
    }	
    
    return;
  }
  
  
  void SetPOI(double mu){
    RooRealVar * firstPOI = dynamic_cast<RooRealVar*>(mc->GetParametersOfInterest()->first());
    firstPOI->setVal(mu);
    return  ;
  }
  
  
  bool IsSimultaneousPdfOK(){
    
    bool IsOK=true;
    
    bool IsSimultaneousPDF = strcmp(mc->GetPdf()->ClassName(),"RooSimultaneous")==0;
    if (!IsSimultaneousPDF){
      cout << " ERROR : no Simultaneous PDF was found, will stop here." << endl;
      cout << " You need to investigate your input histogramms." << endl;
      IsOK = false;
    }
    
    return IsOK;

  }


  bool IsChannelNameOK(){

    bool IsOK=true; 
    if( !IsSimultaneousPdfOK() ) return false;

    RooSimultaneous* simPdf = (RooSimultaneous*)(mc->GetPdf());
    RooCategory* channelCat = (RooCategory*) (&simPdf->indexCat());
    TIterator* iter = channelCat->typeIterator() ;
    RooCatType* tt = NULL;    
    while((tt=(RooCatType*) iter->Next()) ){
      string channelName =  tt->GetName();
      if (channelName.find("/")!=string::npos){
	cout << endl;
	cout << "One of the channel name contain a caracter \"/\" : " << endl;
	cout << "  - "  << channelName << endl;
	cout << "This is mis-intrepreted by roofit in the reading of the workspace. " << endl;
	cout << "Please change the channel name in the xml file to run this code." << endl;
	cout << endl;
	IsOK = false;
      }
    }


    return IsOK;
  }

  
  void PrintModelObservables(){
    
    RooArgSet* AllObservables = (RooArgSet*) mc->GetObservables();
    TIterator* iter = AllObservables->createIterator() ;
    RooAbsArg* MyObs = NULL;    
    cout << endl;
    cout << "List of model Observables : "  << endl;
    cout << "----------------------------"  << endl;
    while( (MyObs = (RooAbsArg*) iter->Next()) )
      MyObs->Print();
    
    return;
  }

  
  void PrintNuisanceParameters(){
    
    RooArgSet nuis = *mc->GetNuisanceParameters();
    TIterator* itr = nuis.createIterator();
    RooRealVar* arg;
    cout << endl;
    cout << "List of nuisance parameters : "  << endl;
    cout << "----------------------------"  << endl;    
    while ((arg=(RooRealVar*)itr->Next())) {
      if (!arg) continue;
      cout << arg->GetName()  << " : " << arg->getVal() << "+/-" << arg->getError() << endl;
    }
    return;
  }


  void PrintAllParametersAndValues(RooArgSet para){
    TIterator* itr = para.createIterator();
    RooRealVar* arg;
    cout << endl;
    cout << "List of parameters : "  << endl;
    cout << "----------------------------"  << endl;    
    while ((arg=(RooRealVar*)itr->Next())) {
      if (!arg) continue;
      cout << arg->GetName() << " = " << arg->getVal() << endl;
    }
    return;
  }

  void PrintSubChannels(){
    
    RooMsgService::instance().setGlobalKillBelow(ERROR);

    if( !IsSimultaneousPdfOK() ) return;
    
    RooSimultaneous* simPdf = (RooSimultaneous*)(mc->GetPdf());
    RooCategory* channelCat = (RooCategory*) (&simPdf->indexCat());
    TIterator* iter = channelCat->typeIterator() ;
    RooCatType* tt = NULL;    

    while((tt=(RooCatType*) iter->Next()) ){
      
      RooAbsPdf  *pdftmp  = simPdf->getPdf( tt->GetName() );
      RooAbsData *datatmp = data->reduce(Form("%s==%s::%s",channelCat->GetName(),channelCat->GetName(),tt->GetName()));

      cout << endl;
      cout << endl;
      cout << "Details on channel " << tt->GetName() << " : "  << endl;
      cout << "----------------------------------------------------------" << endl;      
      datatmp->Print();
      pdftmp->Print();
      PrintNumberOfEvents(pdftmp);

    }

    return;
  }


  void PrintNumberOfEvents(RooAbsPdf *pdf){
    
    RooRealVar* firstPOI = (RooRealVar*) mc->GetParametersOfInterest()->first();
    double val_sym=1;
    cout 
      << Form(" %3s |","")
      << Form(" %-32s |","Nuisance Parameter") 
      << Form(" %18s |","Signal events") 
      << Form(" %18s |","% Change (+1sig)") 
      << Form(" %18s |","% Change (-1sig)") 
      << Form(" %18s |","Background events") 
      << Form(" %18s |","% Change (+1sig)") 
      << Form(" %18s |","% Change (-1sig)") 
      << endl;
  
    int inuis=-1;
    RooArgSet  *obstmp  = pdf->getObservables( *mc->GetObservables() ) ;
    RooRealVar *myobs   = ((RooRealVar*) obstmp->first());

    RooArgSet nuis = *mc->GetNuisanceParameters();
    TIterator* itr = nuis.createIterator();
    RooRealVar* arg;
    while ((arg=(RooRealVar*)itr->Next())) {
      if (!arg) continue;
      //
      ++inuis;
      //

      double val_hi = val_sym;
      double val_lo = -val_sym;
      double val_nom = arg->getVal();
      if (string(arg->GetName()) == "Lumi"){
	val_nom = w->var("nominalLumi")->getVal();
	val_hi  = w->var("nominalLumi")->getVal() * (1+LumiRelError);
	val_lo  = w->var("nominalLumi")->getVal() * (1-LumiRelError);
      }
      //
      arg->setVal(val_hi);
      firstPOI->setVal(0);
      double b_hi = pdf->expectedEvents(*myobs);
      firstPOI->setVal(1);
      double s_hi = pdf->expectedEvents(*myobs)-b_hi;
      //
      arg->setVal(val_lo);
      firstPOI->setVal(0);
      double b_lo = pdf->expectedEvents(*myobs);
      firstPOI->setVal(1);
      double s_lo = pdf->expectedEvents(*myobs)-b_lo;
      //
      arg->setVal(val_nom);
      firstPOI->setVal(0);
      double b_nom = pdf->expectedEvents(*myobs);
      firstPOI->setVal(1);
      double s_nom = pdf->expectedEvents(*myobs)-b_nom;
      //
      double x_nom = s_nom ;
      double x_hi  = 0; if (s_nom) x_hi = (s_hi-s_nom)/s_nom; 
      double x_lo  = 0; if (s_nom) x_lo = (s_lo-s_nom)/s_nom; 
      double y_nom = b_nom ;
      double y_hi  = 0; if (b_nom) y_hi = (b_hi-b_nom)/b_nom; 
      double y_lo  = 0; if (b_nom) y_lo = (b_lo-b_nom)/b_nom; 

      cout 
	<< Form(" %3d |",inuis)
	<< Form(" %-32s |",arg->GetName()) 
	<< Form(" %18.2f |",x_nom) 
	<< Form(" %18.2f |",100*x_hi) 
	<< Form(" %18.2f |",100*x_lo) 
	<< Form(" %18.2f |",y_nom) 
	<< Form(" %18.2f |",100*y_hi) 
	<< Form(" %18.2f |",100*y_lo) 
	<< endl;
    } 

    return;
  }
  
  void SetStyle(){
    gStyle->SetOptStat(0);
    
    return;
  }
  
 
}

