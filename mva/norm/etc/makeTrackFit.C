//==================================================================
//  From Soshi Nov. 23, 2012
//
// Modified: Zinonas, 8 Dec 2012
//==================================================================

#include "TMinuit.h"
//#include "utils.h"

int NDATA;
double DATA[10000];

double sumTAU;  double dTAU[400];
double sumQCD;  double dQCD[400];
double sumQCD2; double dQCD2[400]; double dQCD_shape[400];
double sumHIG;  double dHIG[400];
double sumELW;  double dELW[400];
double sumDATA; double trackDATA[400];
double statQCD[400];
double sumTAU_pileuplow;  double dTAU_pileuplow[400];
double sumTAU_pileuphigh; double dTAU_pileuphigh[400];

double shapeUncertainty = 0.05;
double signalAcceptanceUncertainty = 0.10;

const int iRandomSeed = 65539;   // Seed for flat random number.


void Fraction1P3P(const string&s, TH2F *h, float fitQCD);

//=========================
// Log Likelihood Function
//=========================
void trackFit(Int_t &npar, Double_t *gin, Double_t &f,Double_t *par, Int_t iflag) {

  double xqcd = par[0];
  double xtau = 1.0-xqcd;
  double xelw = par[11]*sumELW/sumDATA;

  f = 1.0e30;
  if (xtau<0.0) { f = 1.0e30; return; }
  if (xqcd<0.0) { f = 1.0e30; return; }
  if (xtau>1.0) { f = 1.0e30; return; }
  if (xqcd>1.0) { f = 1.0e30; return; }

  double xshape = 0.0;
  for (int i = 0; i < 400; i++) {
    if (trackDATA[i]<1) { continue; }
    if (dTAU[i]+dQCD[i]+dELW[i]==0.0) { continue; }
    double wtau = sumDATA*xtau*dTAU[i];
    double wqcd = sumDATA*xqcd*dQCD[i];
    double welw = sumDATA*xelw*dELW[i];

    wqcd += sumDATA*xqcd*par[10]*(dQCD2[i]-dQCD[i]);
    wtau += sumDATA*xtau*par[12]*(dTAU_pileuplow[i]-dTAU[i]);
    wtau += sumDATA*xtau*par[13]*(dTAU_pileuphigh[i]-dTAU[i]);

    // treatment of low statistics events
    if (i==21) {
      if (statQCD[i]>0) {
        double xqcdexp = par[1];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }
    if (i==22) {
      if (statQCD[i]>0) {
        double xqcdexp = par[2];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }
    if (i==23) {
      if (statQCD[i]>0) {
        double xqcdexp = par[3];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }
    if (i==41) {
      if (statQCD[i]>0) {
        double xqcdexp = par[4];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }
    if (i==42) {
      if (statQCD[i]>0) {
        double xqcdexp = par[5];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }
    if (i==43) {
      if (statQCD[i]>0) {
        double xqcdexp = par[6];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }
    if (i==61) {
      if (statQCD[i]>0) {
        double xqcdexp = par[7];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }
    if (i==62) {
      if (statQCD[i]>0) {
        double xqcdexp = par[8];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }
    if (i==63) {
      if (statQCD[i]>0) {
        double xqcdexp = par[9];
        double xqcdpois = 1.0*statQCD[i]*TMath::Log(1.0*statQCD[i]/xqcdexp); 
        xshape += (xqcdpois+xqcdexp-1.0*statQCD[i]);
        wqcd   *= xqcdexp/statQCD[i];
      }
    }

    double xexpall = wtau+wqcd+welw;
    double xexpois = 1.0*trackDATA[i]*TMath::Log(1.0*trackDATA[i]/xexpall);
    xshape += (xexpois+xexpall-1.0*trackDATA[i]);
  }

  double normELWK  = TMath::Gaus(par[11],1.0,0.1);

//  f = 2.0*xshape;
//  f = 2.0*(xshape-TMath::Log(normBin12)-TMath::Log(normBin12)-TMath::Log(normBin13)
//                 -TMath::Log(normBin21)-TMath::Log(normBin22)-TMath::Log(normBin23)
//                 -TMath::Log(normBin31)-TMath::Log(normBin32)-TMath::Log(normBin33)
//		 -TMath::Log(normELWK)
//		 -TMath::Log(normSig11)-TMath::Log(normSig13)
//		 -TMath::Log(normSig31)-TMath::Log(normSig33));

  f = 2.0*(xshape-TMath::Log(normELWK));

}
///       .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.
/// 	:::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\.
/// 	      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'
void makeTrackFit(){

  int imode = 0; // 0:read data  1:make psedo-data

  int imode2 = 0; 	// 1: fit for control region   
                  	// 2: VBF-CR 
                  	// 3: VBF-SR 
                  	// 4: Boosted-CR 
                  	// 5: Boosted-SR
                  	// 6: VBF-CR w/ mass window
                  	// 7: Boosted-CR w/ mass window
				  	/// Katy - Zinonas: 0

  //char *psfilename = "track.ps";
  //char *filename   = "hist_track_alpgen.root";
  
  /// Katy - Zinonas
  char *psfilename = "track_vbf_H125_v6.eps";
  char *filename   = "TrackFitHistograms_vbf_H125_v6.root";

  TRandom *rand = new TRandom(iRandomSeed);     // Initialization

  double rho = 1.0;   // MSSM/SM
  double lumi = 1.0;//5154.43; //pb-1

  TFile *ffl = new TFile(filename,"read");

  TH2F* hist2D_kTtrack_data;
  TH2F* hist2D_kTtrack_h125;
  TH2F* hist2D_kTtrack_ztt;
  TH2F* hist2D_kTtrack_qcd;
  TH2F* hist2D_kTtrack_qcdstat;
  TH2F* hist2D_kTtrack_qcdsyst;
  TH2F* hist2D_kTtrack_elwk;
  TH2F* hist2D_kTtrack_ztt_probe;
  TH2F* hist2D_kTtrack_qcd_probe;
  TH2F* hist2D_kTtrack_ztt_pileuplow;
  TH2F* hist2D_kTtrack_ztt_pileuphigh;
  if (imode2==0) {  ///Katy - Zinonas
    hist2D_kTtrack_data    = (TH2F*) ffl->Get("hist_900001");		//hist2D_kTtrack_data OS
    hist2D_kTtrack_h125    = (TH2F*) ffl->Get("hist_800000");		//hist2D_kTtrack_h125
    hist2D_kTtrack_ztt     = (TH2F*) ffl->Get("hist_800001");		//hist2D_kTtrack_ztt
    hist2D_kTtrack_elwk    = (TH2F*) ffl->Get("hist_800005");		//hist2D_kTtrack_elwk
    hist2D_kTtrack_qcd     = (TH2F*) ffl->Get("hist_900003");		//hist2D_kTtrack_qcd
    hist2D_kTtrack_qcdstat = (TH2F*) ffl->Get("hist_900003");		//hist2D_kTtrack_qcd
    hist2D_kTtrack_qcdsyst = (TH2F*) ffl->Get("hist_900003");		//hist2D_kTtrack_qcdsyst
    hist2D_kTtrack_ztt_pileuplow  = (TH2F*) ffl->Get("hist_800001");		//hist2D_kTtrack_ztt TMP
    hist2D_kTtrack_ztt_pileuphigh = (TH2F*) ffl->Get("hist_800001");		//hist2D_kTtrack_ztt TMP
  }
  if (imode2==1) {
    hist2D_kTtrack_data    = (TH2F*) ffl->Get("hist2D_kTtrack_data");
    hist2D_kTtrack_h125    = (TH2F*) ffl->Get("hist2D_kTtrack_h125");
    hist2D_kTtrack_ztt     = (TH2F*) ffl->Get("hist2D_kTtrack_ztt");
    hist2D_kTtrack_elwk    = (TH2F*) ffl->Get("hist2D_kTtrack_elwk");
    hist2D_kTtrack_qcd     = (TH2F*) ffl->Get("hist2D_kTtrack_qcd");
    hist2D_kTtrack_qcdstat = (TH2F*) ffl->Get("hist2D_kTtrack_qcd");
    hist2D_kTtrack_qcdsyst = (TH2F*) ffl->Get("hist2D_kTtrack_qcdsyst");
    hist2D_kTtrack_ztt_pileuplow  = (TH2F*) ffl->Get("hist2D_kTtrack_ztt");
    hist2D_kTtrack_ztt_pileuphigh = (TH2F*) ffl->Get("hist2D_kTtrack_ztt");
  }
  if (imode2==2) {
    hist2D_kTtrack_data    = (TH2F*) ffl->Get("hist2D_kTtrack_data_vbfCR");
    hist2D_kTtrack_h125    = (TH2F*) ffl->Get("hist2D_kTtrack_h125_vbfCR");
    hist2D_kTtrack_ztt     = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_vbfCR");
    hist2D_kTtrack_elwk    = (TH2F*) ffl->Get("hist2D_kTtrack_elwk_vbfCR");
    hist2D_kTtrack_qcd     = (TH2F*) ffl->Get("hist2D_kTtrack_qcd_vbfCR");
    hist2D_kTtrack_qcdstat = (TH2F*) ffl->Get("hist2D_kTtrack_qcd_vbfCR");
    hist2D_kTtrack_qcdsyst = (TH2F*) ffl->Get("hist2D_kTtrack_qcdsyst_vbfCR");
    hist2D_kTtrack_ztt_pileuplow  = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_vbfCR");
    hist2D_kTtrack_ztt_pileuphigh = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_vbfCR");
  }
  if (imode2==3) {
    hist2D_kTtrack_data    = (TH2F*) ffl->Get("hist2D_kTtrack_data_vbfSR");
    hist2D_kTtrack_h125    = (TH2F*) ffl->Get("hist2D_kTtrack_h125_vbfSR");
    hist2D_kTtrack_ztt     = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_vbfSR");
    hist2D_kTtrack_elwk    = (TH2F*) ffl->Get("hist2D_kTtrack_elwk_vbfSR");
    hist2D_kTtrack_qcd     = (TH2F*) ffl->Get("hist2D_kTtrack_qcd_vbfSR");
    hist2D_kTtrack_qcdstat = (TH2F*) ffl->Get("hist2D_kTtrack_qcd_vbfSR");
    hist2D_kTtrack_qcdsyst = (TH2F*) ffl->Get("hist2D_kTtrack_qcdsyst_vbfSR");
    hist2D_kTtrack_ztt_pileuplow  = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_vbfSR");
    hist2D_kTtrack_ztt_pileuphigh = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_vbfSR");
  }
  if (imode2==4) {
    hist2D_kTtrack_data    = (TH2F*) ffl->Get("hist2D_kTtrack_data_boostCR");
    hist2D_kTtrack_h125    = (TH2F*) ffl->Get("hist2D_kTtrack_h125_boostCR");
    hist2D_kTtrack_ztt     = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_boostCR");
    hist2D_kTtrack_elwk    = (TH2F*) ffl->Get("hist2D_kTtrack_elwk_boostCR");
    hist2D_kTtrack_qcd     = (TH2F*) ffl->Get("hist2D_kTtrack_qcd_boostCR");
    hist2D_kTtrack_qcdstat = (TH2F*) ffl->Get("hist2D_kTtrack_qcd_boostCR");
    hist2D_kTtrack_qcdsyst = (TH2F*) ffl->Get("hist2D_kTtrack_qcdsyst_boostCR");
    hist2D_kTtrack_ztt_pileuplow  = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_boostCR");
    hist2D_kTtrack_ztt_pileuphigh = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_boostCR");
  }
  if (imode2==5) {
    hist2D_kTtrack_data    = (TH2F*) ffl->Get("hist2D_kTtrack_data_boostSR");
    hist2D_kTtrack_h125    = (TH2F*) ffl->Get("hist2D_kTtrack_h125_boostSR");
    hist2D_kTtrack_ztt     = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_boostSR");
    hist2D_kTtrack_elwk    = (TH2F*) ffl->Get("hist2D_kTtrack_elwk_boostSR");
    hist2D_kTtrack_qcd     = (TH2F*) ffl->Get("hist2D_kTtrack_qcd_boostSR");
    hist2D_kTtrack_qcdstat = (TH2F*) ffl->Get("hist2D_kTtrack_qcd_boostSR");
    hist2D_kTtrack_qcdsyst = (TH2F*) ffl->Get("hist2D_kTtrack_qcdsyst_boostSR");
    hist2D_kTtrack_ztt_pileuplow  = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_boostSR");
    hist2D_kTtrack_ztt_pileuphigh = (TH2F*) ffl->Get("hist2D_kTtrack_ztt_boostSR");
  }
  if (imode2==6) {
    hist2D_kTtrack_data    = (TH2F*) ffl->Get("hist2D_kTtrackZ_data_vbfCR");
    hist2D_kTtrack_h125    = (TH2F*) ffl->Get("hist2D_kTtrackZ_h125_vbfCR");
    hist2D_kTtrack_ztt     = (TH2F*) ffl->Get("hist2D_kTtrackZ_ztt_vbfCR");
    hist2D_kTtrack_elwk    = (TH2F*) ffl->Get("hist2D_kTtrackZ_elwk_vbfCR");
    hist2D_kTtrack_qcd     = (TH2F*) ffl->Get("hist2D_kTtrackZ_qcd_vbfCR");
    hist2D_kTtrack_qcdstat = (TH2F*) ffl->Get("hist2D_kTtrackZ_qcd_vbfCR");
    hist2D_kTtrack_qcdsyst = (TH2F*) ffl->Get("hist2D_kTtrackZ_qcdsyst_vbfCR");
    hist2D_kTtrack_ztt_pileuplow  = (TH2F*) ffl->Get("hist2D_kTtrackZ_ztt_vbfCR");
    hist2D_kTtrack_ztt_pileuphigh = (TH2F*) ffl->Get("hist2D_kTtrackZ_ztt_vbfCR");
  }
  if (imode2==7) {
    hist2D_kTtrack_data    = (TH2F*) ffl->Get("hist2D_kTtrackZ_data_boostCR");
    hist2D_kTtrack_h125    = (TH2F*) ffl->Get("hist2D_kTtrackZ_h125_boostCR");
    hist2D_kTtrack_ztt     = (TH2F*) ffl->Get("hist2D_kTtrackZ_ztt_boostCR");
    hist2D_kTtrack_elwk    = (TH2F*) ffl->Get("hist2D_kTtrackZ_elwk_boostCR");
    hist2D_kTtrack_qcd     = (TH2F*) ffl->Get("hist2D_kTtrackZ_qcd_boostCR");
    hist2D_kTtrack_qcdstat = (TH2F*) ffl->Get("hist2D_kTtrackZ_qcd_boostCR");
    hist2D_kTtrack_qcdsyst = (TH2F*) ffl->Get("hist2D_kTtrackZ_qcdsyst_boostCR");
    hist2D_kTtrack_ztt_pileuplow  = (TH2F*) ffl->Get("hist2D_kTtrackZ_ztt_boostCR");
    hist2D_kTtrack_ztt_pileuphigh = (TH2F*) ffl->Get("hist2D_kTtrackZ_ztt_boostCR");
  }

  //hist2D_kTtrack_ztt_probe = (TH2F*) ffl->Get("hist2D_kTtrack_ztt");
  //hist2D_kTtrack_qcd_probe = (TH2F*) ffl->Get("hist2D_kTtrack_qcd");
  ///Katy - Zinonas:
  hist2D_kTtrack_ztt_probe = (TH2F*) ffl->Get("hist_800001");
  hist2D_kTtrack_qcd_probe = (TH2F*) ffl->Get("hist_900002");	 
  
  
  TH1F* hist1D_track_TAU = new TH1F("hist1D_track_TAU","",400,0,400);
  TH1F* hist1D_track_QCD = new TH1F("hist1D_track_QCD","",400,0,400);
  TH1F* hist1D_track_ELW = new TH1F("hist1D_track_ELW","",400,0,400);

  //=====================================
  // Make PDF for track multiplicity fit
  //=====================================
  sumTAU  = 0.0;
  sumQCD  = 0.0;
  sumQCD2 = 0.0;
  sumHIG  = 0.0;
  sumELW  = 0.0;
  sumTAU_pileuplow = 0.0;
  sumTAU_pileuphigh = 0.0;
  int NbinX    = hist2D_kTtrack_ztt->GetNbinsX();
  int NbinY    = hist2D_kTtrack_ztt->GetNbinsY();
  for (int i=0; i<NbinX; i++) {
    for (int j=0; j<NbinY; j++) {
      if (hist2D_kTtrack_ztt->GetBinContent(i+1,j+1)>0)  { sumTAU += hist2D_kTtrack_ztt->GetBinContent(i+1,j+1); }
      if (hist2D_kTtrack_qcd->GetBinContent(i+1,j+1)>0)  { sumQCD += hist2D_kTtrack_qcd->GetBinContent(i+1,j+1); }
      if (hist2D_kTtrack_h125->GetBinContent(i+1,j+1)>0) { sumHIG += hist2D_kTtrack_h125->GetBinContent(i+1,j+1); }
      if (hist2D_kTtrack_elwk->GetBinContent(i+1,j+1)>0) { sumELW += hist2D_kTtrack_elwk->GetBinContent(i+1,j+1); }
      if (hist2D_kTtrack_qcdsyst->GetBinContent(i+1,j+1)>0)  { sumQCD2 += hist2D_kTtrack_qcdsyst->GetBinContent(i+1,j+1); }
      if (hist2D_kTtrack_ztt_pileuplow->GetBinContent(i+1,j+1)>0)  { sumTAU_pileuplow += hist2D_kTtrack_ztt_pileuplow->GetBinContent(i+1,j+1); }
      if (hist2D_kTtrack_ztt_pileuphigh->GetBinContent(i+1,j+1)>0) { sumTAU_pileuphigh += hist2D_kTtrack_ztt_pileuphigh->GetBinContent(i+1,j+1); }
    }
  }
  for (int i=0; i<NbinX; i++) {
    for (int j=0; j<NbinY; j++) {
      if (hist2D_kTtrack_ztt->GetBinContent(i+1,j+1)>0)  { dTAU[NbinY*i+j] = hist2D_kTtrack_ztt->GetBinContent(i+1,j+1)/sumTAU; }
      if (hist2D_kTtrack_qcd->GetBinContent(i+1,j+1)>0)  { dQCD[NbinY*i+j] = hist2D_kTtrack_qcd->GetBinContent(i+1,j+1)/sumQCD; }
      if (hist2D_kTtrack_h125->GetBinContent(i+1,j+1)>0) { dHIG[NbinY*i+j] = hist2D_kTtrack_h125->GetBinContent(i+1,j+1)/sumHIG; }
      if (hist2D_kTtrack_elwk->GetBinContent(i+1,j+1)>0) { dELW[NbinY*i+j] = hist2D_kTtrack_elwk->GetBinContent(i+1,j+1)/sumELW; }
      if (hist2D_kTtrack_qcdsyst->GetBinContent(i+1,j+1)>0)  { dQCD2[NbinY*i+j] = hist2D_kTtrack_qcdsyst->GetBinContent(i+1,j+1)/sumQCD2; }
      if (hist2D_kTtrack_ztt_pileuplow->GetBinContent(i+1,j+1)>0)  { dTAU_pileuplow[NbinY*i+j] = hist2D_kTtrack_ztt_pileuplow->GetBinContent(i+1,j+1)/sumTAU_pileuplow; }
      if (hist2D_kTtrack_ztt_pileuphigh->GetBinContent(i+1,j+1)>0) { dTAU_pileuphigh[NbinY*i+j] = hist2D_kTtrack_ztt_pileuphigh->GetBinContent(i+1,j+1)/sumTAU_pileuphigh; }

//      statQCD[NbinY*i+j] = hist2D_kTtrack_qcd->GetBinContent(i+1,j+1)/sumQCD*hist2D_kTtrack_qcd->GetEntries();
      statQCD[NbinY*i+j] = hist2D_kTtrack_qcdstat->GetBinContent(i+1,j+1);
      hist1D_track_TAU -> SetBinContent(NbinY*i+j+1,dTAU[NbinY*i+j]);
      hist1D_track_QCD -> SetBinContent(NbinY*i+j+1,dQCD[NbinY*i+j]);
      hist1D_track_ELW -> SetBinContent(NbinY*i+j+1,dELW[NbinY*i+j]);
    }
  }

  // Scaling EW background
  sumELW *= lumi;

  //====================================
  // Read real DATA or make pseudo-data
  //====================================
  sumDATA = 0;
  for (int i=0; i<NbinX; i++) {
    for (int j=0; j<NbinY; j++) {
      trackDATA[NbinY*i+j] = 0.0;
    }
  }

  TH1D *hist_Data = new TH1D("hist_Data","",60,0,300);

  // Read data
  if (imode==0) {
    for (int i=0; i<NbinX; i++) {
      for (int j=0; j<NbinY; j++) {
        trackDATA[NbinY*i+j] = hist2D_kTtrack_data->GetBinContent(i+1,j+1);
        sumDATA += trackDATA[NbinY*i+j];
      }
    }
  }

  // Or make pseudo-data
  if (imode==1) {
    for (int i=0; i<TMath::Nint(sumTAU); i++)  { sumDATA++; trackDATA[TMath::Nint(hist1D_track_TAU->GetRandom()-0.5)]++; }
    for (int i=0; i<TMath::Nint(sumQCD); i++)  { sumDATA++; trackDATA[TMath::Nint(hist1D_track_QCD->GetRandom()-0.5)]++; }
    for (int i=0; i<TMath::Nint(sumELW); i++)  { sumDATA++; trackDATA[TMath::Nint(hist1D_track_ELW->GetRandom()-0.5)]++; }
  }

  //==============
  // Do track fit
  //==============
  double fitQCD,errQCD;
  double cov11,cov11er;
  double cov12,cov12er;
  double cov13,cov13er;
  double cov21,cov21er;
  double cov22,cov22er;
  double cov23,cov23er;
  double cov31,cov31er;
  double cov32,cov32er;
  double cov33,cov33er;
  double qcdsh,qcdsher;
  double xelwk,xelwker;
  double pllw,pllwer;
  double plhg,plhger;

  double xLog1,xLog2,edm1,edm2,errdef;
  int nvpar,nparx,icstat;

  int ierflg = 0;
  double arglist[23];
  double vstart[23];
  double step[23];

//  vstart[0] = sumQCD/(sumTAU+sumQCD+sumELW);
  vstart[0] = 0.7;

  cout << "Starting fit param " << vstart[0] << " " << sumQCD << " " << sumTAU << " " << sumELW << endl;


  vstart[1] = statQCD[21];
  vstart[2] = statQCD[22];
  vstart[3] = statQCD[23];
  vstart[4] = statQCD[41];
  vstart[5] = statQCD[42];
  vstart[6] = statQCD[43];
  vstart[7] = statQCD[61];
  vstart[8] = statQCD[62];
  vstart[9] = statQCD[63];
  vstart[10] = 0.0;
  vstart[11] = 1.0;
  vstart[12] = 0.0;
  vstart[13] = 0.0;

  for (int i=0; i<14; i++) { step[i] = 0.001; }

  TMinuit *trkMinuit1 = new TMinuit(13);
  trkMinuit1->SetFCN(trackFit);
  trkMinuit1->mnparm( 0,"Fraction of QCD        ", vstart[0], step[0], 0.,0.,ierflg);
  trkMinuit1->mnparm( 1,"stat uncertainty (1,1) ", vstart[1], step[1], 0.,0.,ierflg);
  trkMinuit1->mnparm( 2,"stat uncertainty (1,2) ", vstart[2], step[2], 0.,0.,ierflg);
  trkMinuit1->mnparm( 3,"stat uncertainty (1,3) ", vstart[3], step[3], 0.,0.,ierflg);
  trkMinuit1->mnparm( 4,"stat uncertainty (2,1) ", vstart[4], step[4], 0.,0.,ierflg);
  trkMinuit1->mnparm( 5,"stat uncertainty (2,2) ", vstart[5], step[5], 0.,0.,ierflg);
  trkMinuit1->mnparm( 6,"stat uncertainty (2,3) ", vstart[6], step[6], 0.,0.,ierflg);
  trkMinuit1->mnparm( 7,"stat uncertainty (3,1) ", vstart[7], step[7], 0.,0.,ierflg);
  trkMinuit1->mnparm( 8,"stat uncertainty (3,2) ", vstart[8], step[8], 0.,0.,ierflg);
  trkMinuit1->mnparm( 9,"stat uncertainty (3,3) ", vstart[9], step[9], 0.,0.,ierflg);
  trkMinuit1->mnparm(10,"QCD shape uncertainty  ",vstart[10],step[10], 0.,0.,ierflg);
  trkMinuit1->mnparm(11,"Electroweak            ",vstart[11],step[11], 0.,0.,ierflg);
  trkMinuit1->mnparm(12,"signal pileuo low      ",vstart[12],step[12], 0.,0.,ierflg);
  trkMinuit1->mnparm(13,"signal pileuo high     ",vstart[13],step[13], 0.,0.,ierflg);

  // Systematics for statistical fluctuation
//  arglist[0] =   2; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
//  arglist[0] =   3; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
//  arglist[0] =   4; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
//  arglist[0] =   5; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
//  arglist[0] =   6; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
//  arglist[0] =   7; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
//  arglist[0] =   8; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
//  arglist[0] =   9; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
  arglist[0] =  11; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);

  arglist[0] =  13; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);
  arglist[0] =  14; trkMinuit1->mnexcm("FIX", arglist ,1,ierflg);

  arglist[0] = 0.0; trkMinuit1->mnexcm("MINI", arglist ,0,ierflg);
  arglist[0] = 1000; trkMinuit1->mnexcm("MIGRAD", arglist ,0,ierflg);
//  arglist[0] = 0.0; arglist[1] = 9.0; trkMinuit1->mnexcm("MINOS", arglist ,2,ierflg);
  trkMinuit1->GetParameter(0,fitQCD,errQCD);
  trkMinuit1->GetParameter(1,cov11,cov11er);
  trkMinuit1->GetParameter(2,cov12,cov12er);
  trkMinuit1->GetParameter(3,cov13,cov13er);
  trkMinuit1->GetParameter(4,cov21,cov21er);
  trkMinuit1->GetParameter(5,cov22,cov22er);
  trkMinuit1->GetParameter(6,cov23,cov23er);
  trkMinuit1->GetParameter(7,cov31,cov31er);
  trkMinuit1->GetParameter(8,cov32,cov32er);
  trkMinuit1->GetParameter(9,cov33,cov33er);
  trkMinuit1->GetParameter(10,qcdsh,qcdsher);
  trkMinuit1->GetParameter(11,xelwk,xelwker);
  trkMinuit1->GetParameter(12,pllw,pllwer);
  trkMinuit1->GetParameter(13,plhg,plhger);

  trkMinuit1->mnstat(xLog1,edm1,errdef,nvpar,nparx,icstat);
  // Print result
  //    trkMinuit1->mnprin(3,amin);
//  trkMinuit1->mnerrs(4,effTauMediumErrR,effTauMediumErrL,eparab,globcc);
  trkMinuit1->mnexcm("STOP",arglist,0,ierflg);
  trkMinuit1->Delete();

  if (statQCD[21]>0) { dQCD[21] *= cov11/statQCD[21]; }
  if (statQCD[22]>0) { dQCD[22] *= cov12/statQCD[22]; }
  if (statQCD[23]>0) { dQCD[23] *= cov13/statQCD[23]; }
  if (statQCD[41]>0) { dQCD[41] *= cov21/statQCD[41]; }
  if (statQCD[42]>0) { dQCD[42] *= cov22/statQCD[42]; }
  if (statQCD[43]>0) { dQCD[43] *= cov23/statQCD[43]; }
  if (statQCD[61]>0) { dQCD[61] *= cov31/statQCD[61]; }
  if (statQCD[62]>0) { dQCD[62] *= cov32/statQCD[62]; }
  if (statQCD[63]>0) { dQCD[63] *= cov33/statQCD[63]; }

  for (int i=0; i<400; i++) {
    dQCD[i] += qcdsh*(dQCD2[i]-dQCD[i]);
    dTAU[i] += pllw*(dTAU_pileuplow[i]-dTAU[i]);
    dTAU[i] += plhg*(dTAU_pileuphigh[i]-dTAU[i]);
  }

  double fQCD13 = dQCD[21]+dQCD[23]+dQCD[61]+dQCD[63];
  double fTAU13 = dTAU[21]+dTAU[23]+dTAU[61]+dTAU[63];
  double fHIG13 = dHIG[21]+dHIG[23]+dHIG[61]+dHIG[63];
  double fELW13 = dELW[21]+dELW[23]+dELW[61]+dELW[63];

//  double obsQCD = sumDATA*fitQCD*fQCD13;
  double obsQCD = sumDATA*fitQCD;
  double obsH   = sumHIG;
  double obsEWK = xelwk*sumELW;
  double obsZ   = sumDATA-obsQCD-obsEWK;

  printf("\n");
  printf("Fitting result  :  %9.4f +- %9.4f\n",fitQCD,errQCD);
  printf("Pileup low      :  %9.4f\n",pllw);
  printf("Pileup high     :  %9.4f\n",plhg);
  printf("QCD shape       :  %9.4f\n",qcdsh);
  printf("cov (1,1)       :  %9.4f +- %9.4f ( %9.4f )\n",cov11,cov11er,statQCD[21]);
  printf("cov (1,2)       :  %9.4f +- %9.4f ( %9.4f )\n",cov12,cov12er,statQCD[22]);
  printf("cov (1,3)       :  %9.4f +- %9.4f ( %9.4f )\n",cov13,cov13er,statQCD[23]);
  printf("cov (2,1)       :  %9.4f +- %9.4f ( %9.4f )\n",cov21,cov21er,statQCD[41]);
  printf("cov (2,2)       :  %9.4f +- %9.4f ( %9.4f )\n",cov22,cov22er,statQCD[42]);
  printf("cov (2,3)       :  %9.4f +- %9.4f ( %9.4f )\n",cov23,cov23er,statQCD[43]);
  printf("cov (3,1)       :  %9.4f +- %9.4f ( %9.4f )\n",cov31,cov31er,statQCD[61]);
  printf("cov (3,2)       :  %9.4f +- %9.4f ( %9.4f )\n",cov32,cov32er,statQCD[62]);
  printf("cov (3,3)       :  %9.4f +- %9.4f ( %9.4f )\n",cov33,cov33er,statQCD[63]);
  printf("---------------------------------\n");
  printf("Expected QCD    :  %9.4f +- %9.4f\n",obsQCD,obsQCD*errQCD/fitQCD);
  printf("Expected Z      :  %9.4f +- %9.4f\n",obsZ,obsZ*errQCD/fitQCD);
  printf("Expected Higgs  :  %9.4f\n",obsH);
  printf("Expected ELWK   :  %9.4f\n",obsEWK);
  printf("Observed data   :  %9.4f\n",sumDATA);
  printf("\n");
  printf("Used QCD events :  %9.4f +- %9.4f\n",obsQCD*fQCD13,obsQCD*errQCD/fitQCD*fQCD13);
  printf("Used Z events   :  %9.4f +- %9.4f\n",obsZ*fTAU13,obsZ*errQCD/fitQCD*fTAU13);
  printf("Used Higgs      :  %9.4f\n",obsH*fHIG13);
  printf("Used ELWK       :  %9.4f\n",obsEWK*fELW13);
  printf("fQCD13          :  %9.4f\n",fQCD13);
  printf("fTAU13          :  %9.4f\n",fTAU13);
  printf("\n");
  printf("Extrapolation\n");
  printf("Used QCD events :  %9.4f +- %9.4f\n",obsQCD*fQCD13*hist2D_kTtrack_qcd_probe->GetSum()/hist2D_kTtrack_qcd->GetSum(),obsQCD*errQCD/fitQCD*fQCD13*hist2D_kTtrack_qcd_probe->GetSum()/hist2D_kTtrack_qcd->GetSum());
  printf("Used Z events   :  %9.4f +- %9.4f\n",obsZ*fTAU13*hist2D_kTtrack_ztt_probe->GetSum()/hist2D_kTtrack_ztt->GetSum(),obsZ*errQCD/fitQCD*fTAU13*hist2D_kTtrack_ztt_probe->GetSum()/hist2D_kTtrack_ztt->GetSum());
  printf("Used Higgs      :  %9.4f\n",obsH*fHIG13);
  printf("Used ELWK       :  %9.4f\n",obsEWK*fELW13);
	  
   Fraction1P3P("Fraction of QCD f(QCD)  ", hist2D_kTtrack_qcd, fitQCD);


  ffl -> Close();
//  ps  -> Close();
}
///       .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.
/// 	:::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\.
/// 	      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'
void Fraction1P3P(const string&s, TH2F *h, float fitQCD){
	
	//Soshi:
	//
	// the "fitQCD" is the fraction of QCD events without OS requirement
	//
	// the fraction of the QCD in the OS events:
	// f(QCD) = fitQCD x N(track template; 1p/3p-QCD) / N(track template; QCD) 
	//
	// Then, in the OS events, you need to normalize QCD with
	//N(QCD) = f(QCD) x N(Data) .
	//
	int b1 = 2;
	double bc1p1p = h->GetBinContent(b1, b1);
	double bc1p3p = h->GetBinContent(b1, b1+2);
	double bc3p1p = h->GetBinContent(b1+2, b1);
	double bc3p3p = h->GetBinContent(b1+3, b1+3);
	  

	double bc_1p3p =bc1p1p+bc1p3p+bc3p1p+bc3p3p;
	
	double area = h->Integral(-1,-1,-1,-1, "");
	cout<<endl;
	cout<<s<<endl;
	cout<<"N(track template; QCD) = "<<area<<endl;
	cout<<"N(track template; 1p/3p-QCD) = "<<bc_1p3p<<endl;
	double f = fitQCD * bc_1p3p / area;
	cout<<"f(QCD) = fitQCD x N(track template; 1p/3p-QCD) / N(track template; QCD) = "<<f<<endl;	

}
