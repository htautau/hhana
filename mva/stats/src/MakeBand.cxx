
#include "TTree.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include "TH1.h"
#include <stdlib.h>
#include <sstream>
#include "THStack.h"
#include "TSystem.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TObjArray.h"
#include "TString.h"
#include "TGraph.h"
#include "TLine.h"
#include "TLegend.h"
#include "TGraphAsymmErrors.h"
#include "include/MyPlot.h"
#include "TLatex.h"
#include "TMarker.h"
#include "Math/DistFunc.h"
using namespace std;

#include "AtlasStyle.C"

int main(int argc, char**argv)
{
  SetAtlasStyle();
  vector<float> masspoints;
  vector<TFile* > limitfiles;
  
  TString PlotName;
  TString outFileName;

  if(TString(argv[1]).Contains("--Ratio")){
    int N= (argc-2)/2;

    cout<<"Preparing ratio"<<endl;
    TString plotName=argv[2+N];
    plotName.ReplaceAll("excl","ratio");
    plotName.ReplaceAll(".root",".eps");
    MyPlot Plot("RatioPlot",0,"Ratio");
    Plot.setYTitle("95% CL limit on #sigma/#sigma_{SM}");
    Plot.setXTitle("m_{H} [GeV]");
    Plot.setRmin(0.75);
    Plot.setRmax(1.25);

  TLegend leg(0.45,0.75,0.8,0.93);
  leg.SetFillColor(0); leg.SetBorderSize(0);
  leg.SetFillStyle(0);

  // leg.AddEntry(&g_exp,"expected","l");

  

    //  Plot.addLegend();
    for(int iA=2+N;iA<argc;iA++){
      TString fileName=argv[iA];
      TFile* f= new TFile(fileName);
      TGraph* g_exp=(TGraph*)f->Get("g_exp");
    
      TGraphAsymmErrors* g_1sigma=(TGraphAsymmErrors*)f->Get("g_1sigma");
      TGraphAsymmErrors* g_2sigma=(TGraphAsymmErrors*)f->Get("g_2sigma");
      
      g_1sigma->SetFillColor(kGreen);
      g_2sigma->SetFillColor(kYellow);
      g_exp->SetLineStyle(iA-N);
      g_exp->SetLineColor(iA-N-1);
      if(iA-N-1==3)       g_exp->SetLineColor(6);
      if(iA==11)       g_exp->SetLineColor(6);

      leg.AddEntry(g_exp,argv[ iA-N],"l");
      
      g_1sigma->SetLineStyle(iA-N);
      g_2sigma->SetLineStyle(iA-N);
      g_1sigma->SetTitle("#pm 1 #sigma");
      g_2sigma->SetTitle("#pm 2 #sigma");
      g_exp->SetTitle("expected");
      g_exp->GetYaxis()->SetRangeUser(0,15);
      g_exp->GetXaxis()->SetTitle("m_{H} [GeV]");
      Plot.add(g_exp,"L");
      if(iA==2+N){
	g_exp->SetLineColor(1);
	Plot.add(g_2sigma,"3,NORATIO");
	Plot.add(g_1sigma,"3,NORATIO");

	leg.AddEntry(g_1sigma,"expected #pm 1 #sigma","lf");
	leg.AddEntry(g_2sigma,"expected #pm 2 #sigma","lf");
      }
      
    }//args
    TCanvas* c=    Plot.getCanvas();
    TGraph g(1);
    g.SetPoint(0,125,2.4 );
    g.SetMarkerStyle(20);
    g.Draw("P");
    leg.Draw();
    c->Print(plotName);
    plotName.ReplaceAll(".eps",".pdf");
    c->Print(plotName);

  }


  // Make plot of significance
  else if( TString(argv[1]).Contains("P0")){
    for(int iA=1;iA<argc;iA++){
      TString fileName=argv[iA];
      TObjArray* arr=   fileName.Tokenize("_");

    for(int iT=0;iT<arr->GetSize();iT++){
      if(  arr->At(iT) && ((TObjString*) arr->At(iT))->String().IsDigit()){
	masspoints.push_back(	((TObjString*) arr->At(iT))->String().Atoi());
	break;
      }
    }
    cout<<"Mass:"<<masspoints.back() <<endl;
    TString massString;
   
    massString+=masspoints.back();
    if(iA==1){
      PlotName=fileName;
      PlotName=PlotName.ReplaceAll(massString,"significance");
      PlotName=PlotName.ReplaceAll(".root",".eps");
      outFileName=PlotName;
      //      outFileName=outFileName.ReplaceAll(".eps",".root");
      outFileName=outFileName.ReplaceAll(".eps","_SIG.root");
    }
    //    cout<<argv[iA]<<endl;
    cout<<fileName<<endl;    
    TFile* f= new TFile(fileName);
   
    limitfiles.push_back(f);
  }
  cout <<"loaded files.."<<endl;
  TCanvas c("sig","sig",500,500);
  TGraph g_obs(masspoints.size());
  TGraph g_exp(masspoints.size());
  TGraphAsymmErrors g_1sigma(masspoints.size());
  TGraphAsymmErrors g_2sigma(masspoints.size());

  for(int iM=0;iM<masspoints.size();iM++){
    TH1D* h=(TH1D*) limitfiles[iM]->Get("hypo");
    if(!h) cout<<"could not find limit histo for mass"<<masspoints[iM]<<endl;
    g_obs.SetPoint(iM,masspoints[iM],ROOT::Math::gaussian_cdf_c(h->GetBinContent(1)));
    g_exp.SetPoint(iM,masspoints[iM],ROOT::Math::gaussian_cdf_c(h->GetBinContent(2)));    
  }
  
  cout <<"Made TGraph.."<<endl;

  g_exp.SetLineStyle(2);
  g_obs.SetLineStyle(1);
  g_obs.SetMarkerSize(0.8);

  TH1D h("axis","axis",1000,-500,500);
   h.GetXaxis()->SetRangeUser(100,150);
   // h.GetXaxis()->SetRangeUser(-1,1);
  h.Draw("AXIS");
  h.GetXaxis()->SetRangeUser(100,150);
  h.GetXaxis()->SetRangeUser(100,150);
  h.GetYaxis()->SetRangeUser(10E-5,100);
  h.GetYaxis()->SetTitle("P_{0}");
  h.GetXaxis()->SetTitle("m_{H} [GeV]");
  
  

  g_exp.Draw("sameL");

  // g_exp.GetYaxis()->SetTitle("95% CL limit on #sigma/#sigma_{SM}");
  // g_exp.GetXaxis()->SetTitle("m_{H} [GeV]");
  
  g_exp.Draw("sameL");
  g_obs.Draw("sameLP");
  
  TLine line;
  line.SetLineStyle(2);
  line.SetLineColor(2);
  line.DrawLine(100,ROOT::Math::gaussian_cdf_c(0), 150, ROOT::Math::gaussian_cdf_c(0)  );
  line.DrawLine(100,ROOT::Math::gaussian_cdf_c(1), 150, ROOT::Math::gaussian_cdf_c(1)  );
  line.DrawLine(100,ROOT::Math::gaussian_cdf_c(2), 150, ROOT::Math::gaussian_cdf_c(2)  );
  line.DrawLine(100,ROOT::Math::gaussian_cdf_c(3), 150, ROOT::Math::gaussian_cdf_c(3)  );
  //line.DrawLine(100,ROOT::Math::gaussian_cdf_c(4), 150, ROOT::Math::gaussian_cdf_c(4)  );
  //  line.DrawLine(100,ROOT::Math::gaussian_cdf_c(5), 150, ROOT::Math::gaussian_cdf_c(5)  );


  TLegend leg(0.7,0.82,0.89,0.92);
  leg.SetFillColor(0); leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.AddEntry(&g_exp,"expected","l");
  leg.AddEntry(&g_obs,"observed","pl");

  leg.Draw();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.03);
  double m_lumi=20300;
  TString lumi=Form("#font[42]{ #scale[0.6]{#int} L dt = %.1f fb^{-1}}",m_lumi/1000.);
  latex.SetTextSize(0.027);
  //  latex.DrawLatex(0.8,0.9 ,"#l + #tau_{h}"); 
  latex.SetTextSize(0.03);
  latex.DrawLatex(0.2,0.9 ,"#font[72]{ATLAS}#font[42]{  Internal} ");
  latex.DrawLatex(0.2,0.85 ,lumi); 
  latex.DrawLatex(0.2,0.80 ,"#sqrt{#it{s}} = 8 TeV ");


  latex.SetNDC(false);
  latex.SetTextSize(0.02);
  latex.SetTextColor(2);
  latex.DrawLatex(152,ROOT::Math::gaussian_cdf_c(1),"1#sigma");
  latex.DrawLatex(152,ROOT::Math::gaussian_cdf_c(2),"2#sigma");
  latex.DrawLatex(152,ROOT::Math::gaussian_cdf_c(3),"3#sigma");
  //  latex.DrawLatex(152,ROOT::Math::gaussian_cdf_c(4),"4#sigma");

  c.SetLogy();
  
  c.RedrawAxis();
  c.Print(PlotName);
  
 

  TFile* outFile=new TFile(outFileName,"recreate");
  g_exp.Write("g_exp");

  outFile->Close();  

  
  }

  else{
  for(int iA=1;iA<argc;iA++){
    // cout<<"Argument "<<iA<<endl;
    TString fileName=argv[iA];
    TObjArray* arr=   fileName.Tokenize("_");
    //    TObjArray* arr=   fileName.Tokenize("B_");
    
    for(int iT=0;iT<arr->GetSize();iT++){
      //  cout<< ((TObjString*) arr->At(iT))->String()<<endl;
      
     //  if( TString(( (TObjString*) arr->At(iT))->String())=="NOM" ){
// 	masspoints.push_back(150); break;
//       }
      
//       if( TString(( (TObjString*) arr->At(iT))->String())=="M10" ){
// 	masspoints.push_back(-10); break;}
//       if( TString(( (TObjString*) arr->At(iT))->String())=="M20" ){
// 	masspoints.push_back(-20); break;}
//       if( TString(( (TObjString*) arr->At(iT))->String())=="M30" ){
// 	masspoints.push_back(-30); break;}
//       if( TString(( (TObjString*) arr->At(iT))->String())=="M40" ){
// 	masspoints.push_back(-40); break;}
      

//       if(  arr->At(iT) && ((TObjString*) arr->At(iT))->String().IsFloat()){
// 	if (((TObjString*) arr->At(iT))->String().Atof() < 100){
// 	  masspoints.push_back(	((TObjString*) arr->At(iT))->String().Atof());
// 	  break;}
//       }

      if(  arr->At(iT) && ((TObjString*) arr->At(iT))->String().IsDigit()){
	masspoints.push_back(	((TObjString*) arr->At(iT))->String().Atoi());
	break;
      }
    }
    cout<<"Mass:"<<masspoints.back() <<endl;

    TString massString;
   
    massString+=masspoints.back();
    if(iA==1){
      PlotName=fileName;
      PlotName=PlotName.ReplaceAll(massString,"excl");
      PlotName=PlotName.ReplaceAll(".root",".eps");
      outFileName=PlotName;
      //      outFileName=outFileName.ReplaceAll(".eps",".root");
      outFileName=outFileName.ReplaceAll(".eps","_LIMIT.root");
    }
    //    cout<<argv[iA]<<endl;
    cout<<fileName<<endl;    
    TFile* f= new TFile(fileName);
   
    limitfiles.push_back(f);
  }
  cout <<"loaded files.."<<endl;
  TCanvas c("limit","limit",500,500);
  TGraph g_obs(masspoints.size());
  TGraph g_exp(masspoints.size());
  TGraphAsymmErrors g_1sigma(masspoints.size());
  TGraphAsymmErrors g_2sigma(masspoints.size());

  for(int iM=0;iM<masspoints.size();iM++){
    TH1D* h=(TH1D*) limitfiles[iM]->Get("limit");
    if(!h) cout<<"could not find limit histo for mass"<<masspoints[iM]<<endl;
    g_obs.SetPoint(iM,masspoints[iM],h->GetBinContent(1));
    g_exp.SetPoint(iM,masspoints[iM],h->GetBinContent(2));
    g_1sigma.SetPoint(iM,masspoints[iM],h->GetBinContent(2));
    g_2sigma.SetPoint(iM,masspoints[iM],h->GetBinContent(2));
    
//     g_1sigma.SetPointEYlow(iM, 0.1);
//     g_1sigma.SetPointEYhigh(iM, 0.2);

    g_1sigma.SetPointEYhigh(iM, h->GetBinContent(4)-h->GetBinContent(2));
    g_1sigma.SetPointEYlow(iM, h->GetBinContent(2)-h->GetBinContent(5));

    g_2sigma.SetPointEYhigh(iM, h->GetBinContent(3)-h->GetBinContent(2));
    g_2sigma.SetPointEYlow(iM, h->GetBinContent(2)-h->GetBinContent(6));
    
  }

   cout <<"Made TGraph.."<<endl;
   
  


  g_1sigma.SetFillColor(kGreen);
  g_2sigma.SetFillColor(kYellow);
  g_exp.SetLineStyle(2);
  g_obs.SetLineStyle(1);
  g_obs.SetMarkerSize(0.8);
  g_1sigma.SetLineStyle(2);
  g_2sigma.SetLineStyle(2);

  TH1D h("axis","axis",1000,-500,500);
   h.GetXaxis()->SetRangeUser(100,150);
   // h.GetXaxis()->SetRangeUser(-1,1);
  h.Draw("AXIS");
  h.GetXaxis()->SetRangeUser(100,150);
  h.GetXaxis()->SetRangeUser(100,150);
  h.GetYaxis()->SetRangeUser(0,5);
  h.GetYaxis()->SetTitle("95% CL limit on #sigma/#sigma_{SM}");
  h.GetXaxis()->SetTitle("m_{H} [GeV]");
  
  

  g_exp.Draw("sameL");
  //g_exp.GetYaxis()->SetRangeUser(10,40);
  //g_exp.GetXaxis()->SetRangeUser(100,150);
  //  g_exp.GetXaxis()->SetRangeUser(-40,150);
  g_exp.GetYaxis()->SetTitle("95% CL limit on #sigma/#sigma_{SM}");
  g_exp.GetXaxis()->SetTitle("m_{H} [GeV]");
  
  g_2sigma.Draw("same3");
  g_1sigma.Draw("same3");
  g_exp.Draw("sameL");
  g_obs.Draw("samePL");
  
  TLine line;
  line.SetLineStyle(2);
  line.SetLineColor(2);
  line.DrawLine(100,1, 150,1.  );

  TLegend leg(0.5,0.7,0.75,0.85);
  leg.SetFillColor(0); leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  // leg.AddEntry(&g_exp,"expected","l");
  leg.AddEntry(&g_obs,"observed","pl");
  leg.AddEntry(&g_1sigma,"expected #pm 1 #sigma","lf");
  leg.AddEntry(&g_2sigma,"expected #pm 2 #sigma","lf");
  leg.Draw();
 TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.03);
  double m_lumi=20300;
  TString lumi=Form("#font[42]{ #scale[0.6]{#int} L dt = %.1f fb^{-1}}",m_lumi/1000.);
  
  latex.SetTextSize(0.027);
  //  latex.DrawLatex(0.8,0.9 ,"#l + #tau_{h}"); 
  latex.SetTextSize(0.03);
  latex.DrawLatex(0.2,0.9 ,"#font[72]{ATLAS}#font[42]{  Internal} ");
  latex.DrawLatex(0.2,0.85 ,lumi); 
  latex.DrawLatex(0.2,0.80 ,"#sqrt{#it{s}} = 8 TeV ");

  // latex.SetNDC(false);
  // latex.SetTextSize(0.02);
  // latex.SetTextColor(2);
  // latex.DrawLatex(152,1,"1#sigma_{SM}");

  c.RedrawAxis();
  c.Print(PlotName);
  
 

  TFile* outFile=new TFile(outFileName,"recreate");
  g_exp.Write("g_exp");
  g_1sigma.Write("g_1sigma");
  g_2sigma.Write("g_2sigma");
  outFile->Close();
  }
}
