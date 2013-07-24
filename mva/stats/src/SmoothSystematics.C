// c++
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

// root
#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include <TMath.h>



// ===== Function definition ====================
TH1* EqualArea(TH1* n, TH1* s, float frac=0.5);
TH1* EqualAreaGabriel(TH1* n, TH1* s);
void MakeATest();
TString MakeFname(TString cat, TString syst);
// ==============================================



void RedoInputFiles(TString outputdir, int method){

    vector<TString> AllProc, AllCat, AllSyst;
    AllProc.clear(); AllCat.clear(); AllSyst.clear();

    // Processes
    AllProc.push_back("Data");
    AllProc.push_back("SameSign");
    AllProc.push_back("Top_OTHERS");
    AllProc.push_back("ZllAllFilters_Jet2Tau_OTHERS");
    AllProc.push_back("ZllAllFilters_l2Tau_OTHERS");
    AllProc.push_back("diboson_OTHERS");
    AllProc.push_back("W_Jets_AllFilters");
    AllProc.push_back("Emb");
    AllProc.push_back("SignalggF_125");
    AllProc.push_back("SignalVBF_125");
    AllProc.push_back("SignalWH_125");
    AllProc.push_back("SignalZH_125");

    // Categories
    AllCat.push_back("VBFMerged");
    //AllCat.push_back("Boost_Merged");

    // Systematics
    AllSyst.push_back("EES_PSStat");
    AllSyst.push_back("EES_R12");
    AllSyst.push_back("EES_Zee");
    AllSyst.push_back("ER");
    AllSyst.push_back("ISOL");
    AllSyst.push_back("JESB");
    AllSyst.push_back("JESDET");
    AllSyst.push_back("JESETAINTERMODEL");
    AllSyst.push_back("JESETAINTERSTAT");
    AllSyst.push_back("JESFLAVRESP");
    AllSyst.push_back("JESFLAV");
    AllSyst.push_back("JESMIX");
    AllSyst.push_back("JESMODEL");
    AllSyst.push_back("JESMU");
    AllSyst.push_back("JESNPV");
    AllSyst.push_back("JESPUPT");
    AllSyst.push_back("JESPURHO");
    //AllSyst.push_back("JESSINGLEP");
    AllSyst.push_back("JESSTAT");
    AllSyst.push_back("METRESO");
    AllSyst.push_back("MET");
    AllSyst.push_back("MS");
    AllSyst.push_back("TES");
    AllSyst.push_back("WPT");


    for (unsigned icat=0 ; icat<AllCat.size() ; icat++){
        for (unsigned isyst=0 ; isyst<AllSyst.size() ; isyst++){

            TFile *fNO = new TFile( outputdir+"/"+MakeFname(AllCat[icat],"NOM")         , "RECREATE" );
            TFile *fDO = new TFile( outputdir+"/"+MakeFname(AllCat[icat],AllSyst[isyst]+ "DOWN") , "RECREATE" );
            TFile *fUP = new TFile( outputdir+"/"+MakeFname(AllCat[icat],AllSyst[isyst]+ "UP"  ) , "RECREATE" );

            for (unsigned iproc=0 ; iproc<AllProc.size() ; iproc++){

                // 1 - Load varied histogram for each proc x cat
                TFile *f = NULL;
                f = TFile::Open("../ws_inputs/"+MakeFname(AllCat[icat], "NOM"));
                TH1F  *hnom  = (TH1F*) f->Get(AllProc[iproc]);

                f = TFile::Open("../ws_inputs/"+MakeFname(AllCat[icat], AllSyst[isyst]+"UP"));
                TH1F  *hup  = (TH1F*) f->Get(AllProc[iproc]);

                f  = TFile::Open("../ws_inputs/"+MakeFname(AllCat[icat], AllSyst[isyst]+"DOWN"));
                TH1F  *hdo  = (TH1F*) f->Get(AllProc[iproc]);

                // 2 - Make the new varied histograms
                TH1F *hup_new,*hdo_new;
                if (method==0){
                    hup_new = hup;
                    hdo_new = hdo;
                }
                else if (method==1){
                    hup_new = (TH1F*) EqualAreaGabriel(hnom,hup);
                    hdo_new = (TH1F*) EqualAreaGabriel(hnom,hdo);
                }
                else if (method==2){
                    hup_new = (TH1F*) EqualArea(hnom,hup, 1.5);
                    hdo_new = (TH1F*) EqualArea(hnom,hdo, 1.5);
                }
                else {
                    hup_new = hup;
                    hdo_new = hdo;
                }


                // 3 -Rename and save them in the output root file
                hnom->SetName(AllProc[iproc]);
                hup_new->SetName(AllProc[iproc]);
                hdo_new->SetName(AllProc[iproc]);
                fNO->cd(); hnom->Write(); 
                fDO->cd(); hdo_new->Write(); 
                fUP->cd(); hup_new->Write(); 
            }

            fNO->Close();
            fUP->Close();
            fDO->Close();

        }
    }

    return;
}




void MakeATest(){

    // proc x cat x syst
    TFile *f = NULL;
    f = TFile::Open("../ws_inputs/bdt_NOMBDT_Boost_Merged.root");
    TH1F  *hnom  = (TH1F*) f->Get("Emb");
    //hnom->Rebin(10);

    f = TFile::Open("../ws_inputs/bdt_TESUPBDT_Boost_Merged.root");
    TH1F  *hup  = (TH1F*) f->Get("Emb");
    //hup->Rebin(10);

    f  = TFile::Open("../ws_inputs/bdt_TESDOWNBDT_Boost_Merged.root");
    TH1F  *hdo  = (TH1F*) f->Get("Emb");
    //hdo->Rebin(10);

    /*
       TH1F *hup = (TH1F*) hnom->Clone();
       hup->Reset();
       TH1F *hdo = (TH1F*) hnom->Clone();
       hdo->Reset();

       for (int ib=0 ; ib<hnom->GetNbinsX()+1 ; ib++){
       double scale = (double (ib)+0.5) /(hnom->GetNbinsX()+1) * 0.01;
       hup->SetBinContent(ib, hnom->GetBinContent(ib) * (1+scale) );
       hdo->SetBinContent(ib, hnom->GetBinContent(ib) * (1-scale) );
       hup->SetBinError(ib, hnom->GetBinError(ib));
       hdo->SetBinError(ib, hnom->GetBinError(ib));

       double err = hnom->GetBinError(ib);
       hnom->SetBinError(ib, err);

       }
       */

    TH1F *hup_new = (TH1F*) EqualArea(hnom,hup, 1.5);
    TH1F *hdo_new = (TH1F*) EqualArea(hnom,hdo, 1.5);

    TCanvas *can = new TCanvas("Systematic","Systematic",1100,400);
    can->Divide(2,1);
    can->cd(1);
    hup->Divide(hnom);
    hdo->Divide(hnom);
    hup->Draw();
    hdo->Draw("same");
    hnom->Draw("same");

    can->cd(2);
    hup_new->Divide(hnom);
    hdo_new->Divide(hnom);
    hup_new->Draw();
    hdo_new->Draw("same");
    hnom->Draw("same");

    return;

}


TH1* EqualArea(TH1* n, TH1* s, float frac) {

    // find boundaries where integral = frac of max
    vector< pair<int, int> > bins;

    // find max of first set
    int maxBin(23);
    float tot(0), tots(0);
    float err(0);

    for(int b=1; b<n->GetNbinsX()+1; b++) {

        if(fabs(s->GetBinContent(b)-n->GetBinContent(b)) / n->GetBinError(b) < frac){
            //if(n->GetBinError(b) / n->GetBinContent(b) > frac){
            tot = 0;
            err = 0;
            for(int c=b; c<maxBin+1; c++) {
                tot  += n->GetBinContent(c);
                tots += s->GetBinContent(c);
                err = sqrt(err*err + n->GetBinError(c)*n->GetBinError(c));

                // expanding integral until cover fraction & picks up tails
                //if( err / tot < frac ) {
                if( fabs(tots-tot) / err < frac ) {
                    // need to catch tails of distribution and do not want to cut off too soon
                    // if integral of remaining bins is < frac of max && that + region in question < max - take it all
                    float tmpTot(0),tmpTotS(0);
                    float tmpErr(0),tmpErrS(0);
                    for(int d=c+1; d<maxBin+1; d++) {
                        tmpTot += n->GetBinContent(d);
                        tmpErr += sqrt(tmpErr*tmpErr + n->GetBinError(d)*n->GetBinError(d));
                        tmpTotS += s->GetBinContent(d);
                        tmpErrS = sqrt(tmpErr*tmpErr + s->GetBinError(d)*s->GetBinError(d));
                    }
                    //if( tmpErr / tmpTot > frac) {
                    if( fabs(tmpTot-tmpTotS) / tmpErr > frac) {
                        c = maxBin;
                    }

                    bins.push_back( make_pair(b,c) );
                    b = c; // advance outer loop
                    break;
                }

                // save last pair before loop completes
                if(c==maxBin){
                    bins.push_back( make_pair(b,c) );
                    b=c;
                    break;
                } // if loop ending
            } // second loop
        } // if content < frac of max
        else{
            bins.push_back( make_pair(b,b) );
        }
        if(b==maxBin) { // in the last bin
            maxBin += 23;
        }
    } // outer loop over bins

    TString newName("new_");
    newName.Append(s->GetName());
    TH1F* newSyst = new TH1F(newName, newName, s->GetNbinsX(), s->GetBinLowEdge(1), s->GetBinLowEdge( s->GetNbinsX()+1 ));
    newSyst->Sumw2();

    for(unsigned int i=0; i<bins.size(); i++) {
        float nom=n->Integral( (bins.at(i)).first, (bins.at(i)).second );
        double sysErr(0);
        float sys=s->IntegralAndError( (bins.at(i)).first, (bins.at(i)).second , sysErr);

        // fill new systematic hist
        for(int b=(bins.at(i)).first; b<(bins.at(i)).second+1; b++) {
            //newSyst->SetBinContent(b, sys * n->GetBinContent(b) / nom );
            //newSyst->SetBinError(b  , sysErr * n->GetBinContent(b) / nom);

            if(nom>0) {
                newSyst->SetBinContent(b, (sys/nom) * n->GetBinContent(b));
                //newSyst->SetBinError(b, sysErr);
            } else {
                newSyst->SetBinContent(b, nom);
                //newSyst->SetBinError(b,sysErr);
            }

        }
    } // loop over equal area bins
    return newSyst;
}


TH1* EqualAreaGabriel(TH1* n, TH1* s) {
    //cout << "EQUAL AREA" << endl;
    float frac = 0.05; // minimal statistical error

    // find boundaries where integral = frac of max
    vector< pair<int, int> > bins;

    // find max of first set
    int maxBin(23);
    float tot(0);
    float err(0);

    for(int b=1; b<n->GetNbinsX()+1; b++) {
        //cout << b <<  "\t" << maxBin << "\t" << n->GetBinError(b) / n->GetBinContent(b) << endl;
        if(n->GetBinError(b) / n->GetBinContent(b) > frac){
            tot = 0;
            err = 0;
            for(int c=b; c<maxBin+1; c++) {
                tot += n->GetBinContent(c);
                err = sqrt(err*err + n->GetBinError(c)*n->GetBinError(c));
                //cout << "\t" << err / tot << endl;
                // expanding integral until cover fraction & picks up tails
                if( err / tot < frac ) {
                    // need to catch tails of distribution and do not want to cut off too soon
                    // if integral of remaining bins is < frac of max && that + region in question < max - take it all
                    float tmpTot(0);
                    float tmpErr(0);
                    for(int d=c+1; d<maxBin+1; d++) {
                        tmpTot += n->GetBinContent(d);
                        tmpErr = sqrt(tmpErr*tmpErr + n->GetBinError(d)*n->GetBinError(d));
                    }
                    if( tmpErr / tmpTot > frac) {
                        c = maxBin;
                    }

                    bins.push_back( make_pair(b,c) );
                    b = c; // advance outer loop
                    break;
                }

                // save last pair before loop completes
                if(c==maxBin){
                    bins.push_back( make_pair(b,c) );
                    b=c;
                    break;
                } // if loop ending
            } // second loop
        } // if content < frac of max
        else{
            bins.push_back( make_pair(b,b) );
        }
        if(b==maxBin) { // in the last bin
            maxBin += 23;
        }
    } // outer loop over bins

    TString newName("new_");
    newName.Append(s->GetName());
    TH1F* newSyst = new TH1F(newName, newName, s->GetNbinsX(), s->GetBinLowEdge(1), s->GetBinLowEdge( s->GetNbinsX()+1 ));
    newSyst->Sumw2();

    for(unsigned int i=0; i<bins.size(); i++) {
        float nom=n->Integral( (bins.at(i)).first, (bins.at(i)).second );
        float sys=s->Integral( (bins.at(i)).first, (bins.at(i)).second );
        // fill new systematic hist
        for(int b=(bins.at(i)).first; b<(bins.at(i)).second+1; b++) {
            if(nom>0) {
                newSyst->SetBinContent(b, (sys/nom) * n->GetBinContent(b));
            } else {
                newSyst->SetBinContent(b, nom);
            }
        }
    } // loop over equal area bins
    return newSyst;
} // EqualArea


TString MakeFname(TString cat, TString syst){
    TString fname = "bdt_"+syst+"BDTmmc_"+cat+".root";
    if (cat=="Boost_Merged") fname= "bdt_"+syst+"BDT_"+cat+".root";
    return fname;
}
