// c++
#include <iostream>
#include <vector>

// root
#include "TH1.h"
#include <TMath.h>


namespace Smooth {

// ===== Function definition ====================
TH1* EqualArea(TH1* n, TH1* s, float frac=0.5);
TH1* EqualAreaGabriel(TH1* n, TH1* s);
// ==============================================


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

}
