/*
Author: Aaron Armbruster
Date:   2012-06-01
Email:  armbrusa@umich.edu
Description: 

Compute statistical significance with profile likelihood test stat. 
Option for uncapped test stat is added (doUncap), as well as an option
to choose which mu value to profile observed data at before generating expected


Modified by Noel Dawe
*/

#include <iostream>
#include <sstream>
#include <iomanip>

#include "TH1D.h"
#include "Math/MinimizerOptions.h"

#include "RooCategory.h"
#include "RooWorkspace.h"
#include "RooStats/ModelConfig.h"
#include "RooDataSet.h"
#include "RooMinimizerFcn.h"
#include "RooMinimizer.h"
#include "RooNLLVar.h"
#include "RooRealVar.h"
#include "RooSimultaneous.h"
#include "TSystem.h"


using namespace std;
using namespace RooFit;
using namespace RooStats;


RooDataSet* make_asimov_data(
        RooWorkspace* w, ModelConfig* mc,
        bool fluctuate_data = false,
        RooNLLVar* conditioning_nll = NULL,
        double mu_val = 1., double mu_val_profile = 1.,
        bool floating_mu_val_profile = false,
        string* mu_str = NULL, string* mu_prof_str = NULL,
        int print_level = 0);


int minimize(RooNLLVar* nll);


RooSimultaneous* reduce_pdf(RooSimultaneous* simPdf, vector<TString> v_CategoriesToReduce)
{

    RooCategory* channelCat = (RooCategory*) (&simPdf->indexCat());
    RooSimultaneous* simPdf_reduced= new RooSimultaneous("simPdf_reduced","reduced simultaneous pdf", *channelCat) ;
    TIterator *iter = channelCat->typeIterator() ;
    RooCatType *tt  = NULL;
    // Loop over all categories
    while((tt=(RooCatType*) iter->Next()) ){
        cout << endl;
        cout << endl;
        cout << " -- On category " << tt->GetName() << " " << endl;
        TString tmpName= tt->GetName();
        // If category is to be removed continue
        bool skipit=false;
        for(int iSkip=0;iSkip<v_CategoriesToReduce.size();iSkip++){
            if(  tmpName.Contains( (v_CategoriesToReduce[iSkip]) ) ){
                cout<<" Skipping"<<tt->GetName() << endl;
                skipit=true;
            }
        }
        if(skipit) continue; 

        ostringstream catname;
        catname << tt->GetName();
        cout<<"Adding "<<tmpName<<" to reduced RooSimultaneous"<<endl;
        RooAbsPdf  *pdftmp  = simPdf->getPdf( tt->GetName() );
        simPdf_reduced->addPdf(*pdftmp,tt->GetName() ) ;    
    }
    return simPdf_reduced;
}


TH1D* significance(RooWorkspace* ws,
        bool observed = false,            // compute observed significance
        double injection_mu = 1,          // mu injected in the asimov data
        bool injection_test = false,      // setup the poi for injection study (false is faster if you're not)
        bool profile = false,             // profile the observed data before generating the Asimov
        double profile_mu = 1,            // mu value to profile the observed data at before generating the Asimov
        bool floating_profile_mu = false, // if true then profile at mu hat
        const char* modelConfigName = "ModelConfig",
        const char* dataName = "obsData",
        bool verbose = false)
{
    string defaultMinimizer    = "Minuit2";     // or "Minuit"
    int defaultStrategy        = 1;             // Minimization strategy. 0-2. 0 = fastest, least robust. 2 = slowest, most robust
    bool doUncap               = 1;             // uncap p0

    if (!ws)
    {
        cout << "ERROR::Workspace is NULL!" << endl;
        return NULL;
    }
    ModelConfig* mc = (ModelConfig*)ws->obj(modelConfigName);
    if (!mc)
    {
        cout << "ERROR::ModelConfig: " << modelConfigName << " doesn't exist!" << endl;
        return NULL;
    }
    RooDataSet* data = (RooDataSet*)ws->data(dataName);
    if (!data)
    {
        cout << "ERROR::Dataset: " << dataName << " doesn't exist!" << endl;
        return NULL;
    }
    if (verbose)
        mc->GetNuisanceParameters()->Print("v");

    // save original state
    ws->saveSnapshot("significance::nominal_globs", *mc->GetGlobalObservables());
    ws->saveSnapshot("significance::nominal_nuis", *mc->GetNuisanceParameters());
    ws->saveSnapshot("significance::nominal_poi", *mc->GetParametersOfInterest());

    // minimizer options
    ROOT::Math::MinimizerOptions::SetDefaultMinimizer(defaultMinimizer.c_str());
    ROOT::Math::MinimizerOptions::SetDefaultStrategy(defaultStrategy);
    ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(1);
    //RooNLLVar::SetIgnoreZeroEntries(1);
    //ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(20000);
    //RooMinimizer::SetMaxFunctionCalls(10000);

    RooArgSet nuis(*mc->GetNuisanceParameters());
    RooRealVar* mu = (RooRealVar*)mc->GetParametersOfInterest()->first();
    RooAbsPdf* pdf = mc->GetPdf();

    if (!doUncap)
        mu->setRange(0, 40);
    else
        mu->setRange(-40, 40);

    RooArgSet nuis_tmp1 = *mc->GetNuisanceParameters();
    RooNLLVar* obs_nll = (observed || profile) ? (RooNLLVar*)pdf->createNLL(*data, Constrain(nuis_tmp1)) : NULL;

    int status, sign;
    double sig=0, q0=0;

    if (observed)
    {
        //ws->loadSnapshot("make_asimov_data::conditional_nuis_0"); ???
        mu->setVal(0);
        mu->setConstant(1);
        status = minimize(obs_nll);
        /*
           if (status < 0) 
           {
           cout << "Retrying with conditional snapshot at mu=1" << endl;
           ws->loadSnapshot("make_asimov_data::conditional_nuis_0");
           status = minimize(obs_nll);
           } ??? */
        double obs_nll_cond = obs_nll->getVal();

        //ws->loadSnapshot("ucmles"); ???
        mu->setConstant(0);
        status = minimize(obs_nll);
        /*
           if (status < 0) 
           {
           cout << "Retrying with conditional snapshot at mu=1" << endl;
           ws->loadSnapshot("make_asimov_data::conditional_nuis_0");
           status = minimize(obs_nll);
           } ??? */
        double obs_nll_min = obs_nll->getVal();

        q0 = 2*(obs_nll_cond - obs_nll_min);
        if (doUncap && mu->getVal() < 0)
            q0 = -q0;

        sign = int(q0 == 0 ? 0 : q0 / fabs(q0));
        if (!doUncap && ((q0 < 0 && q0 > -0.1) || mu->getVal() < 0.001))
            sig = 0; 
        else
            sig = sign*sqrt(fabs(q0));
    }
    else
    {
        string mu_str, mu_prof_str;
        RooDataSet* asimov_data = make_asimov_data(
                ws, mc, false, obs_nll,
                injection_mu, profile_mu, floating_profile_mu,
                &mu_str, &mu_prof_str);
        string condSnapshot = "make_asimov_data::conditional_globs" + mu_prof_str;

        RooArgSet nuis_tmp2 = *mc->GetNuisanceParameters();
        RooNLLVar* asimov_nll = (RooNLLVar*)pdf->createNLL(*asimov_data, Constrain(nuis_tmp2));

        mu->setVal(1);
        mu->setConstant(0);
        if (!injection_test) mu->setConstant(1);

        ws->loadSnapshot(condSnapshot.c_str());
        if (injection_test)
            ws->loadSnapshot("conditional_nuis_injection");
        else
            ws->loadSnapshot("make_asimov_data::conditional_nuis_1");
        if (verbose)
            mc->GetGlobalObservables()->Print("v");
        mu->setVal(0);
        mu->setConstant(1);
        status = minimize(asimov_nll);

        if (status < 0) 
        {
            cout << "Retrying with conditional snapshot at mu=1" << endl;
            ws->loadSnapshot("make_asimov_data::conditional_nuis_0");
            status = minimize(asimov_nll);
        }
        double asimov_nll_cond = asimov_nll->getVal();

        mu->setVal(1);
        if (injection_test)
            ws->loadSnapshot("conditional_nuis_injection");
        else
            ws->loadSnapshot("make_asimov_data::conditional_nuis_1");
        if (injection_test)
            mu->setConstant(0);
        status = minimize(asimov_nll);

        if (status < 0) 
        {
            cout << "Retrying with conditional snapshot at mu=1" << endl;
            ws->loadSnapshot("make_asimov_data::conditional_nuis_0");
            status = minimize(asimov_nll);
        }

        double asimov_nll_min = asimov_nll->getVal();
        q0 = 2*(asimov_nll_cond - asimov_nll_min);
        if (doUncap && mu->getVal() < 0)
            q0 = -q0;

        sign = int(q0 != 0 ? q0/fabs(q0) : 0);
        sig = sign*sqrt(fabs(q0));
    }

    TH1D* h_hypo = new TH1D("significance", "significance", 3, 0, 3);
    h_hypo->SetBinContent(1, sig);
    h_hypo->SetBinContent(2, mu->getVal());
    h_hypo->SetBinError(2, mu->getError());
    h_hypo->SetBinContent(3, q0);

    // restore original state
    ws->loadSnapshot("significance::nominal_globs");
    ws->loadSnapshot("significance::nominal_nuis");
    ws->loadSnapshot("significance::nominal_poi");

    return h_hypo;
}


int minimize(RooNLLVar* nll)
{
    int printLevel = ROOT::Math::MinimizerOptions::DefaultPrintLevel();
    RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
    if (printLevel < 0) RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

    int strat = ROOT::Math::MinimizerOptions::DefaultStrategy();
    RooMinimizer minim(*nll);
    minim.setStrategy(strat);
    minim.setPrintLevel(printLevel);

    int status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());

    if (status != 0 && status != 1 && strat < 2)
    {
        strat++;
        cout << "Fit failed with status " << status << ". Retrying with strategy " << strat << endl;
        minim.setStrategy(strat);
        status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
    }

    if (status != 0 && status != 1 && strat < 2)
    {
        strat++;
        cout << "Fit failed with status " << status << ". Retrying with strategy " << strat << endl;
        minim.setStrategy(strat);
        status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
    }

    if (status != 0 && status != 1)
    {
        cout << "Fit failed with status " << status << endl;
        string minType = ROOT::Math::MinimizerOptions::DefaultMinimizerType();
        string newMinType;
        if (minType == "Minuit2") newMinType = "Minuit";
        else newMinType = "Minuit2";

        cout << "Switching minuit type from " << minType << " to " << newMinType << endl;

        ROOT::Math::MinimizerOptions::SetDefaultMinimizer(newMinType.c_str());
        strat = 1; //ROOT::Math::MinimizerOptions::DefaultStrategy();
        minim.setStrategy(strat);

        status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());


        if (status != 0 && status != 1 && strat < 2)
        {
            strat++;
            cout << "Fit failed with status " << status << ". Retrying with strategy " << strat << endl;
            minim.setStrategy(strat);
            status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
        }

        if (status != 0 && status != 1 && strat < 2)
        {
            strat++;
            cout << "Fit failed with status " << status << ". Retrying with strategy " << strat << endl;
            minim.setStrategy(strat);
            status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
        }
        strat=2;
        ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minType.c_str());
    }

    if (printLevel < 0) RooMsgService::instance().setGlobalKillBelow(msglevel);

    return status;
}


void unfold_constraints(RooArgSet& initial, RooArgSet& final, RooArgSet& obs, RooArgSet& nuis, int& counter)
{
    if (counter > 50)
    {
        cout << "ERROR::Couldn't unfold constraints!" << endl;
        cout << "Initial: " << endl;
        initial.Print("v");
        cout << endl;
        cout << "Final: " << endl;
        final.Print("v");
        exit(1);
    }
    TIterator* itr = initial.createIterator();
    RooAbsPdf* pdf;
    while ((pdf = (RooAbsPdf*)itr->Next()))
    {
        RooArgSet nuis_tmp = nuis;
        RooArgSet constraint_set(*pdf->getAllConstraints(obs, nuis_tmp, false));
        string className(pdf->ClassName());
        if (className != "RooGaussian" && className != "RooLognormal" && className != "RooGamma" && className != "RooPoisson" && className != "RooBifurGauss")
        {
            counter++;
            unfold_constraints(constraint_set, final, obs, nuis, counter);
        }
        else
        {
            final.add(*pdf);
        }
    }
    delete itr;
}


RooDataSet* make_asimov_data(RooWorkspace* w, ModelConfig* mc,
        bool fluctuate_data,
        RooNLLVar* conditioning_nll, 
        double mu_val, double mu_val_profile,
        bool floating_mu_val_profile,
        string* mu_str, string* mu_prof_str,
        int print_level)
{
    ////////////////////
    //make asimov data//
    ////////////////////

    cout << "Creating asimov data at mu = " << mu_val << ", profiling at mu = " << mu_val_profile << endl;

    // save original state
    w->saveSnapshot("make_asimov_data::nominal_globs", *mc->GetGlobalObservables());
    w->saveSnapshot("make_asimov_data::nominal_nuis", *mc->GetNuisanceParameters());

    //ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
    //int strat = ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
    //int printLevel = ROOT::Math::MinimizerOptions::DefaultPrintLevel();
    //ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(-1);
    //RooMinuit::SetMaxIterations(10000);
    //RooMinimizer::SetMaxFunctionCalls(10000);

    RooAbsPdf* combPdf = mc->GetPdf();

    stringstream muStr;
    muStr << setprecision(5);
    muStr << "_" << mu_val;
    if (mu_str) *mu_str = muStr.str();

    stringstream muStrProf;
    muStrProf << setprecision(5);
    muStrProf << "_" << mu_val_profile;
    if (mu_prof_str) *mu_prof_str = muStrProf.str();

    RooRealVar* mu = (RooRealVar*)mc->GetParametersOfInterest()->first();
    mu->setVal(mu_val);

    RooArgSet mc_obs = *mc->GetObservables();
    RooArgSet mc_globs = *mc->GetGlobalObservables();
    RooArgSet mc_nuis = *mc->GetNuisanceParameters();

    //pair the nuisance parameter to the global observable
    RooArgSet mc_nuis_tmp = mc_nuis;
    RooArgList nui_list("ordered_nuis");
    RooArgList glob_list("ordered_globs");
    RooArgSet constraint_set_tmp(*combPdf->getAllConstraints(mc_obs, mc_nuis_tmp, false));
    RooArgSet constraint_set;
    int counter_tmp = 0;
    unfold_constraints(constraint_set_tmp, constraint_set, mc_obs, mc_nuis_tmp, counter_tmp);

    TIterator* cIter = constraint_set.createIterator();
    RooAbsArg* arg;

    /// Go through all constraints
    while ((arg = (RooAbsArg*)cIter->Next()))
    {
        RooAbsPdf* pdf = (RooAbsPdf*)arg;
        if (!pdf) continue;

        /// Catch the nuisance parameter constrained here
        TIterator* nIter = mc_nuis.createIterator();
        RooRealVar* thisNui = NULL;
        RooAbsArg* nui_arg;
        while ((nui_arg = (RooAbsArg*)nIter->Next()))
        {
            if (pdf->dependsOn(*nui_arg))
            {
                thisNui = (RooRealVar*)nui_arg;
                break;
            }
        }
        delete nIter;

        //need this incase the observable isn't fundamental. 
        //in this case, see which variable is dependent on the nuisance parameter and use that.
        RooArgSet* components = pdf->getComponents();
        components->remove(*pdf);
        if (components->getSize())
        {
            TIterator* itr1 = components->createIterator();
            RooAbsArg* arg1;
            while ((arg1 = (RooAbsArg*)itr1->Next()))
            {
                TIterator* itr2 = components->createIterator();
                RooAbsArg* arg2;
                while ((arg2 = (RooAbsArg*)itr2->Next()))
                {
                    if (arg1 == arg2) continue;
                    if (arg2->dependsOn(*arg1))
                    {
                        components->remove(*arg1);
                    }
                }
                delete itr2;
            }
            delete itr1;
        }
        if (components->getSize() > 1)
        {
            cout << "ERROR::Couldn't isolate proper nuisance parameter" << endl;
            return NULL;
        }
        else if (components->getSize() == 1)
        {
            thisNui = (RooRealVar*)components->first();
        }

        TIterator* gIter = mc_globs.createIterator();
        RooRealVar* thisGlob = NULL;
        RooAbsArg* glob_arg;
        while ((glob_arg = (RooAbsArg*)gIter->Next()))
        {
            if (pdf->dependsOn(*glob_arg))
            {
                thisGlob = (RooRealVar*)glob_arg;
                break;
            }
        }
        delete gIter;

        if (!thisNui || !thisGlob)
        {
            cout << "WARNING::Couldn't find nui or glob for constraint: " << pdf->GetName() << endl;
            continue;
        }

        if (print_level > 0) cout << "Pairing nui: " << thisNui->GetName() << ", with glob: " << thisGlob->GetName() << ", from constraint: " << pdf->GetName() << endl;

        nui_list.add(*thisNui);
        glob_list.add(*thisGlob);
    }
    delete cIter;

    RooArgSet nuiSet_tmp(nui_list);

    // conditional profiling
    if (conditioning_nll != NULL)
    {
        if (floating_mu_val_profile)
        {
            // profile at mu hat
            mu->setVal(0);
            mu->setConstant(0);
        }
        else
        {
            mu->setVal(mu_val_profile);
            mu->setConstant(1);
        }
        minimize(conditioning_nll);
    }
    mu->setConstant(0);
    mu->setVal(mu_val);

    //loop over the nui/glob list, grab the corresponding variable from the tmp ws, and set the glob to the value of the nui
    int nrNuis = nui_list.getSize();
    if (nrNuis != glob_list.getSize())
    {
        cout << "ERROR::nui_list.getSize() != glob_list.getSize()!" << endl;
        return NULL;
    }

    for (int i=0;i<nrNuis;i++)
    {
        RooRealVar* nui = (RooRealVar*)nui_list.at(i);
        RooRealVar* glob = (RooRealVar*)glob_list.at(i);
        //cout << "nui: " << nui << ", glob: " << glob << endl;
        //cout << "Setting glob: " << glob->GetName() << ", which had previous val: " << glob->getVal() << ", to conditional val: " << nui->getVal() << endl;
        glob->setVal(nui->getVal());
    }

    // save the snapshots of conditional parameters
    w->saveSnapshot(("make_asimov_data::conditional_globs"+muStrProf.str()).c_str(),*mc->GetGlobalObservables());
    w->saveSnapshot(("make_asimov_data::conditional_nuis" +muStrProf.str()).c_str(),*mc->GetNuisanceParameters());

    if (conditioning_nll == NULL)
    {
        // restore original state
        w->loadSnapshot("make_asimov_data::nominal_globs");
        w->loadSnapshot("make_asimov_data::nominal_nuis");
    }

    if (print_level > 0)
        cout << "Making asimov" << endl;

    // make the asimov data
    mu->setVal(mu_val);

    int iFrame=0;

    const char* weightName="weightVar";
    RooArgSet obsAndWeight;
    //cout << "adding obs" << endl;
    obsAndWeight.add(*mc->GetObservables());
    //cout << "adding weight" << endl;

    RooRealVar* weightVar = NULL;
    if (!(weightVar = w->var(weightName)))
    {
        w->import(*(new RooRealVar(weightName, weightName, 1,0,10000000)));
        weightVar = w->var(weightName);
    }
    //cout << "weightVar: " << weightVar << endl;
    obsAndWeight.add(*w->var(weightName));

    //cout << "defining set" << endl;
    w->defineSet("obsAndWeight",obsAndWeight);

    // MAKE ASIMOV DATA FOR OBSERVABLES
    RooSimultaneous* simPdf = dynamic_cast<RooSimultaneous*>(mc->GetPdf());

    RooDataSet* asimovData;
    if (!simPdf)
    {
        // Get pdf associated with state from simpdf
        RooAbsPdf* pdftmp = mc->GetPdf();

        // Generate observables defined by the pdf associated with this state
        RooArgSet* obstmp = pdftmp->getObservables(*mc->GetObservables()) ;

        if (print_level > 0)
        {
            obstmp->Print();
        }
        
        if (fluctuate_data)
        {
            // https://svnweb.cern.ch/trac/atlasinst/browser/Institutes/Freiburg/Higgs/StatTools/trunk/macros/RunToys.C
            cout<<"Creating extended datasample"<<endl;
            obsAndWeight.Print();
            asimovData = pdftmp->generate(RooArgSet(obsAndWeight),Extended(kTRUE));
            obsAndWeight.Print();
        }
        else
        {
            asimovData = new RooDataSet(
                    ("asimovData"+muStr.str()).c_str(),
                    ("asimovData"+muStr.str()).c_str(),
                    RooArgSet(obsAndWeight),
                    WeightVar(*weightVar));

            RooRealVar* thisObs = ((RooRealVar*)obstmp->first());
            double expectedEvents = pdftmp->expectedEvents(*obstmp);
            double thisNorm = 0;
            for(int jj=0; jj<thisObs->numBins(); ++jj){
                thisObs->setBin(jj);

                thisNorm=pdftmp->getVal(obstmp)*thisObs->getBinWidth(jj);
                if (thisNorm*expectedEvents <= 0)
                {
                    cout << "WARNING::Detected bin with zero expected events (" << thisNorm*expectedEvents
                        << ") ! Please check your inputs. Obs = " << thisObs->GetName()
                        << ", bin = " << jj << endl;
                }
                if (thisNorm*expectedEvents > 0 && thisNorm*expectedEvents < pow(10.0, 18))
                    asimovData->add(*mc->GetObservables(), thisNorm*expectedEvents);
            }
        }
        if (print_level > 0)
        {
            asimovData->Print();
            cout <<"sum entries "<<asimovData->sumEntries()<<endl;
        }
        if(asimovData->sumEntries()!=asimovData->sumEntries()){
            cout << "sum entries is nan"<<endl;
            return NULL;
        }
        if (print_level > 0)
        {
            asimovData->Print();
            cout << endl;
        }
        w->import(*asimovData);
    }
    else
    {
        map<string, RooDataSet*> asimovDataMap;

        //try fix for sim pdf
        RooCategory* channelCat = (RooCategory*)&simPdf->indexCat();//(RooCategory*)w->cat("master_channel");//(RooCategory*) (&simPdf->indexCat());
        //    TIterator* iter = simPdf->indexCat().typeIterator() ;
        TIterator* iter = channelCat->typeIterator() ;
        RooCatType* tt = NULL;
        int nrIndices = 0;
        while((tt=(RooCatType*) iter->Next())) {
            nrIndices++;
        }
        for (int i=0;i<nrIndices;i++){
            channelCat->setIndex(i);
            iFrame++;
            // Get pdf associated with state from simpdf
            RooAbsPdf* pdftmp = simPdf->getPdf(channelCat->getLabel()) ;

            // Generate observables defined by the pdf associated with this state
            RooArgSet* obstmp = pdftmp->getObservables(*mc->GetObservables()) ;

            if (print_level > 0)
            {
                obstmp->Print();
                cout << "on type " << channelCat->getLabel() << " " << iFrame << endl;
            }

            RooDataSet* obsDataUnbinned = new RooDataSet(
                    Form("combAsimovData%d",iFrame),
                    Form("combAsimovData%d",iFrame),
                    RooArgSet(obsAndWeight,*channelCat),
                    WeightVar(*weightVar));
            RooRealVar* thisObs = ((RooRealVar*)obstmp->first());
            double expectedEvents = pdftmp->expectedEvents(*obstmp);
            double thisNorm = 0;

            if (fluctuate_data)
            {
                cout<<"Creating extended datasample"<<endl;
                obsDataUnbinned = pdftmp->generate(RooArgSet(obsAndWeight),Extended(kTRUE));
            }
            else
            {
                for(int jj=0; jj<thisObs->numBins(); ++jj){
                    thisObs->setBin(jj);

                    thisNorm=pdftmp->getVal(obstmp)*thisObs->getBinWidth(jj);
                    if (thisNorm*expectedEvents > 0 && thisNorm*expectedEvents < pow(10.0, 18))
                        obsDataUnbinned->add(*mc->GetObservables(), thisNorm*expectedEvents);
                }
            }
            if (print_level > 0)
            {
                obsDataUnbinned->Print();
                cout <<"sum entries "<<obsDataUnbinned->sumEntries()<<endl;
            }
            if(obsDataUnbinned->sumEntries()!=obsDataUnbinned->sumEntries()){
                cout << "sum entries is nan"<<endl;
                return NULL;
            }

            asimovDataMap[string(channelCat->getLabel())] = obsDataUnbinned;

            if (print_level > 0)
            {
                cout << "channel: " << channelCat->getLabel() << ", data: ";
                obsDataUnbinned->Print();
                cout << endl;
            }
        }

        asimovData = new RooDataSet(
                ("asimovData"+muStr.str()).c_str(),
                ("asimovData"+muStr.str()).c_str(),
                RooArgSet(obsAndWeight,*channelCat),
                Index(*channelCat),
                Import(asimovDataMap),
                WeightVar(*weightVar));
    }

    // restore original state
    w->loadSnapshot("make_asimov_data::nominal_globs");
    //w->loadSnapshot("make_asimov_data::nominal_nuis");

    return asimovData;
}
