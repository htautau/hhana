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
#include "TStopwatch.h"
#include "TFile.h"
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


RooDataSet* makeAsimovData(
        ModelConfig* mc, bool doConditional,
        RooWorkspace* w, RooNLLVar* conditioning_nll,
        double mu_val, string* mu_str, string* mu_prof_str,
        double mu_val_profile, bool doFit);

int minimize(RooNLLVar* nll, RooWorkspace* combWS = NULL);

//int minimize(RooNLLVar* nll);
//int minimize(RooAbsReal* nll);

TH1D* runSig(RooWorkspace* ws,
             const char* modelConfigName = "ModelConfig",
             const char* dataName = "obsData",
             const char* asimov1DataName = "asimovData_1",
             const char* conditional1Snapshot = "conditionalGlobs_1",
             const char* nominalSnapshot = "nominalGlobs")
{
    string defaultMinimizer    = "Minuit";     // or "Minuit"
    int defaultStrategy        = 2;             // Minimization strategy. 0-2. 0 = fastest, least robust. 2 = slowest, most robust

    double mu_profile_value = 1; // mu value to profile the obs data at wbefore generating the expected
    bool doUncap            = 1; // uncap p0
    bool doInj              = 0; // setup the poi for injection study (zero is faster if you're not)
    bool doMedian           = 1; // compute median significance
    bool isBlind            = 1; // Dont look at observed data
    bool doConditional      = !isBlind; // do conditional expected data
    bool doObs              = !isBlind; // compute observed significance

    TStopwatch timer;
    timer.Start();

    if (!ws)
    {
        cout << "ERROR::Workspace is NULL!" << endl;
        return;
    }
    ModelConfig* mc = (ModelConfig*)ws->obj(modelConfigName);
    if (!mc)
    {
        cout << "ERROR::ModelConfig: " << modelConfigName << " doesn't exist!" << endl;
        return;
    }
    RooDataSet* data = (RooDataSet*)ws->data(dataName);
    if (!data)
    {
        cout << "ERROR::Dataset: " << dataName << " doesn't exist!" << endl;
        return;
    }

    mc->GetNuisanceParameters()->Print("v");

    //RooNLLVar::SetIgnoreZeroEntries(1);
    ROOT::Math::MinimizerOptions::SetDefaultMinimizer(defaultMinimizer.c_str());
    ROOT::Math::MinimizerOptions::SetDefaultStrategy(defaultStrategy);
    ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(-1);
    //  cout << "Setting max function calls" << endl;
    //ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(20000);
    //RooMinimizer::SetMaxFunctionCalls(10000);

    ws->loadSnapshot("conditionalNuis_0");
    RooArgSet nuis(*mc->GetNuisanceParameters());

    RooRealVar* mu = (RooRealVar*)mc->GetParametersOfInterest()->first();

    RooAbsPdf* pdf_temp = mc->GetPdf();

    string condSnapshot(conditional1Snapshot);
    RooArgSet nuis_tmp2 = *mc->GetNuisanceParameters();
    RooNLLVar* obs_nll = doObs ? (RooNLLVar*)pdf_temp->createNLL(*data, Constrain(nuis_tmp2)) : NULL;

    RooDataSet* asimovData1 = (RooDataSet*)ws->data(asimov1DataName);
    RooRealVar* emb = (RooRealVar*)mc->GetNuisanceParameters()->find("ATLAS_EMB");
    if (!asimovData1 || (string(inFileName).find("ic10") != string::npos && emb))
    {
        if (emb) emb->setVal(0.7);
        cout << "Asimov data doesn't exist! Please, allow me to build one for you..." << endl;
        string mu_str, mu_prof_str;

        cout << __LINE__ << endl;

        asimovData1 = makeAsimovData(mc, doConditional, ws, obs_nll, 1, &mu_str, &mu_prof_str, mu_profile_value, true);
        condSnapshot="conditionalGlobs"+mu_prof_str;

        //makeAsimovData(mc, true, ws, mc->GetPdf(), data, 0);
        //ws->Print();
        //asimovData1 = (RooDataSet*)ws->data("asimovData_1");
    }

    if (!doUncap) mu->setRange(0, 40);
    else mu->setRange(-40, 40);

    RooAbsPdf* pdf = mc->GetPdf();
    RooArgSet nuis_tmp1 = *mc->GetNuisanceParameters();
    RooNLLVar* asimov_nll = (RooNLLVar*)pdf->createNLL(*asimovData1, Constrain(nuis_tmp1));

    //do asimov
    mu->setVal(1);
    mu->setConstant(0);
    if (!doInj) mu->setConstant(1);

    int status,sign;
    double med_sig=0,obs_sig=0,asimov_q0=0,obs_q0=0;

    if (doMedian)
    {
        ws->loadSnapshot(condSnapshot.c_str());
        if (doInj) ws->loadSnapshot("conditionalNuis_inj");
        else ws->loadSnapshot("conditionalNuis_1");
        mc->GetGlobalObservables()->Print("v");
        mu->setVal(0);
        mu->setConstant(1);
        status = minimize(asimov_nll, ws);
        if (status >= 0) cout << "Success!" << endl;

        if (status < 0) 
        {
            cout << "Retrying with conditional snapshot at mu=1" << endl;
            ws->loadSnapshot("conditionalNuis_0");
            status = minimize(asimov_nll, ws);
            if (status >= 0) cout << "Success!" << endl;
        }
        double asimov_nll_cond = asimov_nll->getVal();

        mu->setVal(1);
        if (doInj) ws->loadSnapshot("conditionalNuis_inj");
        else ws->loadSnapshot("conditionalNuis_1");
        if (doInj) mu->setConstant(0);
        status = minimize(asimov_nll, ws);
        if (status >= 0) cout << "Success!" << endl;

        if (status < 0) 
        {
            cout << "Retrying with conditional snapshot at mu=1" << endl;
            ws->loadSnapshot("conditionalNuis_0");
            status = minimize(asimov_nll, ws);
            if (status >= 0) cout << "Success!" << endl;
        }

        double asimov_nll_min = asimov_nll->getVal();
        asimov_q0 = 2*(asimov_nll_cond - asimov_nll_min);
        if (doUncap && mu->getVal() < 0) asimov_q0 = -asimov_q0;

        sign = int(asimov_q0 != 0 ? asimov_q0/fabs(asimov_q0) : 0);
        med_sig = sign*sqrt(fabs(asimov_q0));

        ws->loadSnapshot(nominalSnapshot);
    }


    if (doObs)
    {
        ws->loadSnapshot("conditionalNuis_0");
        mu->setVal(0);
        mu->setConstant(1);
        status = minimize(obs_nll, ws);
        if (status < 0) 
        {
            cout << "Retrying with conditional snapshot at mu=1" << endl;
            ws->loadSnapshot("conditionalNuis_0");
            status = minimize(obs_nll, ws);
            if (status >= 0) cout << "Success!" << endl;
        }
        double obs_nll_cond = obs_nll->getVal();

        //ws->loadSnapshot("ucmles");
        mu->setConstant(0);
        status = minimize(obs_nll, ws);
        if (status < 0) 
        {
            cout << "Retrying with conditional snapshot at mu=1" << endl;
            ws->loadSnapshot("conditionalNuis_0");
            status = minimize(obs_nll, ws);
            if (status >= 0) cout << "Success!" << endl;
        }

        double obs_nll_min = obs_nll->getVal();

        obs_q0 = 2*(obs_nll_cond - obs_nll_min);
        if (doUncap && mu->getVal() < 0) obs_q0 = -obs_q0;

        sign = int(obs_q0 == 0 ? 0 : obs_q0 / fabs(obs_q0));
        if (!doUncap && (obs_q0 < 0 && obs_q0 > -0.1 || mu->getVal() < 0.001)) obs_sig = 0; 
        else obs_sig = sign*sqrt(fabs(obs_q0));
    }

    cout << "obs: " << obs_sig << endl;

    cout << "Observed significance: " << obs_sig << endl;
    if (med_sig)
    {
        cout << "Median test stat val: " << asimov_q0 << endl;
        cout << "Median significance:   " << med_sig << endl;
    }

    TH1D* h_hypo = new TH1D("hypo","hypo",2,0,2);
    h_hypo->SetBinContent(1, obs_sig);
    h_hypo->SetBinContent(2, med_sig);

    timer.Stop();
    timer.Print();
    return h_hypo;
}


int minimize(RooNLLVar* nll, RooWorkspace* combWS)
{
    bool const_test = 0;

    vector<string> const_vars;
    //  const_vars.push_back("alpha_ATLAS_JES_NoWC_llqq");
    //   const_vars.push_back("alpha_ATLAS_ZBB_PTW_NoWC_llqq");
    //   const_vars.push_back("alpha_ATLAS_ZCR_llqqNoWC_llqq");

    int nrConst = const_vars.size();

    if (const_test)
    {
        for (int i=0;i<nrConst;i++)
        {
            RooRealVar* const_var = combWS->var(const_vars[i].c_str());
            const_var->setConstant(1);
        }
    }

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


    if (status == 0)
        cout<<"Successful fit! "<<endl;
    cout<<"Fit exists with status:"<<status<<endl;

    //   if (status != 0 && status != 1)
    //   {
    //     cout << "Fit failed for mu = " << mu->getVal() << " with status " << status << ". Retrying with pdf->fitTo()" << endl;
    //     combPdf->fitTo(*combData,Hesse(false),Minos(false),PrintLevel(0),Extended(), Constrain(nuiSet_tmp));
    //   }
    if (printLevel < 0) RooMsgService::instance().setGlobalKillBelow(msglevel);


    if (const_test)
    {
        for (int i=0;i<nrConst;i++)
        {
            RooRealVar* const_var = combWS->var(const_vars[i].c_str());
            const_var->setConstant(0);
        }
    }


    return status;
}

void unfoldConstraints(RooArgSet& initial, RooArgSet& final, RooArgSet& obs, RooArgSet& nuis, int& counter)
{
    //he is digging deep into the terms and finds gaussians,lognormals,gammas,poissons and bifurgaussians
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
        //if (constraint_set.getSize() > 1)
        //{
        string className(pdf->ClassName());
        if (className != "RooGaussian" && className != "RooLognormal" && className != "RooGamma" && className != "RooPoisson" && className != "RooBifurGauss")
        {
            counter++;
            unfoldConstraints(constraint_set, final, obs, nuis, counter);
        }
        else
        {
            final.add(*pdf);
        }
    }
    delete itr;
}

RooDataSet* makeAsimovData(ModelConfig* mc, bool doConditional, RooWorkspace* w, RooNLLVar* conditioning_nll, double mu_val, string* mu_str, string* mu_prof_str, double mu_val_profile, bool doFit)
{

    cout << __LINE__ << endl;

    if (mu_val_profile == -999) mu_val_profile = mu_val;


    cout << "Creating asimov data at mu = " << mu_val << ", profiling at mu = " << mu_val_profile << endl;

    //ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
    //int strat = ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
    //int printLevel = ROOT::Math::MinimizerOptions::DefaultPrintLevel();
    //ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(-1);
    //RooMinuit::SetMaxIterations(10000);
    //RooMinimizer::SetMaxFunctionCalls(10000);

    ////////////////////
    //make asimov data//
    ////////////////////

    RooAbsPdf* combPdf = mc->GetPdf();

    int _printLevel = 0;

    stringstream muStr;
    muStr << setprecision(5);
    muStr << "_" << mu_val;
    if (mu_str) *mu_str = muStr.str();

    stringstream muStrProf;
    muStrProf << setprecision(5);
    muStrProf << "_" << mu_val_profile;
    if (mu_prof_str) *mu_prof_str = muStrProf.str();

    RooRealVar* mu = (RooRealVar*)mc->GetParametersOfInterest()->first();//w->var("mu");
    mu->setVal(mu_val);

    RooArgSet mc_obs = *mc->GetObservables();
    RooArgSet mc_globs = *mc->GetGlobalObservables();
    RooArgSet mc_nuis = *mc->GetNuisanceParameters();

    cout << __LINE__ << endl;

    //pair the nuisance parameter to the global observable
    RooArgSet mc_nuis_tmp = mc_nuis;
    RooArgList nui_list("ordered_nuis");
    RooArgList glob_list("ordered_globs");
    RooArgSet constraint_set_tmp(*combPdf->getAllConstraints(mc_obs, mc_nuis_tmp, false));
    RooArgSet constraint_set;
    int counter_tmp = 0;
    unfoldConstraints(constraint_set_tmp, constraint_set, mc_obs, mc_nuis_tmp, counter_tmp);

    TIterator* cIter = constraint_set.createIterator();
    RooAbsArg* arg;

    /// Go through all constraints
    while ((arg = (RooAbsArg*)cIter->Next()))
    {
        RooAbsPdf* pdf = (RooAbsPdf*)arg;
        if (!pdf) continue;
        //     cout << "Printing pdf" << endl;
        //     pdf->Print();
        //     cout << "Done" << endl;

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

        cout << __LINE__ << endl;

        //RooRealVar* thisNui = (RooRealVar*)pdf->getObservables();


        //need this incase the observable isn't fundamental. 
        //in this case, see which variable is dependent on the nuisance parameter and use that.
        RooArgSet* components = pdf->getComponents();
        //     cout << "\nPrinting components" << endl;
        //     components->Print();
        //     cout << "Done" << endl;
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

        cout << __LINE__ << endl;

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
            //return;
            continue;
        }


        cout << __LINE__ << endl;

        if (_printLevel >= 1) cout << "Pairing nui: " << thisNui->GetName() << ", with glob: " << thisGlob->GetName() << ", from constraint: " << pdf->GetName() << endl;

        nui_list.add(*thisNui);
        glob_list.add(*thisGlob);

        //     cout << "\nPrinting Nui/glob" << endl;
        //     thisNui->Print();
        //     cout << "Done nui" << endl;
        //     thisGlob->Print();
        //     cout << "Done glob" << endl;
    }
    delete cIter;


    cout << __LINE__ << endl;

    //save the snapshots of nominal parameters, but only if they're not already saved
    w->saveSnapshot("tmpGlobs",*mc->GetGlobalObservables());
    cout << __LINE__ << endl;
    w->saveSnapshot("tmpNuis",*mc->GetNuisanceParameters());
    if (!w->loadSnapshot("nominalGlobs"))
    {
        cout << "nominalGlobs doesn't exist. Saving snapshot." << endl;
        w->saveSnapshot("nominalGlobs",*mc->GetGlobalObservables());
    }
    else w->loadSnapshot("tmpGlobs");
    if (!w->loadSnapshot("nominalNuis"))
    {
        cout << "nominalNuis doesn't exist. Saving snapshot." << endl;
        w->saveSnapshot("nominalNuis",*mc->GetNuisanceParameters());
    }
    else w->loadSnapshot("tmpNuis");

    cout << __LINE__ << endl;

    RooArgSet nuiSet_tmp(nui_list);
    cout << __LINE__ << endl;

    mu->setVal(mu_val_profile);
    mu->setConstant(1);
    //int status = 0;
    cout << __LINE__ << endl;

    if (doConditional && doFit)
    {
        cout << __LINE__ << endl;

        minimize(conditioning_nll);
        cout << __LINE__ << endl;

        // cout << "Using globs for minimization" << endl;
        // mc->GetGlobalObservables()->Print("v");
        // cout << "Starting minimization.." << endl;
        // RooAbsReal* nll;
        // if (!(nll = map_data_nll[combData])) nll = combPdf->createNLL(*combData, RooFit::Constrain(nuiSet_tmp));
        // RooMinimizer minim(*nll);
        // minim.setStrategy(0);
        // minim.setPrintLevel(1);
        // status = minim.minimize(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
        // if (status != 0)
        // {
        //   cout << "Fit failed for mu = " << mu->getVal() << " with status " << status << endl;
        // }
        // cout << "Done" << endl;

        //combPdf->fitTo(*combData,Hesse(false),Minos(false),PrintLevel(0),Extended(), Constrain(nuiSet_tmp));
    }
    mu->setConstant(0);
    mu->setVal(mu_val);

    cout << __LINE__ << endl;

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

    cout << __LINE__ << endl;

    //save the snapshots of conditional parameters
    cout << "Saving conditional snapshots" << endl;
    cout << "Glob snapshot name = " << "conditionalGlobs"+muStrProf.str() << endl;
    cout << "Nuis snapshot name = " << "conditionalNuis"+muStrProf.str() << endl;
    w->saveSnapshot(("conditionalGlobs"+muStrProf.str()).c_str(),*mc->GetGlobalObservables());
    w->saveSnapshot(("conditionalNuis" +muStrProf.str()).c_str(),*mc->GetNuisanceParameters());

    if (!doConditional)
    {
        w->loadSnapshot("nominalGlobs");
        w->loadSnapshot("nominalNuis");
    }

    if (_printLevel >= 1) cout << "Making asimov" << endl;
    //make the asimov data (snipped from Kyle)
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


    cout << __LINE__ << endl;

    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    // MAKE ASIMOV DATA FOR OBSERVABLES

    // dummy var can just have one bin since it's a dummy
    //if(w->var("ATLAS_dummyX"))  w->var("ATLAS_dummyX")->setBins(1);

    //cout <<" check expectedData by category"<<endl;
    //RooDataSet* simData=NULL;
    RooSimultaneous* simPdf = dynamic_cast<RooSimultaneous*>(mc->GetPdf());

    RooDataSet* asimovData;
    if (!simPdf)
    {
        // Get pdf associated with state from simpdf
        RooAbsPdf* pdftmp = mc->GetPdf();//simPdf->getPdf(channelCat->getLabel()) ;

        // Generate observables defined by the pdf associated with this state
        RooArgSet* obstmp = pdftmp->getObservables(*mc->GetObservables()) ;

        if (_printLevel >= 1)
        {
            obstmp->Print();
        }

        cout << __LINE__ << endl;

        asimovData = new RooDataSet(("asimovData"+muStr.str()).c_str(),("asimovData"+muStr.str()).c_str(),RooArgSet(obsAndWeight),WeightVar(*weightVar));

        cout << __LINE__ << endl;

        RooRealVar* thisObs = ((RooRealVar*)obstmp->first());
        double expectedEvents = pdftmp->expectedEvents(*obstmp);
        double thisNorm = 0;
        for(int jj=0; jj<thisObs->numBins(); ++jj){
            thisObs->setBin(jj);

            thisNorm=pdftmp->getVal(obstmp)*thisObs->getBinWidth(jj);
            if (thisNorm*expectedEvents <= 0)
            {
                cout << "WARNING::Detected bin with zero expected events (" << thisNorm*expectedEvents << ") ! Please check your inputs. Obs = " << thisObs->GetName() << ", bin = " << jj << endl;
            }
            if (thisNorm*expectedEvents > 0 && thisNorm*expectedEvents < pow(10.0, 18)) asimovData->add(*mc->GetObservables(), thisNorm*expectedEvents);
        }

        if (_printLevel >= 1)
        {
            asimovData->Print();
            cout <<"sum entries "<<asimovData->sumEntries()<<endl;
        }
        if(asimovData->sumEntries()!=asimovData->sumEntries()){
            cout << "sum entries is nan"<<endl;
            exit(1);
        }

        //((RooRealVar*)obstmp->first())->Print();
        //cout << "expected events " << pdftmp->expectedEvents(*obstmp) << endl;

        w->import(*asimovData);

        if (_printLevel >= 1)
        {
            asimovData->Print();
            cout << endl;
        }
    }
    else
    {
        map<string, RooDataSet*> asimovDataMap;

        cout << __LINE__ << endl;

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

            if (_printLevel >= 1)
            {
                obstmp->Print();
                cout << "on type " << channelCat->getLabel() << " " << iFrame << endl;
            }

            RooDataSet* obsDataUnbinned = new RooDataSet(Form("combAsimovData%d",iFrame),Form("combAsimovData%d",iFrame),RooArgSet(obsAndWeight,*channelCat),WeightVar(*weightVar));
            RooRealVar* thisObs = ((RooRealVar*)obstmp->first());
            double expectedEvents = pdftmp->expectedEvents(*obstmp);
            double thisNorm = 0;
            for(int jj=0; jj<thisObs->numBins(); ++jj){
                thisObs->setBin(jj);

                thisNorm=pdftmp->getVal(obstmp)*thisObs->getBinWidth(jj);
                if (thisNorm*expectedEvents > 0 && thisNorm*expectedEvents < pow(10.0, 18)) obsDataUnbinned->add(*mc->GetObservables(), thisNorm*expectedEvents);
            }

            if (_printLevel >= 1)
            {
                obsDataUnbinned->Print();
                cout <<"sum entries "<<obsDataUnbinned->sumEntries()<<endl;
            }
            if(obsDataUnbinned->sumEntries()!=obsDataUnbinned->sumEntries()){
                cout << "sum entries is nan"<<endl;
                exit(1);
            }

            // ((RooRealVar*)obstmp->first())->Print();
            // cout << "pdf: " << pdftmp->GetName() << endl;
            // cout << "expected events " << pdftmp->expectedEvents(*obstmp) << endl;
            // cout << "-----" << endl;

            asimovDataMap[string(channelCat->getLabel())] = obsDataUnbinned;//tempData;

            if (_printLevel >= 1)
            {
                cout << "channel: " << channelCat->getLabel() << ", data: ";
                obsDataUnbinned->Print();
                cout << endl;
            }
        }

        cout << __LINE__ << endl;

        asimovData = new RooDataSet(("asimovData"+muStr.str()).c_str(),("asimovData"+muStr.str()).c_str(),RooArgSet(obsAndWeight,*channelCat),Index(*channelCat),Import(asimovDataMap),WeightVar(*weightVar));
        w->import(*asimovData);
    }

    cout << __LINE__ << endl;

    //bring us back to nominal for exporting
    //w->loadSnapshot("nominalNuis");
    w->loadSnapshot("nominalGlobs");

    //ROOT::Math::MinimizerOptions::SetDefaultPrintLevel(printLevel);

    return asimovData;
}
