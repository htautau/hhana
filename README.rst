.. warning::

    UNDER CONSTRUCTION

Skimming
========

The skimming is performed by the ``hhskim.py`` script.

Run the skims on the grid (after setting up the panda client and your VOMS
proxy with the phys-higgs production role)::

    ./skim --yall mc11_hadhad mc12_hadhad data11_hadhad data12_hadhad embed11_hadhad embed12_hadhad


Datasets
========

After the skims are finished and downloaded, update the paths in
``higgstautau/datasets_config.yml`` and update the datasets database::

    ./dsdb --reset hh

Then launch the batch jobs that create all the analysis ntuples (nominal and
systematics) with::

    ./run_all_hh.sh

Automatically organize ROOT and log files with::

    init-ntup

The above also merges the output from the subjobs for embedding and data.

Add the Higgs pt weights::

    make higgs-pt

This creates copies of all signal ROOT files
``ntuples/running/hhskim/hhskim*tautauhh*.root`` at
``ntuples/running/hhskim/weighted.hhskim*tautauhh*.root``.

Make a backup of the original signal files::

    mkdir ntuples/running/hhskim/higgs_unweighted
    mv ntuples/running/hhskim/hhskim*tautauhh*.root ntuples/running/hhskim/higgs_unweighted

Then remove ``weighted.`` from the weighted signal files::

    rename weighted. "" ntuples/running/hhskim/weighted.hhskim*tautauhh*.root

Then move the hhskim running directory to production (replace XX with the skim
version number)::

    mkdir ntuples/prod_vXX
    mv ntuples/running/hhskim ntuples/prod_vXX

Update the production path in the ``Makefile`` (``HHNTUP``)
and ``mva/__init__.py`` (``NTUPLE_PATH``).

And finally create the merged ``hhskim.root`` and ``hhskim.h5``::

    make ntup


Background Normalizations
=========================

Generate the cache of all background normalizations::

    make norms


Variable Plots
==============

Create all validation plots with::

    make plots


BDT Training
============

Run the batch jobs that train the BDTs at each mass point with::

    make train

Go get some coffee.


BDT Validation Plots
--------------------

Create all the BDT validation plots with::

    make mva-plots


Workspaces
==========

Binning Optimization
--------------------

Run the batch jobs to determine the optimal binning for each mass point in each
category and year::

    make binning

Go get some coffee.


Creating Workpaces
------------------

Run the batch jobs that create the workspaces with::

    make mva-workspaces
    make cuts-workspaces

Go get some coffee.


Fixing Workspaces
-----------------

Apply all of the HSG4 workspace fixes with::

    cd workspaces
    fix-workspace --verbose --fill-empties hh_nos_nonisol_ebz_cuts hh_nos_nonisol_ebz_mva

Replace the path above with the actual path if different.

Go take a long walk.


Combining Workspaces
---------------------

Move to the directory containing all the MVA workspaces to combine::

    cd workspaces/hh_nos_nonisol_ebz_mva_fixed

Combine workspaces across years with::

    for mass in $(seq 100 5 150); do
        combine hh_11_vbf_$mass hh_12_vbf_$mass --name hh_vbf_$mass;
        combine hh_11_boosted_$mass hh_12_boosted_$mass --name hh_boosted_$mass;
        combine hh_11_combination_$mass hh_12_combination_$mass --name hh_combination_$mass;
    done

Check your email.

Move to the directory containing all the CBA workspaces to combine::

    cd workspaces/hh_nos_nonisol_ebz_cuts_fixed

Create VBF and boosted combinations for each year, and a combination
across years::

    for mass in $(seq 100 5 150); do
        combine hh_11_cuts_boosted_loose_$mass hh_11_cuts_boosted_tight_$mass --name hh_11_cuts_boosted_$mass;
        combine hh_12_cuts_boosted_loose_$mass hh_12_cuts_boosted_tight_$mass --name hh_12_cuts_boosted_$mass;
        combine hh_11_cuts_vbf_lowdr_$mass hh_11_cuts_vbf_highdr_$mass --name hh_11_cuts_vbf_$mass;
        combine hh_12_cuts_vbf_lowdr_$mass hh_12_cuts_vbf_highdr_loose_$mass hh_12_cuts_vbf_highdr_tight_$mass --name hh_12_cuts_vbf_$mass;
        combine hh_11_cuts_boosted_$mass hh_12_cuts_boosted_$mass --name hh_cuts_boosted_$mass;
        combine hh_11_cuts_vbf_$mass hh_12_cuts_vbf_$mass --name hh_cuts_vbf_$mass;
        combine hh_11_combination_$mass hh_12_combination_$mass --name hh_combination_$mass;
    done


Fitting
=======

Calculate the significance for each workspace with::

    multisig workspaces/hh_nos_nonisol_ebz_mva_fixed workspaces/hh_nos_nonisol_ebz_cuts_fixed


Creating p-value Plots
======================

References
==========

https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/NuisanceParameterPullsWithRanking
