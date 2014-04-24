.. warning::

    UNDER CONSTRUCTION

Datasets
========

After producing the analysis ntuples in the higgstautau package they are
prepared and merged here in the htt package.

Automatically organize ROOT and log files with::

    init-ntup

The above also merges the output from the subjobs for embedding and data.

Then move the running hhskim directory to production::

    mv ntuples/prod/hhskim ntuples/prod/hhskim_old
    mv ntuples/running/hhskim ntuples/prod/hhskim


Background Normalizations
=========================


Variable Plots
==============

BDT Training
============

Run the batch jobs that train the BDTs at each mass point with::

    make train

Go get some coffee.


BDT Validation Plots
====================


Binning Optimization
====================

Run the batch jobs to determine the optimal binning for each mass point in each
category and year::

    make binning

Go get some coffee.


Workspaces
==========

Creating Workpaces
------------------

Run the batch jobs that create the workspaces with::

    make mva-workspaces
    make cuts-workspaces

Go get some coffee.


Fixing Workspaces
-----------------

Apply all of the HSG4 workspace fixes with::

    fix-workspace workspaces/hh_nos_nonisol_ebz

This will create the directory containing ``workspaces/hh_nos_nonisol_ebz_fixed``
with all of the patched workspaces.

Replace the path above with the actual path if different.

Check your email.


Combining Workspaces
---------------------

Move to the directory containing all the workspaces to combine::

    cd workspaces/hh_nos_nonisol_ebz_fixed

Combine workspaces across years with::

   for mass in $(seq 100 5 150); do
      combine hh_11_vbf_$mass hh_12_vbf_$mass --output hh_vbf_$mass --name hh_vbf_$mass;
      combine hh_11_boosted_$mass hh_12_boosted_$mass --output hh_boosted_$mass --name hh_boosted_$mass;
      combine hh_11_combination_$mass hh_12_combination_$mass --output hh_combination_$mass --name hh_combination_$mass;
   done

Check your email.


Fitting
=======

Calculate the significance for each workspace with::

    multisig workspaces/hh_nos_nonisol_ebz_fixed/


Creating p-value Plots
======================

References
==========

https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/NuisanceParameterPullsWithRanking
