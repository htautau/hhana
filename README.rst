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


BDT Validation Plots
====================


Binning Optimization
====================

Run the batch jobs to determine the optimal binning for each mass point in each
category and year::

    make binning


Workspaces
==========

Creating Workpaces
------------------

Run the batch jobs that create the BDT workspaces with::

    make bdt-workspaces


Fixing Workspaces
-----------------


Combining Workspaces
---------------------

Combine workspaces with::

   for mass in $(seq 100 5 150); do
      combine hh_11_vbf_$mass hh_12_vbf_$mass --output hh_vbf_$mass --name hh_vbf_$mass;
   done

Fitting
=======
