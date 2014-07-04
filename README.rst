.. warning::

    UNDER CONSTRUCTION

Dependencies
============

* `NumPy <http://www.numpy.org/>`_::

   pip install --user numpy

* `HDF5 <http://www.hdfgroup.org/HDF5/>`_::

   yum install hdf5 hdf5-devel

* `PyTables <http://www.pytables.org/moin>`_::

   pip install --user tables

* `scikit-learn <http://scikit-learn.org/stable/>`_::

   git clone git://github.com/ndawe/scikit-learn.git
   cd scikit-learn
   git checkout -b htt origin/htt
   python setup.py install --user

* `rootpy <https://github.com/rootpy/rootpy>`_::

   git clone git://github.com/rootpy/rootpy.git
   cd rootpy
   python setup.py install --user

* `root_numpy <https://pypi.python.org/pypi/root_numpy>`_::

   pip install --user root_numpy

* `yellowhiggs <https://pypi.python.org/pypi/yellowhiggs/>`_::

   pip install --user yellowhiggs

* `GitPython <https://github.com/gitpython-developers/GitPython>`_::

   pip install --user GitPython

* `tabulate <https://pypi.python.org/pypi/tabulate>`_::

   pip install --user tabulate


Data Preparation
================

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


Combining Workspaces
---------------------

Move to the directory containing all the MVA workspaces to combine::

    cd workspaces/hh_nos_nonisol_ebz_mva

Combine workspaces across years with::

    for mass in $(seq 100 5 150); do
        combine hh_11_vbf_$mass hh_12_vbf_$mass --name hh_vbf_$mass;
        combine hh_11_boosted_$mass hh_12_boosted_$mass --name hh_boosted_$mass;
        combine hh_11_combination_$mass hh_12_combination_$mass --name hh_combination_$mass;
    done

Check your email.

Move to the directory containing all the CBA workspaces to combine::

    cd workspaces/hh_nos_nonisol_ebz_cuts

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


Fixing Workspaces
-----------------

Apply all of the HSG4 workspace fixes with::

    cd workspaces
    fix-workspace --quiet --symmetrize --prune-shapes --chi2-thresh 0.9 hh_nos_nonisol_ebz_mva
    fix-workspace --quiet --symmetrize --prune-shapes --chi2-thresh 0.9 hh_nos_nonisol_ebz_cuts

Scan of the nuisance parameters
-------------------------------

Construct the profile of every nuisance parameter  (NP)::

    # submit a batch job for each NP. If --submit is omitted simply print the command.
    multinp scans_fit --file path_to_measurement_file.root --submit
    # merge all the output in a single file and compute the nominal NLL for normalisation
    multinp merge --file path_to_measurement_file.root --jobs -1
    # Clean the directory from the individual pickle files (keep only the master)
    multinp clean --file path_to_measurement_file.root

Update the paths in plot-nuis and plot the profiles with::

    plot-nuis


Pulls of the nuisance parameters
--------------------------------

Compute the pull of each nuisance parameter with::

    multinp pulls --file path_to_measurement_file.root --jobs -1

Update the path in plot-ranking and plot the ranking with::

   plot-ranking

Significance
------------

Compute the expected significance (bkg. only hypothesis) with::

    # Walk trough the directory and subdirectory and look for workspaces
    multisig path_to_directory_containing_workspaces

Postfit plot
------------

Compute the postfit histograms and errors with::

    # --fit_var bdt_score/mmc_mass
    plot-postfit path_to_measurement_file.root --fit-var bdt_score --force-fit --jobs -1
    # If the fit has already been performed
    plot-postfit path_to_measurement_file.root --fit-var bdt_score
