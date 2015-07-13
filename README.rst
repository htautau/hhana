.. warning::

    UNDER CONSTRUCTION

`Dependencies <dependencies.rst>`__
===================================

* `dependencies <dependencies.rst>`__

* Latest 5.X `ROOT <http://root.cern.ch/drupal/>`_ with PyROOT, RooFit,
  Minuit2, HistFactory enabled.

* `NumPy <http://www.numpy.org/>`_::

   pip install --user numpy

* `HDF5 <http://www.hdfgroup.org/HDF5/>`_::

   yum install hdf5 hdf5-devel

* `PyTables <http://www.pytables.org/moin>`_::

   pip install --user tables

* `scikit-learn <http://scikit-learn.org/stable/>`_ (private branch with more
  support for event weights)::

   git clone git://github.com/ndawe/scikit-learn.git
   cd scikit-learn
   git checkout -b htt origin/htt
   python setup.py install --user

* `matplotlib <http://matplotlib.org/>`_::

   pip install --user matplotlib

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

* `hhntup <https://github.com/htautau/hhntup>`_::

   git clone git://github.com/htautau/hhntup.git

* `hhana <https://github.com/htautau/hhana>`_::

   git clone git://github.com/htautau/hhana.git


Data Preparation (DEPRECATED)
=============================

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

When the batch jobs are done, create the workspace combinations with::

    make combine-mva
    make combine-cuts


Fixing Workspaces
-----------------

Apply all of the HSG4 workspace fixes with::

    cd workspaces
    workspace-fix --quiet --symmetrize --prune-shapes --chi2-thresh 0.9 hh_nos_nonisol_ebz_mva
    workspace-fix --quiet --symmetrize --prune-shapes --chi2-thresh 0.9 hh_nos_nonisol_ebz_cuts

Scan of the nuisance parameters
-------------------------------

Construct the profile of every nuisance parameter (NP)::

    # submit a batch job for each NP. If --submit is omitted simply print the command.
    workspace-multinp scans_fit --submit --file path_to_measurement_file.root
    # merge all the output in a single file and compute the nominal NLL for normalisation
    workspace-multinp merge --jobs -1 --file path_to_measurement_file.root
    # Clean the directory from the individual pickle files (keep only the master)
    workspace-multinp clean --file path_to_measurement_file.root

Plot the NP profiles with::

    plot-nuis path_to_measurement_file.root


Pulls of the nuisance parameters
--------------------------------

Compute the pull of each nuisance parameter with::

    multinp pulls --jobs -1 --file path_to_measurement_file.root

Plot the NP ranking/pulls with::

    plot-ranking path_to_measurement_file.root

Significance
------------

Compute the expected significance (bkg. only hypothesis) with::

    # Walk trough the directory and subdirectory and look for workspaces
    workspace-multisig path_to_directory_containing_workspaces

Postfit plot
------------

Compute the postfit histograms and errors with::

    # --fit_var bdt_score/mmc_mass
    plot-postfit path_to_measurement_file.root --fit-var bdt_score --force-fit --jobs -1
    # If the fit has already been performed
    plot-postfit path_to_measurement_file.root --fit-var bdt_score


References
==========

https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/NuisanceParameterPullsWithRanking
https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/StatisticsTools
