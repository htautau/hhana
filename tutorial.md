# Introduction

This tutorial aims to introduce the hhana analysis package to the participants of the 2015 HLeptons+Tau workshop.
It is meant to be run on the CERN lxplus interactive computer nodes.

There is two components:
- Re-running the Run1 analysis
- Running the first background studies for run2

For the latter, the tutorial will focus on plotting the data/background comparison, workspace creation (stat-only) and significance and nuisance parameter ranking calculation.

## Run1 Analysis

### Initial setup

```bash
mkdir hhana_testbed
cd hhana_testbed/
cp /afs/cern.ch/user/q/qbuat/public/hhana_run/initialize_run1.sh .
source initialize_run1.sh
source setup_lxplus.sh
cd hhana/
source setup.sh
# lxplus uses bsub, the qsub functionality needs to be disabled
export NO_QSUB=1
```

### data/bkg plotting


```bash
plot-features
# Running in DEBUG mode to get a more verbose print out.
export DEBUG=1; plot-features --categories mva
unset DEBUG
```

### Workspace creation and 'fixing'

```bash
workspace mva --unblind --years 2012 --masses 125 --clf-mass 125;
# Optional: (running with the systematic uncert. is very long)
workspace mva --systematics --unblind --years 2012 --masses 125 --clf-mass 125;
```

```bash
fix-workspace --quiet --symmetrize --prune-shapes --chi2-thresh 0.9 --drop-others-shapes --prune-norms workspaces/hh_nos_nonisol_ebz_stat_mva

```

### Downloading the Run1 hadhad workspaces

```bash
mkdir -p workspaces
cd workspaces
svn export svn+ssh://svn.cern.ch/reps/atlasphys/Physics/Higgs/HSG4/software/workspaces/Run1Paper/hadhad/trunk run1_comb
cd ..
```

### Significance and nuisance parameter ranking

```bash
# expected significance
multisig workspaces/run1_comb/* --name combined
# observed significance
multisig workspaces/run1_comb/* --name combined --unblind
# plotting
plot-pvalues workspaces/run1_comb --categories mva --unblind
```

```bash
# Calculation of the pulls (using multithreading)
multinp pulls --file workspaces/run1_comb/hh_combination_125/measurement_hh_combination_125.root --name combined --jobs -1
# The big ugly plot
plot-ranking workspaces/run1_comb/hh_combination_125/measurement_hh_combination_125.root
# Only plot the TES components (using pattern matching)
plot-ranking workspaces/run1_comb/hh_combination_125/measurement_hh_combination_125.root --patterns *TES*
```

## Run2 Analysis

### Setup
Start from a fresh terminal and do:

```bash
mkdir hhana_testbed_run2
cp /afs/cern.ch/user/q/qbuat/public/hhana_run/initialize_run2.sh .
source initialize_run2.sh
source setup_lxplus.sh
cd hhana/
source setup.sh
```
### Running the normalization

```bash
norm --no-embedding
```

### Making the preselection plotbook

```bash
# potentially slow due to the EOS I/O and the current size of the ntuples (almost no selection)
plot-features --no-embedding --categories presel
```

### Printing the yields table
```bash
compare-fakes-model --no-embedding
```

### Getting your hands in the framework

The following code snippets are meant to be run in the python interpreter. Typing `python` will open the console and you can then paste the following blocks of code:

#### Navigating through the database

```python
# import the database from the hhdb package
from hhdb import datasets
db = datasets.Database(name='datasets_hh_c', verbose=True)
# print all the datasets keys
db.keys()
# print a specific dataset
ds = db['PoPy8_ggH125_tautauhh']
ds
# print the tabulated (cross-section, kfactor, filter efficiency)
print ds.xsec_kfact_effic
# import the sample info for Pythia Ztautau
from hhdb import samples as samples_db
sample_info = samples_db.get_sample('hadhad', 2015, 'background', 'pythia_ztautau')
# list the datasets used for this sample
sample_info['samples']
# list the datasets info for EWK (outdated but good example of the logic)
sample_info = samples_db.get_sample('hadhad', 2015, 'background', 'ewk')
sample_info['samples']
```

#### Navigating through the data tables and basic histogram filling

```python
from mva.samples.db import get_file
hfile = get_file(hdf=True)
# print the entire file
hfile
# print the list of tables
hfile.root
# get one table
H_VBF = hfile.root.PoPy8_VBFH125_tautauhh
H_VBF
# list all the variables
H_VBF.colnames
# get the numpy array (loading the table in the memory for quick access)
rec_array = H_VBF.read()
type(rec_array)
rec_array
# get a sub-table by applying some cuts
from rootpy.tree import Cut
cut = Cut('HLT_tau35_medium1_tracktwo_tau25_medium1_tracktwo_L1TAU20IM_2TAU12IM == 1')
rec_array = H_VBF.read_where(cut.where())

arr = rec_array['ditau_vis_mass']
# import rootpy Histogramming (wrapper of ROOT::TH1)
from rootpy.plotting import Hist
h_mvis = Hist(12, 30, 150)
from root_numpy import fill_hist
fill_hist(h_mvis, arr)
list(h_mvis.y())
```

#### Filling histograms using the sample classes
You can run and modify `first_plots_xtau_hadhad.py` and run it from the terminal:
```bash
python -i first_plots_xtau_hadhad.py
```
