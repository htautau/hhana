# Introduction

This tutorial aims to introduce the hhana analysis package to the participants of the 2015 HLeptons+Tau workshop.
It is meant to be run on the CERN lxplus interactive computer nodes.

There is two components:
- Re-running the Run1 analysis
- Running the first backgroun studies for run2

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
