#! /bin/bash
source /cluster/data10/software/root-5.32-patches-64/bin/thisroot.sh

CHANNEL=$1
CHANNEL_DIR=$2

TAG=00-00-02
WORKSPACE=SYSTEMATICS
SCRIPT=NuisanceCheck-${TAG}

echo "Using tag ${SCRIPT}"

for mass in $(seq 125 5 125)
do
    echo "*************************************"
    echo "Checking ${CHANNEL} limits for mass point ${mass}"
    
    for category in ggf boosted vbf
    do
        echo "--------------------------------------"
        echo "Category: ${category}"
        echo "--------------------------------------"

        (root -l -b -q ./scripts/${SCRIPT}/FitCrossCheckForLimits.C+"(LimitCrossCheck::PlotFitCrossChecks(\
            \"./results/${CHANNEL_DIR}/${category}_${mass}_combined_${WORKSPACE}_model.root\",\
            \"./results/${CHANNEL_DIR}/\",\
            \"combined\",\
            \"ModelConfig\",\
            \"asimovData\"))" 2>&1) | tee log_check_${CHANNEL}_${mass}_${category}.txt
    done
    
    echo "--------------------------------------------------"
    echo "Checking combination limit for channel ${CHANNEL}"
    echo "--------------------------------------------------"
    
    (root -l -b -q ./scripts/${SCRIPT}/FitCrossCheckForLimits.C+"(LimitCrossCheck::PlotFitCrossChecks(\
        \"./results/${CHANNEL_DIR}/${mass}_combined_${WORKSPACE}_model.root\",\
        \"./results/${CHANNEL_DIR}/\",\
        \"combined\",\
        \"ModelConfig\",\
        \"asimovData\"))" 2>&1) | tee log_check_${CHANNEL}_${mass}_combined.txt
done
