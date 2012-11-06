#! /bin/bash
source setup_root.sh

CHANNEL=$1
shift
if [ -z "$@" ]
then
    MASSES=$(seq 100 5 150)
else
    MASSES=$@
fi

TAG=trunk
WORKSPACE=SYSTEMATICS
SCRIPT=NuisanceCheck-${TAG}

echo "Using tag ${SCRIPT}"

for mass in $MASSES
do
    echo "*************************************"
    echo "Checking ${CHANNEL} limits for mass point ${mass}"
    
    for category in ggf boosted vbf
    do
        echo "--------------------------------------"
        echo "Category: ${category}"
        echo "--------------------------------------"

        (root -l -b -q ./scripts/${SCRIPT}/FitCrossCheckForLimits.C+"(LimitCrossCheck::PlotFitCrossChecks(\
            \"./results/${CHANNEL}/${category}_${mass}_combined_${WORKSPACE}_model.root\",\
            \"./results/${CHANNEL}/\",\
            \"combined\",\
            \"ModelConfig\",\
            \"obsData\"))" 2>&1) | tee log_check_${CHANNEL}_${mass}_${category}.txt
    done
    
    echo "--------------------------------------------------"
    echo "Checking combination limit for channel ${CHANNEL}"
    echo "--------------------------------------------------"
    
    (root -l -b -q ./scripts/${SCRIPT}/FitCrossCheckForLimits.C+"(LimitCrossCheck::PlotFitCrossChecks(\
        \"./results/${CHANNEL}/${mass}_combined_${WORKSPACE}_model.root\",\
        \"./results/${CHANNEL}/\",\
        \"combined\",\
        \"ModelConfig\",\
        \"obsData\"))" 2>&1) | tee log_check_${CHANNEL}_${mass}_combined.txt
done
