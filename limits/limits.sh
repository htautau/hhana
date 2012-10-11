#! /bin/bash
source /cluster/data10/software/root-5.32-patches-64/bin/thisroot.sh

CHANNEL=$1
CHANNEL_DIR=$2

rm -f limit_${CHANNEL}_combined.txt
rm -f limit_${CHANNEL}_ggf.txt
rm -f limit_${CHANNEL}_boosted.txt
rm -f limit_${CHANNEL}_vbf.txt

TAG=00-01-02
WORKSPACE=SYSTEMATICS

echo "Using tag runAsymptoticsCLs-"${TAG}

for mass in $(seq 100 5 150)
do
    echo "*************************************"
    echo "Calculating ${CHANNEL} limits for mass point ${mass}"
    
    for category in ggf boosted vbf
    do
        echo "--------------------------------------"
        echo "Category: ${category}"
        echo "--------------------------------------"

        if [ "$TAG" == "old" ]
        then
            echo "Running the old limit script"
            (root -l -b -q ./scripts/runAsymptoticsCLs-old/runAsymptoticsCLs.C+"(\
                \"./results/${CHANNEL_DIR}/${CHANNEL}_${category}_${mass}_combined_${WORKSPACE}_model.root\",\
                \"combined\",\
                \"ModelConfig\",\
                \"asimovData\",\
                \"asimovData_0\",\
                \"conditionalGlobs_0\",\
                \"nominalGlobs\",\
                \"${mass}\",\
                ${mass},\
                0.95)" 2>&1) | tee log_${CHANNEL}_${mass}_${category}.txt
        else
            (root -l -b -q ./scripts/runAsymptoticsCLs-${TAG}/runAsymptoticsCLs.C+"(\
                \"./results/${CHANNEL_DIR}/${CHANNEL}_${category}_${mass}_combined_${WORKSPACE}_model.root\",\
                \"combined\",\
                \"ModelConfig\",\
                \"asimovData\",\
                \"asimovData_0\",\
                \"hadhad\",\
                \"${mass}\",\
                0.95)" 2>&1) | tee log_${CHANNEL}_${mass}_${category}.txt
            echo ${mass} >> limit_${CHANNEL}_${category}.txt
            grep -A 6 -h "Correct bands" log_${CHANNEL}_${mass}_${category}.txt >> limit_${CHANNEL}_${category}.txt
        fi
    done
    
    echo "--------------------------------------"
    echo "Combination: ${CHANNEL}"
    echo "--------------------------------------"
    
    if [ "$TAG" == "old" ]
    then
        echo "Running the old limit script"
        (root -l -b -q ./scripts/runAsymptoticsCLs-old/runAsymptoticsCLs.C+"(\
            \"./results/${CHANNEL_DIR}/${CHANNEL}_${mass}_combined_${WORKSPACE}_model.root\",\
            \"combined\",\
            \"ModelConfig\",\
            \"asimovData\",\
            \"asimovData_0\",\
            \"conditionalGlobs_0\",\
            \"nominalGlobs\",\
            \"${mass}\",\
            ${mass},\
            0.95)" 2>&1) | tee log_${CHANNEL}_${mass}_combined.txt
    else
        (root -l -b -q ./scripts/runAsymptoticsCLs-${TAG}/runAsymptoticsCLs.C+"(\
            \"./results/${CHANNEL_DIR}/${CHANNEL}_${mass}_combined_${WORKSPACE}_model.root\",\
            \"combined\",\
            \"ModelConfig\",\
            \"asimovData\",\
            \"asimovData_0\",\
            \"hadhad\",\
            \"${mass}\",\
            0.95)" 2>&1) | tee log_${CHANNEL}_${mass}_combined.txt
        echo ${mass} >> limit_${CHANNEL}_combined.txt
        grep -A 6 -h "Correct bands" log_${CHANNEL}_${mass}_combined.txt >> limit_${CHANNEL}_combined.txt
    fi
done
