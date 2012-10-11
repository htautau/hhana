#! /bin/bash
source /cluster/data10/software/root-5.32-patches-64/bin/thisroot.sh

rm -f limit_hh_combined.txt
rm -f limit_hh_ggf.txt
rm -f limit_hh_boosted.txt
rm -f limit_hh_vbf.txt

TAG=00-01-02
WORKSPACE=SYSTEMATICS

echo "Using tag runAsymptoticsCLs-"${TAG}

for mass in $(seq 115 5 150)
do
    echo "*************************************"
    echo "Calculating limits for mass point ${mass}"
    
    for category in ggf boosted vbf
    do
        echo "--------------------------------------"
        echo "Category: ${category}"
        echo "--------------------------------------"

        if [ "$TAG" == "old" ]
        then
            echo "Running the old limit script"
            (root -l -b -q ./scripts/runAsymptoticsCLs-old/runAsymptoticsCLs.C+"(\
                \"./results/hadhad/hh_${category}_${mass}_combined_${WORKSPACE}_model.root\",\
                \"combined\",\
                \"ModelConfig\",\
                \"asimovData\",\
                \"asimovData_0\",\
                \"conditionalGlobs_0\",\
                \"nominalGlobs\",\
                \"${mass}\",\
                ${mass},\
                0.95)" 2>&1) | tee log_hh_${mass}_${category}.txt
        else
            (root -l -b -q ./scripts/runAsymptoticsCLs-${TAG}/runAsymptoticsCLs.C+"(\
                \"./results/hadhad/hh_${category}_${mass}_combined_${WORKSPACE}_model.root\",\
                \"combined\",\
                \"ModelConfig\",\
                \"asimovData\",\
                \"asimovData_0\",\
                \"hadhad\",\
                \"${mass}\",\
                0.95)" 2>&1) | tee log_hh_${mass}_${category}.txt
            echo ${mass} >> limit_hh_${category}.txt
            grep -A 6 -h "Correct bands" log_hh_${mass}_${category}.txt >> limit_hh_${category}.txt
        fi
    done
    
    echo "--------------------------------------"
    echo "Combination"
    echo "--------------------------------------"
    
    if [ "$TAG" == "old" ]
    then
        echo "Running the old limit script"
        (root -l -b -q ./scripts/runAsymptoticsCLs-old/runAsymptoticsCLs.C+"(\
            \"./results/hadhad/hh_${mass}_combined_${WORKSPACE}_model.root\",\
            \"combined\",\
            \"ModelConfig\",\
            \"asimovData\",\
            \"asimovData_0\",\
            \"conditionalGlobs_0\",\
            \"nominalGlobs\",\
            \"${mass}\",\
            ${mass},\
            0.95)" 2>&1) | tee log_hh_${mass}_combined.txt
    else
        (root -l -b -q ./scripts/runAsymptoticsCLs-${TAG}/runAsymptoticsCLs.C+"(\
            \"./results/hadhad/hh_${mass}_combined_${WORKSPACE}_model.root\",\
            \"combined\",\
            \"ModelConfig\",\
            \"asimovData\",\
            \"asimovData_0\",\
            \"hadhad\",\
            \"${mass}\",\
            0.95)" 2>&1) | tee log_hh_${mass}_combined.txt
        echo ${mass} >> limit_hh_combined.txt
        grep -A 6 -h "Correct bands" log_hh_${mass}_combined.txt >> limit_hh_combined.txt
    fi
done
