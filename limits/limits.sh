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

rm -f limit_${CHANNEL}_combined.txt
rm -f limit_${CHANNEL}_ggf.txt
rm -f limit_${CHANNEL}_boosted.txt
rm -f limit_${CHANNEL}_vbf.txt

TAG=00-01-02
WORKSPACE=SYSTEMATICS

echo "Using tag runAsymptoticsCLs-"${TAG}

for mass in $MASSES
do
    echo "*************************************"
    echo "Calculating ${CHANNEL} limits for mass point ${mass}"
    
    for category in ggf boosted vbf
    do
        echo "--------------------------------------"
        echo "Category: ${category}"
        echo "--------------------------------------"

        (root -l -b -q ./scripts/runAsymptoticsCLs-${TAG}/runAsymptoticsCLs.C+"(\
            \"./results/${CHANNEL}/${category}_${mass}_combined_${WORKSPACE}_model.root\",\
            \"combined\",\
            \"ModelConfig\",\
            \"asimovData\",\
            \"asimovData_0\",\
            \"${CHANNEL}\",\
            \"${mass}\",\
            0.95)" 2>&1) | tee log_${CHANNEL}_${mass}_${category}.txt
        echo ${mass} >> limit_${CHANNEL}_${category}.txt
        grep -A 6 -h "Correct bands" log_${CHANNEL}_${mass}_${category}.txt >> limit_${CHANNEL}_${category}.txt

    done
    
    echo "--------------------------------------"
    echo "Combination: ${CHANNEL}"
    echo "--------------------------------------"
    
    (root -l -b -q ./scripts/runAsymptoticsCLs-${TAG}/runAsymptoticsCLs.C+"(\
        \"./results/${CHANNEL}/${mass}_combined_${WORKSPACE}_model.root\",\
        \"combined\",\
        \"ModelConfig\",\
        \"asimovData\",\
        \"asimovData_0\",\
        \"${CHANNEL}\",\
        \"${mass}\",\
        0.95)" 2>&1) | tee log_${CHANNEL}_${mass}_combined.txt
    echo ${mass} >> limit_${CHANNEL}_combined.txt
    grep -A 6 -h "Correct bands" log_${CHANNEL}_${mass}_combined.txt >> limit_${CHANNEL}_combined.txt

done
