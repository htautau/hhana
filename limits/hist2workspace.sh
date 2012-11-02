#! /bin/bash

source /cluster/data10/software/root-5.34.03-64/bin/thisroot.sh
cp $ROOTSYS/etc/HistFactorySchema.dtd ./config

CHANNEL=$1
shift
if [ -z "$@" ]
then
    MASSES=$(seq 100 5 150)
else
    MASSES=$@
fi

rm -f hist2workspace_${CHANNEL}.log

for mass in $MASSES
do
    (hist2workspace ./config/${CHANNEL}_combination_${mass}.xml 2>&1) | tee --append hist2workspace_${CHANNEL}.log
    for category in ggf boosted vbf
    do
        (hist2workspace ./config/${CHANNEL}_combination_${category}_${mass}.xml 2>&1) | tee --append hist2workspace_${CHANNEL}.log
    done
done
