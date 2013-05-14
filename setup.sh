# This script will work in either bash or zsh.

# deterine path to this script
# http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE_HIGGSTAUTAU_MVA_SETUP="${BASH_SOURCE[0]:-$0}"

DIR_HIGGSTAUTAU_MVA_SETUP="$( dirname "$SOURCE_HIGGSTAUTAU_MVA_SETUP" )"
while [ -h "$SOURCE_HIGGSTAUTAU_MVA_SETUP" ]
do 
  SOURCE_HIGGSTAUTAU_MVA_SETUP="$(readlink "$SOURCE_HIGGSTAUTAU_MVA_SETUP")"
  [[ $SOURCE_HIGGSTAUTAU_MVA_SETUP != /* ]] && SOURCE_HIGGSTAUTAU_MVA_SETUP="$DIR_HIGGSTAUTAU_MVA_SETUP/$SOURCE_HIGGSTAUTAU_MVA_SETUP"
  DIR_HIGGSTAUTAU_MVA_SETUP="$( cd -P "$( dirname "$SOURCE_HIGGSTAUTAU_MVA_SETUP"  )" && pwd )"
done
DIR_HIGGSTAUTAU_MVA_SETUP="$( cd -P "$( dirname "$SOURCE_HIGGSTAUTAU_MVA_SETUP" )" && pwd )"

echo "sourcing ${SOURCE_HIGGSTAUTAU_MVA_SETUP}..."

export HIGGSTAUTAU_NTUPLE_DIR=${DIR_HIGGSTAUTAU_MVA_SETUP}/ntuples
export HIGGSTAUTAU_LIMITS_DIR=${DIR_HIGGSTAUTAU_MVA_SETUP}/limits/data
export HIGGSTAUTAU_PLOTS_DIR=${DIR_HIGGSTAUTAU_MVA_SETUP}/plots
export HIGGSTAUTAU_MVA_DIR=$DIR_HIGGSTAUTAU_MVA_SETUP

if [ -f ${DIR_HIGGSTAUTAU_MVA_SETUP}/../higgstautau/setup.sh ]
then
    source ${DIR_HIGGSTAUTAU_MVA_SETUP}/../higgstautau/setup.sh
fi
if [ -f ${DIR_HIGGSTAUTAU_MVA_SETUP}/../TrackFit/setup.sh ]
then
    source ${DIR_HIGGSTAUTAU_MVA_SETUP}/../TrackFit/setup.sh
fi
