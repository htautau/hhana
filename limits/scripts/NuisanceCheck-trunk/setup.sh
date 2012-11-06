# lxplus only
source /afs/cern.ch/sw/lcg/contrib/gcc/4.3/x86_64-slc5-gcc43-opt/setup.sh

# ROOT and Python setup (CHECK ROOT VERSION HERE)
export ROOTSYS=/afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.02/x86_64-slc5-gcc43-opt/root

export PYTHONDIR=/afs/cern.ch/sw/lcg/external/Python/2.6.2/x86_64-slc5-gcc43-opt
export LD_LIBRARY_PATH=$ROOTSYS/lib:$PYTHONDIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/afs/cern.ch/sw/lcg/external/xrootd/3.1.0p2/x86_64-slc5-gcc43-opt/lib64:$ROOTSYS/lib:$PYTHONDIR/lib:$LD_LIBRARY_PATH
export PATH=$ROOTSYS/bin:$PYTHONDIR/bin:$PATH
export PYTHONPATH=$ROOTSYS/lib:$PYTHONDIR/lib:$PYTHONPATH

# BOOST
export BOOSTDIR=/afs/cern.ch/sw/lcg/external/Boost/1.47.0_python2.6/x86_64-slc5-gcc43-opt
export BOOSTFLAGS=$BOOSTDIR/include/boost-1_47
export BOOSTLIB=$BOOSTDIR/lib

