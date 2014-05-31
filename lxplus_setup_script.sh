curl -O http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.10-patch1.tar.bz2
tar xvf hdf5-1.8.10-patch1.tar.bz2
cd hdf5-1.8.10-patch1
./configure
make install
export HDF5_DIR="`pwd`/hdf5/"
cd ..

curl -O https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.9.tar.gz
tar xvfz virtualenv-1.9.tar.gz
d virtualenv-1.9
python virtualenv.py atlasVE
source atlasVE/bin/activate
cd ..

pip install numpy
pip install scipy
pip install matplotlib
pip install readline
pip install termcolor
pip install lxml
pip install PyYAML
pip install cython
pip install numexpr
pip install tables
pip install ordereddict

git clone https://github.com/rootpy/rootpy.git
cd rootpy
python setup.py install
cd ..

git clone https://github.com/rootpy/root_numpy.git
cd root_numpy
python setup.py install
cd ..

git clone https://github.com/ndawe/atlastools.git
cd atlastools
python setup.py install
cd ..

git clone https://github.com/ndawe/goodruns.git
cd goodruns
python setup.py install
cd ..

git clone https://github.com/ndawe/yellowhiggs.git
cd yellowhiggs
python setup.py install
cd ..

git clone https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
git remote add ndawe https://github.com/ndawe/scikit-learn.git
git fetch ndawe
git checkout ensemble_grid_search
python setup.py install
cd ..

svn co svn+ssh://svn.cern.ch/reps/atlasphys/Physics/Higgs/HSG4/software/common/externaltools/trunk externaltools
cd externaltools
./fetch -u kongt
./patch
./waf configure build install

