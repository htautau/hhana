# ---> python imports
from multiprocessing import Process
import pickle
import os

# ---> root/rootpy imports
from rootpy import asrootpy
from rootpy.io import root_open
from rootpy.utils.lock import lock
from rootpy.stats import Workspace

# ---> local imports
from .import log; log = log[__name__]

class NuisParScan(Process):
    """
    Process to run a fit of a workspace by
    fixing a NP to a given value and floating
    all the other fit parameters. This class is meant to be instanciate
    in a tuple or a list of processes to be run in //
    """
    def __init__(self,
                 pickle_name,
                 ws,
                 mc,
                 nuispar_name,
                 nuispar_val,
                 ws_snapshot='StartingPoint'):
        super(NuisParScan, self).__init__()
        self.pickle_name = pickle_name
        self.ws = ws
        self.mc = mc
        self.nuispar_name = nuispar_name
        self.nuispar_val = nuispar_val
        self.ws_snapshot = ws_snapshot
    def run(self):
        # get the nll value for the given nuisance parameter fixed at the given val
        nll = get_nuis_nll(self.ws,
                           self.mc,
                           self.nuispar_name,
                           self.nuispar_val,
                           self.ws_snapshot)

        # write the value into a pickle
        with lock(self.pickle_name):
            with open(self.pickle_name) as pickle_file:
                scans = pickle.load(pickle_file)
                scans.append((self.nuispar_val, nll))
            with open(self.pickle_name, 'w') as pickle_file:
                pickle.dump(scans, pickle_file)


# ------------------------------------------------
def get_nuis_nll(ws, mc, nuispar_name, nuispar_val, ws_snapshot):
    """
    Perform the global fit with all the NPs floating except the scanned one.
    This one is fixed to a given value (between -5 and 5)
    - Parameters:
    - ws: RooWorkspace
    - mc: ModelConfig
    - nuispar_name: string name of the NP (alpha_ATLAS_***)
    - nuispar_val: chosen value of the NP nuispar_name
    - ws_snapshot: snapshot name of the nominal (all floating) fit
    """
    nuisance_params = mc.GetNuisanceParameters()
    nuispar = nuisance_params.find(nuispar_name)
    ws.loadSnapshot(ws_snapshot)
    nuispar.setVal(nuispar_val)
    param_const = get_nuisance_params(mc, constant=[nuispar_name])
    roo_min = asrootpy(ws).fit(param_const=param_const, print_level=-1)
    fitres = roo_min.save()
    log.info('for {0} = {1}, nll = {2}'.format(nuispar_name,
                                               nuispar_val,
                                               fitres.minNll()))
    return fitres.minNll()


# ------------------------------------------------
def get_nuis_nll_nofit(ws, mc, nll_func, np_name, ws_snapshot):
    """
    Return a list of nll values computed for different values
    of a given nuisance parameter (NP)
    Parameters:
    - ws: RooWorkspace
    - mc: ModelConfig
    - nll_func: RooAbsReal object (nll function)
    - np_name: Name of the NP of interest
    - ws_snapshot: Name of the nominal fit snapshot
    TODO: Write description
    """
    nuisance_params = mc.GetNuisanceParameters()
    np = nuisance_params.find(np_name)
    ws.loadSnapshot(ws_snapshot)
    np_scans = []
    NP_TESTED_VALS = [0.1*i for i in range(-50,51)] #range(-5,6)
    for val in NP_TESTED_VALS:
        np.setVal(val)
        np_scans.append((val, nll_func.getVal()))
    return sorted(np_scans)




# ------------------------------------------------
def get_nuisance_params(mc, constant=None):
    """
    Return a dictionary of the nuisance parameters
    setting them all to floating except for params specified
    in constant
    Parameters:
    - mc: modelconfig object
    - constant: list of constant parameters
    """
    nuisIter = mc.GetNuisanceParameters().createIterator()
    params = {}
    while True:
        nuis = nuisIter.Next()
        if not nuis:
            break
        params[nuis.GetName()] = False
    if isinstance(constant, (list, tuple)):
        for par_name in constant:
            params[par_name] = True
    return params

