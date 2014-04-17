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

class NuisParsScan(Process):
    """
    TODO: Write description
    """
    def __init__(self,
                 pickle_name,
                 ws,
                 mc,
                 nuispar_name,
                 nuispar_val,
                 ws_snapshot='StartingPoint'):
        super(NuisParsScan, self).__init__()
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
                nuispars_scans = pickle.load(pickle_file)
                nuispars_scans[self.nuispar_name].append((self.nuispar_val, nll))
            with open(self.pickle_name, 'w') as pickle_file:
                pickle.dump(nuispars_scans, pickle_file)


# ------------------------------------------------
def get_nuis_nll(ws, mc, nuispar_name, nuispar_val, ws_snapshot):
    """
    TODO: Write description
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








