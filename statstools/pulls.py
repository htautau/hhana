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
from nuisance import get_nuisance_params
from .import log; log = log[__name__]

class NuisancePullScan(Process):
    """
    TODO: write description
    """
    def __init__(self,
                 pickle_name,
                 ws,
                 mc,
                 poi_name,
                 np_name,
                 ws_snapshot='StartingPoint'):
        super(NuisancePullScan, self).__init__()
        self.pickle_name = pickle_name
        self.ws = ws
        self.mc = mc
        self.poi_name = poi_name
        self.np_name = np_name
        self.ws_snapshot = ws_snapshot

    def run(self):
        # get the pulls 
        poi_prefit_pull, poi_postfit_pull, np_pull = get_pull(self.ws, self.mc, self.poi_name,
                                                              self.np_name, self.ws_snapshot)

        # write the value into a pickle
        with lock(self.pickle_name):
            with open(self.pickle_name) as pickle_file:
                pulls = pickle.load(pickle_file)
                if not isinstance(pulls, dict):
                    pulls = {}
                pulls[self.np_name] = {'poi_prefit':poi_prefit_pull, 'poi_postfit':poi_postfit_pull, 'np':np_pull}
            with open(self.pickle_name, 'w') as pickle_file:
                    pickle.dump(pulls, pickle_file)


# ------------------------------------------------
def get_pull(ws, mc, poi_name, np_name, ws_snapshot):
    """
    TODO: Write a description
    """
    nuisance_params = mc.GetNuisanceParameters()
    np = nuisance_params.find(np_name)
    params_of_interest = mc.GetParametersOfInterest()
    poi = params_of_interest.find(poi_name)

    ws.loadSnapshot(ws_snapshot)
    np_nom_val = np.getVal()
    poi_nom_val = poi.getVal()
    log.info( '{0} nominal value = {1}'.format(poi_name, poi_nom_val))
    log.info( '{0} nominal value = {1}'.format(np_name, np_nom_val))

    # --------------
    param_const = get_nuisance_params(mc)
    for key in param_const.keys():
            param_const[key] = True

    param_const[np_name] = False
    roo_min = asrootpy(ws).fit(param_const=param_const, print_level=-1)
    fitres = roo_min.save()
    fitres.Print()
    np_fitted_val = (np.getErrorLo(), np_nom_val, np.getErrorHi())
    param_const[np_name] = True

    #--------------------
    ws.loadSnapshot(ws_snapshot)
    np.setVal(np_nom_val+np_fitted_val[1])
    roo_min = asrootpy(ws).fit(param_const=param_const, print_level=-1)
    fitres = roo_min.save()
    fitres.Print()
    poi_up_val = poi.getVal()
    
    #--------------------
    ws.loadSnapshot(ws_snapshot)
    np.setVal(np_nom_val-np_fitted_val[1])
    roo_min = asrootpy(ws).fit(param_const=param_const, print_level=-1)
    fitres = roo_min.save()
    fitres.Print()
    poi_down_val = poi.getVal()

    #--------------------
    ws.loadSnapshot(ws_snapshot)
    np.setVal(np_nom_val+1)
    roo_min = asrootpy(ws).fit(param_const=param_const, print_level=-1)
    fitres = roo_min.save()
    fitres.Print()
    poi_prefit_up_val = poi.getVal()
    
    #--------------------
    ws.loadSnapshot(ws_snapshot)
    np.setVal(np_nom_val-1)
    roo_min = asrootpy(ws).fit(param_const=param_const, print_level=-1)
    fitres = roo_min.save()
    fitres.Print()
    poi_prefit_down_val = poi.getVal()

    #--------------------
    poi_prefit_val = (poi_prefit_down_val, poi_nom_val, poi_prefit_up_val)
    poi_fitted_val = (poi_down_val, poi_nom_val, poi_up_val)
    log.info( '{0} pulls = {1}'.format(poi_name, poi_prefit_val))
    log.info( '{0} pulls = {1}'.format(poi_name, poi_fitted_val))
    log.info( '{0} pulls = {1}'.format(np_name, np_fitted_val))

    return poi_prefit_val, poi_fitted_val, np_fitted_val

    # --> Step 1: global fit + save snapshot
    # --> Step 2 (for each np)
    # Step 2.1: save the best fit value of the NP
    # Step 2.2: set all parameters (except the one of interest) to constant
    # Step 2.3: fit the NP and get the errors on it (two methods exist)
    # Step 2.4: Redo the global fit with all NP fixed and the one studied fixed at nom+/-err
    #           to get the variation on the POI


    
