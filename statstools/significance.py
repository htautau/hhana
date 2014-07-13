import os
import pickle

from rootpy.io import root_open
from .extern import significance as _significance
from .parallel import Worker


def significance(workspace,
                 observed=False,
                 injection=1.,
                 profile=False,
                 injection_test=False,
                 **fit_params):
    floating_profile_mu = False
    profile_mu = 1.
    if isinstance(profile, basestring):
        if profile == 'hat':
            floating_profile_mu = True
        else:
            profile_mu = float(profile)
        profile = True
    elif profile is None:
        profile = False
    elif profile is not False and profile is not True:
        profile_mu = float(profile)
        profile = True
    model_config = workspace.obj('ModelConfig')
    # remember nominal workspace
    workspace.saveSnapshot('nominal_globs', model_config.global_observables)
    workspace.saveSnapshot('nominal_nuis', model_config.nuisance_parameters)
    workspace.saveSnapshot('nominal_poi', model_config.poi)
    hist = _significance(workspace, observed,
                         injection, injection_test,
                         profile, profile_mu,
                         floating_profile_mu)
    # reset workspace
    workspace.loadSnapshot('nominal_globs')
    workspace.loadSnapshot('nominal_nuis')
    workspace.loadSnapshot('nominal_poi')
    if not hist:
        # fit error
        raise RuntimeError("fit failure")
    # significance, mu, and mu error
    return hist.GetBinContent(1), hist.GetBinContent(2), hist.GetBinError(2)


class SignificanceWorker(Worker):

    def __init__(self, file, workspace_name,
                 refit=False,
                 observed=False,
                 injection=1.,
                 injection_test=False,
                 profile=False,
                 **fit_params):
        super(SignificanceWorker, self).__init__()
        self.file = file
        self.workspace_name = workspace_name
        self.refit = refit
        self.observed = observed
        self.injection = injection
        self.injection_test = injection_test
        self.profile = profile
        self.fit_params = fit_params

    def work(self):
        pickle_name = os.path.splitext(self.file)[0]
        if self.profile is not False and self.profile is not None:
            pickle_name += '_profiled_mu{0}'.format(self.profile)
        if self.observed:
            pickle_name += '_observed'
        pickle_name += '.pickle'
        if os.path.exists(pickle_name) and not self.refit:
            with open(pickle_name, 'r') as pickle_file:
                result = pickle.load(pickle_file)
            if self.workspace_name in result:
                return result[self.workspace_name]
        # get the significance of the workspace
        with root_open(self.file) as file:
            ws = file[self.workspace_name]
            result = significance(ws,
                                  observed=self.observed,
                                  injection=self.injection,
                                  injection_test=self.injection_test,
                                  profile=self.profile,
                                  **self.fit_params)
        # write the value into a pickle
        with open(pickle_name, 'w') as pickle_file:
            pickle.dump({self.workspace_name: result}, pickle_file)
        return result
