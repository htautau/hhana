import ROOT
from .extern import make_asimov_data as _make_asimov_data


def make_asimov_data(workspace,
                     mu=1., profile=False,
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
    if profile:
        pdf = model_config.GetPdf()
        data = workspace.data('obsData')
        obs_nll = pdf.createNLL(
            data,
            ROOT.RooFit.Constrain(model_config.nuisance_parameters),
            ROOT.RooFit.GlobalObservables(model_config.global_observables))
        obs_nll.__class__ = ROOT.RooNLLVar
    else:
        obs_nll = None
    asimov = _make_asimov_data(workspace, model_config,
                               obs_nll, mu, profile_mu,
                               floating_profile_mu)
    # reset workspace
    workspace.loadSnapshot('nominal_globs')
    workspace.loadSnapshot('nominal_nuis')
    workspace.loadSnapshot('nominal_poi')
    return asimov
