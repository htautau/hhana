import ROOT
from .extern import make_asimov_data as _make_asimov_data


def make_asimov_data(workspace,
                     mu=1., profile=False,
                     **fit_params):
    model_config = workspace.obj('ModelConfig')
    # remember nominal workspace
    workspace.saveSnapshot('nominal_obs', model_config.global_observables)
    workspace.saveSnapshot('nominal_nuis', model_config.nuisance_parameters)
    workspace.saveSnapshot('nominal_poi', model_config.poi)
    if isinstance(profile, basestring):
        if profile == 'hat':
            fit_params.setdefault('print_level', -1)
            # unconditional fit
            workspace.fit(**fit_params)
            poi = model_config.poi.first()
            profile_mu = poi.getVal()
            profile = True
            # reset workspace
            workspace.loadSnapshot('nominal_obs')
            workspace.loadSnapshot('nominal_nuis')
            workspace.loadSnapshot('nominal_poi')
        else:
            profile_mu = float(profile)
            profile = True
    elif profile is None:
        profile = False
        profile_mu = 1.
    elif profile is False or profile is True:
        profile_mu = 1.
    else:
        profile_mu = float(profile)
        profile = True
    if profile:
        pdf = model_config.GetPdf()
        data = workspace.data('obsData')
        obs_nll = pdf.createNLL(
            data, ROOT.RooFit.Constrain(model_config.nuisance_parameters))
        obs_nll.__class__ = ROOT.RooNLLVar
    else:
        obs_nll = None
    asimov = _make_asimov_data(workspace, model_config,
                               obs_nll, mu, profile_mu)
    # reset workspace
    workspace.loadSnapshot('nominal_obs')
    workspace.loadSnapshot('nominal_nuis')
    workspace.loadSnapshot('nominal_poi')
    return asimov
