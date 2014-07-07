import ROOT
from .significance import make_asimov_data as _make_asimov_data


def make_asimov_data(workspace,
                     mu=1., profile=False,
                     **fit_params):
    model_config = workspace.obj('ModelConfig')
    if isinstance(profile, basestring):
        if profile == 'hat':
            fit_params.setdefault('print_level', -1)
            workspace.fit(**fit_params)
            poi = workspace.obj('ModelConfig').GetParametersOfInterest().first()
            profile_mu = poi.getVal()
            profile = True
        else:
            profile_mu = float(profile)
            profile = True
    elif profile is None:
        profile = False
        profile_mu = 1.
    elif profile in (False, True):
        profile_mu = 1.
    else:
        profile_mu = float(profile)
        profile = True
    if profile:
        pdf = model_config.GetPdf()
        data = workspace.data('obsData')
        obs_nll = pdf.createNLL(
            data, ROOT.RooFit.Constrain(model_config.GetNuisanceParameters()))
        obs_nll.__class__ = ROOT.RooNLLVar
    else:
        obs_nll = None
    return _make_asimov_data(workspace, model_config,
                             obs_nll, mu, profile_mu)
