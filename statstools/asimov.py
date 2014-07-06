import ROOT
from .significance import makeAsimovData


def make_asimov_data(workspace,
                     mu=1., mu_profile=None,
                     unblind=False,
                     **fit_params):
    model_config = workspace.obj('ModelConfig')
    workspace.loadSnapshot('conditionalNuis_0')
    if mu_profile is None:
        mu_profile = mu
    elif mu_profile == 'hat':
        fit_params.setdefault('print_level', -1)
        workspace.fit(**fit_params)
        poi = model_config.GetParametersOfInterest().first()
        mu_profile = poi.getVal()
    if unblind:
        pdf = model_config.GetPdf()
        data = workspace.data('obsData')
        obs_nll = pdf.createNLL(
            data, ROOT.RooFit.Constrain(model_config.GetNuisanceParameters()))
        obs_nll.__class__ = ROOT.RooNLLVar
    else:
        obs_nll = None
    return makeAsimovData(workspace, model_config,
                          unblind, obs_nll,
                          mu, mu_profile)
