from rootpy.stats import histfactory
from rootpy.utils.path import mkdir_p

from . import log; log = log[__name__]
from . import CONST_PARAMS


def write_measurements(path, mass_category_channel,
                       controls=None,
                       silence=False):
    log.info("writing measurements ...")
    if controls is None:
        controls = []
    if not os.path.exists(path):
        mkdir_p(path)
    for mass, category_channel in mass_category_channel.items():
        channels = []
        # make measurement for each category
        # include the control region in each
        for category, channel in category_channel.items():
            name = "hh_category_%s_%d" % (category, mass)
            log.info("writing {0} ...".format(name))
            # make measurement
            measurement = histfactory.make_measurement(
                name, [channel] + (
                    controls[mass].values()
                    if isinstance(controls, dict)
                    else controls),
                POI='SigXsecOverSM',
                const_params=CONST_PARAMS)
            with root_open(os.path.join(path, '{0}.root'.format(name)),
                           'recreate') as workspace_file:
                # mu=1 for Asimov data
                #measurement.SetParamValue('SigXsecOverSM', 1)
                histfactory.write_measurement(measurement,
                    root_file=workspace_file,
                    xml_path=os.path.join(path, name),
                    silence=silence)
            channels.append(channel)
        # make combined measurement
        name = "hh_combination_%d" % mass
        log.info("writing {0} ...".format(name))
        measurement = histfactory.make_measurement(
            name, channels + (
                controls[mass].values()
                if isinstance(controls, dict)
                else controls),
            POI='SigXsecOverSM',
            const_params=CONST_PARAMS)
        with root_open(os.path.join(path, '{0}.root'.format(name)),
                       'recreate') as workspace_file:
            # mu=1 for Asimov data
            #measurement.SetParamValue('SigXsecOverSM', 1)
            histfactory.write_measurement(measurement,
                root_file=workspace_file,
                xml_path=os.path.join(path, name),
                silence=silence)


def write_workspaces(path, mass_category_channel,
                     controls=None,
                     silence=False):
    log.info("writing workspaces ...")
    if controls is None:
        controls = []
    if not os.path.exists(path):
        mkdir_p(path)
    for mass, category_channel in mass_category_channel.items():
        channels = []
        # make workspace for each category
        # include the control region in each
        for category, channel in category_channel.items():
            name = "hh_category_%s_%d" % (category, mass)
            log.info("writing {0} ...".format(name))
            # make workspace
            measurement = histfactory.make_measurement(
                name, [channel] + (
                    controls[mass].values()
                    if isinstance(controls, dict)
                    else controls),
                POI='SigXsecOverSM',
                const_params=CONST_PARAMS)
            workspace = histfactory.make_workspace(measurement, name=name,
                                                   silence=silence)
            with root_open(os.path.join(path, '{0}.root'.format(name)),
                           'recreate') as workspace_file:
                workspace.Write()
                # mu=1 for Asimov data
                #measurement.SetParamValue('SigXsecOverSM', 1)
                histfactory.write_measurement(measurement,
                    root_file=workspace_file,
                    xml_path=os.path.join(path, name),
                    silence=silence)
            channels.append(channel)
        # make combined workspace
        name = "hh_combination_%d" % mass
        log.info("writing {0} ...".format(name))
        measurement = histfactory.make_measurement(
            name, channels + (
                controls[mass].values()
                if isinstance(controls, dict)
                else controls),
            POI='SigXsecOverSM',
            const_params=CONST_PARAMS)
        workspace = histfactory.make_workspace(measurement, name=name,
                                               silence=silence)
        with root_open(os.path.join(path, '{0}.root'.format(name)),
                       'recreate') as workspace_file:
            workspace.Write()
            # mu=1 for Asimov data
            #measurement.SetParamValue('SigXsecOverSM', 1)
            histfactory.write_measurement(measurement,
                root_file=workspace_file,
                xml_path=os.path.join(path, name),
                silence=silence)
