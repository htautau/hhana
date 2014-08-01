from . import log
from .draw import draw
from rootpy.stats.histfactory import HistoSys, split_norm_shape


def draw_samples(
        hist_template,
        expr,
        category,
        region,
        model,
        data=None,
        signal=None,
        cuts=None,
        ravel=True,
        weighted=True,
        **kwargs):
    """
    extra kwargs are passed to draw()
    """
    hist_template = hist_template.Clone()
    hist_template.Reset()
    ndim = hist_template.GetDimension()

    model_hists = []
    for sample in model:
        hist = hist_template.Clone(title=sample.label, **sample.hist_decor)
        hist.decorate(**sample.hist_decor)
        sample.draw_into(hist, expr,
                category, region, cuts,
                weighted=weighted)
        if ndim > 1 and ravel:
            # ravel() the nominal and systematics histograms
            sys_hists = getattr(hist, 'systematics', None)
            hist = hist.ravel()
            hist.title = sample.label
            hist.decorate(**sample.hist_decor)
            if sys_hists is not None:
                hist.systematics = sys_hists
            if hasattr(hist, 'systematics'):
                sys_hists = {}
                for term, _hist in hist.systematics.items():
                    sys_hists[term] = _hist.ravel()
                hist.systematics = sys_hists
        model_hists.append(hist)

    if signal is not None:
        signal_hists = []
        for sample in signal:
            hist = hist_template.Clone(title=sample.label, **sample.hist_decor)
            hist.decorate(**sample.hist_decor)
            sample.draw_into(hist, expr,
                    category, region, cuts,
                    weighted=weighted)
            if ndim > 1 and ravel:
                # ravel() the nominal and systematics histograms
                sys_hists = getattr(hist, 'systematics', None)
                hist = hist.ravel()
                hist.title = sample.label
                hist.decorate(**sample.hist_decor)
                if sys_hists is not None:
                    hist.systematics = sys_hists
                if hasattr(hist, 'systematics'):
                    sys_hists = {}
                    for term, _hist in hist.systematics.items():
                        sys_hists[term] = _hist.ravel()
                    hist.systematics = sys_hists
            signal_hists.append(hist)
    else:
        signal_hists = None

    if data is not None:
        data_hist = hist_template.Clone(title=data.label, **data.hist_decor)
        data_hist.decorate(**data.hist_decor)
        data.draw_into(data_hist, expr, category, region, cuts,
                       weighted=weighted)
        if ndim > 1 and ravel:
            data_hist = data_hist.ravel()

        log.info("Data events: %d" % sum(data_hist.y()))
        log.info("Model events: %f" % sum(sum(model_hists).y()))
        for hist in model_hists:
            log.info("{0} {1}".format(hist.GetTitle(), sum(hist.y())))
        if signal is not None:
            log.info("Signal events: %f" % sum(sum(signal_hists).y()))
        log.info("Data / Model: %f" % (sum(data_hist.y()) /
            sum(sum(model_hists).y())))

    else:
        data_hist = None

    draw(model=model_hists,
         data=data_hist,
         signal=signal_hists,
         category=category,
         **kwargs)


def draw_samples_array(
        vars,
        category,
        region,
        model,
        data=None,
        signal=None,
        cuts=None,
        ravel=False,
        weighted=True,
        weight_hist=None,
        clf=None,
        min_score=None,
        max_score=None,
        plots=None,
        output_suffix='',
        unblind=False,
        **kwargs):
    """
    extra kwargs are passed to draw()
    """
    # filter out plots that will not be made
    used_vars = {}
    field_scale = {}
    if plots is not None:
        for plot in plots:
            if plot in vars:
                var_info = vars[plot]
                if (var_info.get('cats', None) is not None and
                    category.name.upper() not in var_info['cats']):
                    raise ValueError(
                        "variable %s is not valid in the category %s" %
                        (plot, category.name.upper()))
                used_vars[plot] = var_info
                if 'scale' in var_info:
                    field_scale[plot] = var_info['scale']
            else:
                raise ValueError(
                    "variable %s is not defined in mva/variables.py" % plot)
    else:
        for expr, var_info in vars.items():
            if (var_info.get('cats', None) is not None and
                category.name.upper() not in var_info['cats']):
                continue
            used_vars[expr] = var_info
            if 'scale' in var_info:
                field_scale[expr] = var_info['scale']
    vars = used_vars
    if not vars:
        raise RuntimeError("no variables selected")

    model_hists = []
    for sample in model:
        field_hist, _ = sample.get_field_hist(vars)
        sample.draw_array(field_hist,
            category, region, cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            clf=clf,
            min_score=min_score,
            max_score=max_score,)
        model_hists.append(field_hist)

    if signal is not None:
        signal_hists = []
        for sample in signal:
            field_hist, _ = sample.get_field_hist(vars)
            sample.draw_array(field_hist,
                category, region, cuts,
                weighted=weighted,
                field_scale=field_scale,
                weight_hist=weight_hist,
                clf=clf,
                min_score=min_score,
                max_score=max_score)
            signal_hists.append(field_hist)
    else:
        signal_hists = None

    if data is not None:
        data_field_hist, _ = data.get_field_hist(vars)
        data.draw_array(data_field_hist, category, region, cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            clf=clf,
            min_score=min_score,
            max_score=max_score)
        """
        log.info("Data events: %d" % sum(data_hist))
        log.info("Model events: %f" % sum(sum(model_hists)))
        for hist in model_hists:
            log.info("{0} {1}".format(hist.GetTitle(), sum(hist)))
        if signal is not None:
            log.info("Signal events: %f" % sum(sum(signal_hists)))
        log.info("Data / Model: %f" % (sum(data_hist) /
            sum(sum(model_hists))))
        """

    else:
        data_field_hist = None

    figs = {}
    for field, var_info in vars.items():
        if unblind:
            blind = False
        else:
            blind = var_info.get('blind', False)
        output_name = var_info['filename'] + output_suffix
        if cuts:
            output_name += '_' + cuts.safe()

        fig = draw(model=[m[field] for m in model_hists],
             data=data_field_hist[field] if data_field_hist else None,
             data_info=str(data_field_hist[field].datainfo) if data_field_hist else None,
             signal=[s[field] for s in signal_hists] if signal_hists else None,
             category=category,
             name=var_info['root'],
             units=var_info.get('units', None),
             output_name=output_name,
             blind=blind,
             integer=var_info.get('integer', False),
             **kwargs)
        figs[field] = fig
    return figs


def draw_channel(channel, fit=None, no_data=False,
                 ypadding=None, log_ypadding=None, **kwargs):
    """
    Draw a HistFactory::Channel only include OverallSys systematics
    in resulting band as an illustration of the level of uncertainty
    since correlations of the NPs are not known and it is not
    possible to draw the statistically correct error band.
    """
    if fit is not None:
        log.warning("applying snapshot on channel {0}".format(channel.name))
        channel = channel.apply_snapshot(fit)
    if channel.data and channel.data.hist and not no_data:
        data_hist = channel.data.hist
    else:
        data_hist = None
    model_hists = []
    signal_hists = []
    systematics_terms = {}
    sys_names = channel.sys_names()
    for sample in channel.samples:
        nominal_hist = sample.hist
        _systematics = {}
        for sys_name in sys_names:
            systematics_terms[sys_name] = (
                sys_name + '_UP',
                sys_name + '_DOWN')
            dn_hist, up_hist = sample.sys_hist(sys_name)
            hsys = HistoSys(sys_name, low=dn_hist, high=up_hist)
            norm, shape = split_norm_shape(hsys, nominal_hist)
            # include only overallsys component
            _systematics[sys_name + '_DOWN'] = nominal_hist * norm.low
            _systematics[sys_name + '_UP'] = nominal_hist * norm.high
        nominal_hist.systematics = _systematics
        if sample.GetNormFactor('SigXsecOverSM') is not None:
            signal_hists.append(nominal_hist)
        else:
            model_hists.append(nominal_hist)
    if 'systematics' in kwargs:
        del kwargs['systematics']
    figs = []
    for logy in (False, True):
        figs.append(draw(
            data=data_hist,
            model=model_hists or None,
            signal=signal_hists or None,
            systematics=systematics_terms,
            logy=logy,
            ypadding=(log_ypadding or ypadding) if logy else ypadding,
            **kwargs))
    return figs


def draw_channel_array(
        analysis,
        vars,
        category,
        region,
        cuts=None,
        mass=125,
        mode=None,
        scale_125=False,
        ravel=False,
        weighted=True,
        weight_hist=None,
        clf=None,
        min_score=None,
        max_score=None,
        templates=None,
        plots=None,
        output_suffix='',
        unblind=False,
        bootstrap_data=False,
        **kwargs):
    # filter out plots that will not be made
    used_vars = {}
    field_scale = {}
    if plots is not None:
        for plot in plots:
            if plot in vars:
                var_info = vars[plot]
                if (var_info.get('cats', None) is not None and
                    category.name.upper() not in var_info['cats']):
                    raise ValueError(
                        "variable %s is not valid in the category %s" %
                        (plot, category.name.upper()))
                used_vars[plot] = var_info
                if 'scale' in var_info:
                    field_scale[plot] = var_info['scale']
            else:
                raise ValueError(
                    "variable %s is not defined in mva/variables.py" % plot)
    else:
        for expr, var_info in vars.items():
            if (var_info.get('cats', None) is not None and
                category.name.upper() not in var_info['cats']):
                continue
            used_vars[expr] = var_info
            if 'scale' in var_info:
                field_scale[expr] = var_info['scale']
    vars = used_vars
    if not vars:
        raise RuntimeError("no variables selected")

    field_channel = analysis.get_channel_array(vars,
        category, region, cuts,
        include_signal=True,
        mass=mass,
        mode=mode,
        scale_125=scale_125,
        clf=clf,
        min_score=min_score,
        max_score=max_score,
        weighted=weighted,
        templates=templates,
        field_scale=field_scale,
        weight_hist=weight_hist,
        no_signal_fixes=True,
        bootstrap_data=bootstrap_data)

    figs = {}
    for field, var_info in vars.items():
        if unblind:
            blind = False
        else:
            blind = var_info.get('blind', False)
        output_name = var_info['filename'] + output_suffix
        if cuts:
            output_name += '_' + cuts.safe()
        ypadding = kwargs.pop('ypadding', var_info.get('ypadding', None))
        log_ypadding = kwargs.pop('log_ypadding', var_info.get('log_ypadding', None))
        legend_position = kwargs.pop('legend_position', var_info.get('legend', 'right'))
        fig = draw_channel(field_channel[field],
                           data_info=str(analysis.data.info),
                           category=category,
                           name=var_info['root'],
                           units=var_info.get('units', None),
                           output_name=output_name,
                           blind=blind,
                           integer=var_info.get('integer', False),
                           ypadding=ypadding,
                           log_ypadding=log_ypadding,
                           legend_position=legend_position,
                           **kwargs)
        figs[field] = fig
    return field_channel, figs
