

def draw_scatter(fields,
                 category,
                 region,
                 output_name,
                 backgrounds,
                 signals=None,
                 data=None,
                 signal_scale=1.,
                 signal_colors=cm.spring,
                 classifier=None,
                 cuts=None,
                 unblind=False):
    nplots = 1
    figheight = 6.
    figwidth = 6.
    background_arrays = []
    background_clf_arrays = []
    for background in backgrounds:
        background_arrays.append(
            background.merged_records(
                category, region,
                fields=fields,
                cuts=cuts))
        if classifier is not None:
            background_clf_arrays.append(
                background.scores(
                    classifier,
                    category,
                    region,
                    cuts=cuts,
                    systematics=False)['NOMINAL'][0])

    if data is not None:
        nplots += 1
        figwidth += 6.
        data_array = data.merged_records(
            category, region,
            fields=fields,
            cuts=cuts)
        if classifier is not None:
            data_clf_array = data.scores(
                classifier,
                category,
                region,
                cuts=cuts)[0]

    if signals is not None:
        nplots += 1
        figwidth += 6.
        if data is not None:
            signal_index = 3
        else:
            signal_index = 2
        signal_arrays = []
        signal_clf_arrays = []
        for signal in signals:
            signal_arrays.append(
                signal.merged_records(
                    category, region,
                    fields=fields,
                    cuts=cuts))
            if classifier is not None:
                signal_clf_arrays.append(
                    signal.scores(
                        classifier,
                        category,
                        region,
                        cuts=cuts,
                        systematics=False)['NOMINAL'][0])

    if classifier is not None:
        fields = fields + [classifier]
    all_pairs = list(itertools.combinations(fields, 2))

    for x, y in all_pairs:
        # always make the classifier along the x axis
        if not isinstance(y, basestring):
            tmp = x
            x = y
            y = tmp

        with_classifier = not isinstance(x, basestring)

        plt.figure(figsize=(figwidth, figheight), dpi=200)
        axes = []

        ax_bkg = plt.subplot(1, nplots, 1)
        axes.append(ax_bkg)

        if not with_classifier:
            xscale = VARIABLES[x].get('scale', 1.)
        yscale = VARIABLES[y].get('scale', 1.)

        xmin, xmax = float('inf'), float('-inf')
        ymin, ymax = float('inf'), float('-inf')

        for i, (array, background) in enumerate(zip(background_arrays,
                                                    backgrounds)):
            if with_classifier:
                x_array = background_clf_arrays[i]
            else:
                x_array = array[x] * xscale
            y_array = array[y] * yscale

            # update max and min bounds
            lxmin, lxmax = x_array.min(), x_array.max()
            lymin, lymax = y_array.min(), y_array.max()
            if lxmin < xmin:
                xmin = lxmin
            if lxmax > xmax:
                xmax = lxmax
            if lymin < ymin:
                ymin = lymin
            if lymax > ymax:
                ymax = lymax

            weight = array['weight']
            ax_bkg.scatter(
                    x_array, y_array,
                    c=background.hist_decor['color'],
                    label=background.label,
                    s=weight * 10,
                    #edgecolors='',
                    linewidths=1,
                    marker='o',
                    alpha=0.75)

        if data is not None:
            data_ax = plt.subplot(1, nplots, 2)
            axes.append(data_ax)

            if with_classifier:
                x_array = data_clf_array
            else:
                x_array = data_array[x] * xscale
            y_array = data_array[y] * yscale

            # if blinded don't show above the midpoint of the BDT score
            if with_classifier and not unblind:
                midpoint = (x_array.max() + x_array.min()) / 2.
                x_array = x_array[data_clf_array < midpoint]
                y_array = y_array[data_clf_array < midpoint]
                data_ax.text(0.9, 0.2, 'BLINDED',
                                  verticalalignment='center',
                                  horizontalalignment='right',
                                        transform=data_ax.transAxes,
                                        fontsize=20)

            # update max and min bounds
            lxmin, lxmax = x_array.min(), x_array.max()
            lymin, lymax = y_array.min(), y_array.max()
            if lxmin < xmin:
                xmin = lxmin
            if lxmax > xmax:
                xmax = lxmax
            if lymin < ymin:
                ymin = lymin
            if lymax > ymax:
                ymax = lymax

            weight = data_array['weight']
            data_ax.scatter(
                    x_array, y_array,
                    c='black',
                    label=data.label,
                    s=weight * 10,
                    #edgecolors='',
                    linewidths=0,
                    marker='.')

        if signal is not None:
            sig_ax = plt.subplot(1, nplots, signal_index)
            axes.append(sig_ax)

            for i, (array, signal) in enumerate(zip(signal_arrays, signals)):

                if with_classifier:
                    x_array = signal_clf_arrays[i]
                else:
                    x_array = array[x] * xscale
                y_array = array[y] * yscale

                # update max and min bounds
                lxmin, lxmax = x_array.min(), x_array.max()
                lymin, lymax = y_array.min(), y_array.max()
                if lxmin < xmin:
                    xmin = lxmin
                if lxmax > xmax:
                    xmax = lxmax
                if lymin < ymin:
                    ymin = lymin
                if lymax > ymax:
                    ymax = lymax
                color = signal_colors((i + 1) / float(len(signals) + 1))
                weight = array['weight']
                sig_ax.scatter(
                        x_array, y_array,
                        c=color,
                        label=signal.label,
                        s=weight * 10 * signal_scale,
                        #edgecolors='',
                        linewidths=0,
                        marker='o',
                        alpha=0.75)

        xwidth = xmax - xmin
        ywidth = ymax - ymin
        xpad = xwidth * .1
        ypad = ywidth * .1

        if with_classifier:
            x_name = "BDT Score"
            x_filename = "bdt_score"
            x_units = None
        else:
            x_name = VARIABLES[x]['title']
            x_filename = VARIABLES[x]['filename']
            x_units = VARIABLES[x].get('units', None)

        y_name = VARIABLES[y]['title']
        y_filename = VARIABLES[y]['filename']
        y_units = VARIABLES[y].get('units', None)

        for ax in axes:
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)

            ax.legend(loc='upper right')

            if x_units is not None:
                ax.set_xlabel('%s [%s]' % (x_name, x_units))
            else:
                ax.set_xlabel(x_name)
            if y_units is not None:
                ax.set_ylabel('%s [%s]' % (y_name, y_units))
            else:
                ax.set_ylabel(y_name)

        plt.suptitle(category.label)
        plt.savefig(os.path.join(PLOTS_DIR, 'scatter_%s_%s_%s%s.png') % (
            category.name, x_filename, y_filename, output_name),
            bbox_inches='tight')

        """
        Romain Madar:

        Display the 1D histogram of (x_i - <x>)(y_i - <y>) over the events {i}.
        The mean of this distribution will be the "usual correlation" but this
        plot allows to look at the tails and asymmetry, for data and MC.
        """


def get_2d_field_hist(var):
    var_info = VARIABLES[var]
    bins = var_info['bins']
    min, max = var_info['range']
    hist = Hist2D(100, min, max, 100, -1, 1)
    return hist


def draw_2d_hist(classifier,
                 category,
                 region,
                 backgrounds,
                 signals=None,
                 data=None,
                 cuts=None,
                 y=MMC_MASS,
                 output_suffix=''):
    fields = [y]
    background_arrays = []
    background_clf_arrays = []
    for background in backgrounds:
        sys_mass = {}
        for systematic in iter_systematics(True):
            sys_mass[systematic] = (
                background.merged_records(
                    category, region,
                    fields=fields,
                    cuts=cuts,
                    systematic=systematic))
        background_arrays.append(sys_mass)
        background_clf_arrays.append(
            background.scores(
                classifier,
                category,
                region,
                cuts=cuts,
                systematics=True))

    if signals is not None:
        signal_arrays = []
        signal_clf_arrays = []
        for signal in signals:
            sys_mass = {}
            for systematic in iter_systematics(True):
                sys_mass[systematic] = (
                    signal.merged_records(
                        category, region,
                        fields=fields,
                        cuts=cuts,
                        systematic=systematic))
            signal_arrays.append(sys_mass)
            signal_clf_arrays.append(
                signal.scores(
                    classifier,
                    category,
                    region,
                    cuts=cuts,
                    systematics=True))

    xmin, xmax = float('inf'), float('-inf')
    if data is not None:
        data_array = data.merged_records(
            category, region,
            fields=fields,
            cuts=cuts)
        data_clf_array = data.scores(
            classifier,
            category,
            region,
            cuts=cuts)[0]
        lxmin, lxmax = data_clf_array.min(), data_clf_array.max()
        if lxmin < xmin:
            xmin = lxmin
        if lxmax > xmax:
            xmax = lxmax

    for array_dict in background_clf_arrays + signal_clf_arrays:
        for sys, (array, _) in array_dict.items():
            lxmin, lxmax = array.min(), array.max()
            if lxmin < xmin:
                xmin = lxmin
            if lxmax > xmax:
                xmax = lxmax

    yscale = VARIABLES[y].get('scale', 1.)

    if cuts:
        output_suffix += '_' + cuts.safe()
    output_name = "histos_2d_" + category.name + output_suffix + ".root"
    hist_template = get_2d_field_hist(y)

    # scale BDT scores such that they are between -1 and 1
    xscale = max(abs(xmax), abs(xmin))

    with root_open(output_name, 'recreate') as f:

        for background, array_dict, clf_dict in zip(backgrounds,
                                                    background_arrays,
                                                    background_clf_arrays):
            for systematic in iter_systematics(True):
                x_array = clf_dict[systematic][0] / xscale
                y_array = array_dict[systematic][y] * yscale
                weight = array_dict[systematic]['weight']
                hist = hist_template.Clone(name=background.name +
                        ('_%s' % systematic_name(systematic)))
                hist.fill_array(np.c_[y_array, x_array], weights=weight)
                hist.Write()

        if signal is not None:
            for signal, array_dict, clf_dict in zip(signals,
                                                    signal_arrays,
                                                    signal_clf_arrays):
                for systematic in iter_systematics(True):
                    x_array = clf_dict[systematic][0] / xscale
                    y_array = array_dict[systematic][y] * yscale
                    weight = array_dict[systematic]['weight']
                    hist = hist_template.Clone(name=signal.name +
                            ('_%s' % systematic_name(systematic)))
                    hist.fill_array(np.c_[y_array, x_array], weights=weight)
                    hist.Write()

        if data is not None:
            x_array = data_clf_array / xscale
            y_array = data_array[y] * yscale
            weight = data_array['weight']
            hist = hist_template.Clone(name=data.name)
            hist.fill_array(np.c_[y_array, x_array], weights=weight)
            hist.Write()

