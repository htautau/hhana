from statstools.utils import efficiency_cut
from matplotlib import cm
from rootpy.plotting.contrib import plot_corrcoef_matrix


def correlations(signal, signal_weight,
                 background, background_weight,
                 fields, category, output_suffix=''):
    names = [
        VARIABLES[field]['title'] if field in VARIABLES else field
            for field in fields]

    # draw correlation plots
    plot_corrcoef_matrix(signal, fields=names,
         output_name=os.path.join(PLOTS_DIR,
             "correlation_signal_%s%s.png" % (
             category.name, output_suffix)),
         title='%s Signal' % category.label,
         weights=signal_weight)
    plot_corrcoef_matrix(background, fields=names,
         output_name=os.path.join(PLOTS_DIR,
             "correlation_background_%s%s.png" % (
             category.name, output_suffix)),
         title='%s Background' % category.label,
         weights=background_weight)


def plot_grid_scores(grid_scores, best_point, params, name,
                     label_all_bins=False,
                     label_all_ticks=False,
                     n_ticks=10,
                     title=None,
                     format='png'):

    param_names = sorted(grid_scores[0][0].keys())
    param_values = dict([(pname, []) for pname in param_names])
    for pvalues, score, cv_scores in grid_scores:
        for pname in param_names:
            param_values[pname].append(pvalues[pname])

    # remove duplicates
    for pname in param_names:
        param_values[pname] = np.unique(param_values[pname]).tolist()

    scores = np.empty(shape=[len(param_values[pname]) for pname in param_names])

    for pvalues, score, cv_scores in grid_scores:
        index = []
        for pname in param_names:
            index.append(param_values[pname].index(pvalues[pname]))
        scores.itemset(tuple(index), score)

    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.axes([.12, .15, .8, .75])
    cmap = cm.get_cmap('jet', 100)
    img = ax.imshow(scores, interpolation="nearest", cmap=cmap,
            aspect='auto',
            origin='lower')

    if label_all_ticks:
        plt.xticks(range(len(param_values[param_names[1]])),
                param_values[param_names[1]])
        plt.yticks(range(len(param_values[param_names[0]])),
                param_values[param_names[0]])
    else:
        trees = param_values[param_names[1]]
        def tree_formatter(x, pos):
            if x < 0 or x >= len(trees):
                return ''
            return str(trees[int(x)])

        leaves = param_values[param_names[0]]
        def leaf_formatter(x, pos):
            if x < 0 or x >= len(leaves):
                return ''
            return '%.2f' % leaves[int(x)]

        ax.xaxis.set_major_formatter(FuncFormatter(tree_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(leaf_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(n_ticks, integer=True,
            prune='lower', steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(n_ticks, integer=True,
            steps=[1, 2, 5, 10]))
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

    ax.set_xlabel(params[param_names[1]], fontsize=12,
            position=(1., 0.), ha='right')
    ax.set_ylabel(params[param_names[0]], fontsize=12,
            position=(0., 1.), va='top')

    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    for row in range(scores.shape[0]):
        for col in range(scores.shape[1]):
            decor={}
            if ((param_values[param_names[0]].index(best_point[param_names[0]])
                 == row) and
                (param_values[param_names[1]].index(best_point[param_names[1]])
                 == col)):
                decor = dict(weight='bold',
                             bbox=dict(boxstyle="round,pad=0.5",
                                       ec='black',
                                       fill=False))
            if label_all_bins or decor:
                plt.text(col, row, "%.3f" % (scores[row][col]), ha='center',
                         va='center', **decor)
    if title:
        plt.suptitle(title)

    plt.colorbar(img, fraction=.06, pad=0.03)
    plt.axis("tight")
    plt.savefig(os.path.join(PLOTS_DIR, "grid_scores_%s.%s") % (
        name, format), bbox_inches='tight')
    plt.clf()


def hist_scores(hist, scores, systematic='NOMINAL'):
    for sample, scores_dict in scores:
        scores, weight = scores_dict[systematic]
        hist.fill_array(scores, weight)


def plot_clf(background_scores,
             category,
             signal_scores=None,
             signal_scale=1.,
             data_scores=None,
             name=None,
             draw_histograms=True,
             draw_data=False,
             save_histograms=False,
             hist_template=None,
             bins=10,
             min_score=0,
             max_score=1,
             signal_colors=cm.spring,
             systematics=None,
             unblind=False,
             **kwargs):

    if hist_template is None:
        if hasattr(bins, '__iter__'):
            # variable width bins
            hist_template = Hist(bins)
            min_score = min(bins)
            max_score = max(bins)
        else:
            hist_template = Hist(bins, min_score, max_score)

    bkg_hists = []
    for bkg, scores_dict in background_scores:
        hist = hist_template.Clone(title=bkg.label)
        scores, weight = scores_dict['NOMINAL']
        hist.fill_array(scores, weight)
        hist.decorate(**bkg.hist_decor)
        hist.systematics = {}
        for sys_term in scores_dict.keys():
            if sys_term == 'NOMINAL':
                continue
            sys_hist = hist_template.Clone()
            scores, weight = scores_dict[sys_term]
            sys_hist.fill_array(scores, weight)
            hist.systematics[sys_term] = sys_hist
        bkg_hists.append(hist)

    if signal_scores is not None:
        sig_hists = []
        for sig, scores_dict in signal_scores:
            sig_hist = hist_template.Clone(title=sig.label)
            scores, weight = scores_dict['NOMINAL']
            sig_hist.fill_array(scores, weight)
            sig_hist.decorate(**sig.hist_decor)
            sig_hist.systematics = {}
            for sys_term in scores_dict.keys():
                if sys_term == 'NOMINAL':
                    continue
                sys_hist = hist_template.Clone()
                scores, weight = scores_dict[sys_term]
                sys_hist.fill_array(scores, weight)
                sig_hist.systematics[sys_term] = sys_hist
            sig_hists.append(sig_hist)
    else:
        sig_hists = None

    if data_scores is not None and draw_data and unblind is not False:
        data, data_scores = data_scores
        if isinstance(unblind, float):
            if sig_hists is not None:
                # unblind up to `unblind` % signal efficiency
                sum_sig = sum(sig_hists)
                cut = efficiency_cut(sum_sig, 0.3)
                data_scores = data_scores[data_scores < cut]
        data_hist = hist_template.Clone(title=data.label)
        data_hist.decorate(**data.hist_decor)
        data_hist.fill_array(data_scores)
        if unblind >= 1 or unblind is True:
            log.info("Data events: %d" % sum(data_hist))
            log.info("Model events: %f" % sum(sum(bkg_hists)))
            for hist in bkg_hists:
                log.info("{0} {1}".format(hist.GetTitle(), sum(hist)))
            log.info("Data / Model: %f" % (sum(data_hist) / sum(sum(bkg_hists))))
    else:
        data_hist = None

    if draw_histograms:
        output_name = 'event_bdt_score'
        if name is not None:
            output_name += '_' + name
        for logy in (False, True):
            draw(data=data_hist,
                 model=bkg_hists,
                 signal=sig_hists,
                 signal_scale=signal_scale,
                 category=category,
                 name="BDT Score",
                 output_name=output_name,
                 show_ratio=data_hist is not None,
                 model_colors=None,
                 signal_colors=signal_colors,
                 systematics=systematics,
                 logy=logy,
                 **kwargs)
    return bkg_hists, sig_hists, data_hist


def draw_ROC(bkg_scores, sig_scores):
    # draw ROC curves for all categories
    hist_template = Hist(100, -1, 1)
    plt.figure()
    for category, (bkg_scores, sig_scores) in category_scores.items():
        bkg_hist = hist_template.Clone()
        sig_hist = hist_template.Clone()
        hist_scores(bkg_hist, bkg_scores)
        hist_scores(sig_hist, sig_scores)
        bkg_array = np.array(bkg_hist)
        sig_array = np.array(sig_hist)
        # reverse cumsum
        bkg_eff = bkg_array[::-1].cumsum()[::-1]
        sig_eff = sig_array[::-1].cumsum()[::-1]
        bkg_eff /= bkg_array.sum()
        sig_eff /= sig_array.sum()
        plt.plot(sig_eff, 1. - bkg_eff,
                 linestyle='-',
                 linewidth=2.,
                 label=category)
    plt.legend(loc='lower left')
    plt.ylabel('Background Rejection')
    plt.xlabel('Signal Efficiency')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid()
    plt.savefig(os.path.join(PLOTS_DIR, 'ROC.png'), bbox_inches='tight')


def plot_significance(signal, background, ax):

    if isinstance(signal, (list, tuple)):
        signal = sum(signal)
    if isinstance(background, (list, tuple)):
        background = sum(background)

    # plot the signal significance on the same axis
    sig_ax = ax.twinx()
    sig, max_sig, max_cut = significance(signal, background)
    bins = list(background.xedges())[:-1]

    log.info("Max signal significance %.2f at %.2f" % (max_sig, max_cut))

    sig_ax.plot(bins, sig, 'k--', label='Signal Significance')
    sig_ax.set_ylabel(r'$S / \sqrt{S + B}$',
            color='black', fontsize=15, position=(0., 1.), va='top', ha='right')
    #sig_ax.tick_params(axis='y', colors='red')
    sig_ax.set_ylim(0, max_sig * 2)
    plt.text(max_cut, max_sig + 0.02, '(%.2f, %.2f)' % (max_cut, max_sig),
            ha='right', va='bottom',
            axes=sig_ax)
    """
    plt.annotate('(%.2f, %.2f)' % (max_cut, max_sig), xy=(max_cut, max_sig),
            xytext=(max_cut + 0.05, max_sig),
                 arrowprops=dict(color='black', shrink=0.15),
                 ha='left', va='center', color='black')
    """
