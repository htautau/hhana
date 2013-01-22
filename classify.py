from logger import log; log = log[__name__]

import pickle
from operator import itemgetter

import numpy as np
# for reproducibilty
# especially for test/train set selection
np.random.seed(1987) # my birth year ;)

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

# scikit-learn imports
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.grid_search import BoostGridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from rootpy.plotting import Hist
from rootpy.io import open as ropen
from rootpy.extern.tabulartext import PrettyTable

from samples import *
from utils import draw
import variables


def search_flat_bins(bkg_scores, min_score, max_score, bins):

    scores = []
    weights = []
    for bkg, scores_dict in bkg_scores:
        s, w = scores_dict['NOMINAL']
        scores.append(s)
        weights.append(w)
    scores = np.concatenate(scores)
    weights = np.concatenate(weights)

    selection = (min_score <= scores) & (scores < max_score)
    scores = scores[selection]
    weights = weights[selection]

    sort_idx = np.argsort(scores)
    scores = scores[sort_idx]
    weights = weights[sort_idx]

    total_weight = weights.sum()
    bin_width = total_weight / bins

    # inefficient linear search for now
    weights_cumsum = np.cumsum(weights)
    boundaries = [min_score]
    curr_total = bin_width
    for i, cs in enumerate(weights_cumsum):
        if cs >= curr_total:
            boundaries.append((scores[i] + scores[i+1])/2)
            curr_total += bin_width
        if len(boundaries) == bins:
            break
    boundaries.append(max_score)
    return boundaries


def plot_grid_scores(
        grid_scores, best_point, params, name,
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
            return str(leaves[int(x)])

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
    plt.savefig("grid_scores_%s.%s" % (name, format), bbox_inches='tight')
    plt.clf()


def plot_clf(
        background_scores,
        category,
        category_name,
        signal_scores=None,
        signal_scale=1.,
        data_scores=None,
        name=None,
        draw_histograms=True,
        draw_data=False,
        save_histograms=False,
        bins=10,
        min_score=0,
        max_score=1,
        signal_on_top=False,
        signal_colour_map=cm.spring,
        plot_signal_significance=True,
        fill_signal=True,
        systematics=None,
        **kwargs):

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

    if data_scores is not None and draw_data:
        data, data_scores = data_scores
        data_hist = hist_template.Clone(title=data.label)
        data_hist.decorate(**data.hist_decor)
        data_hist.fill_array(data_scores)
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
        draw(data=data_hist,
             model=bkg_hists,
             signal=sig_hists,
             signal_scale=signal_scale,
             plot_signal_significance=plot_signal_significance,
             category=category,
             category_name=category_name,
             name="BDT Score",
             output_name=output_name,
             range=(min_score, max_score),
             show_ratio=data_hist is not None,
             model_colour_map=None,
             signal_colour_map=signal_colour_map,
             signal_on_top=signal_on_top,
             fill_signal=fill_signal,
             systematics=systematics,
             **kwargs)
    return bkg_hists, sig_hists, data_hist


def std(X):

    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


def rec_to_ndarray(rec, fields):

    # Creates a copy and recasts data to a consistent datatype
    return np.vstack([rec[field] for field in fields]).T


class ClassificationProblem(object):

    def __init__(self,
            signals,
            backgrounds,
            fields,
            category,
            region,
            cuts=None,
            standardize=False,
            output_suffix=""):

        self.signals = signals
        self.backgrounds = backgrounds
        self.fields = fields
        self.category = category
        self.region = region
        self.cuts = cuts
        self.standardize = standardize
        self.output_suffix = output_suffix

        assert 'weight' not in fields

        self.signal_arrs = []
        self.signal_weight_arrs = []

        for signal in signals:
            left, right = signal.split(
                category=category,
                region=region,
                fields=fields,
                cuts=cuts)
            self.signal_weight_arrs.append(
                    (left['weight'], right['weight']))
            self.signal_arrs.append(
                    (rec_to_ndarray(left, fields),
                     rec_to_ndarray(right, fields)))

        self.background_arrs = []
        self.background_weight_arrs = []

        for background in backgrounds:
            left, right = background.split(
                category=category,
                region=region,
                fields=fields,
                cuts=cuts)
            self.background_weight_arrs.append(
                    (left['weight'], right['weight']))
            self.background_arrs.append(
                    (rec_to_ndarray(left, fields),
                     rec_to_ndarray(right, fields)))

        # classifiers for the left and right partitions
        # each trained on the opposite partition
        self.clfs = None

    def train(self,
            max_sig=None,
            max_bkg=None,
            norm_sig_to_bkg=True,
            same_size_sig_bkg=True,
            remove_negative_weights=False,
            grid_search=True,
            quick=False,
            cv_nfold=5,
            use_cache=True,
            **clf_params):
        """
        Determine best BDTs on left and right partitions. Each BDT will then be
        used on the other partition.
        """
        self.clfs = [None, None]

        for partition_idx in range(2):

            # recarray -> ndarray
            signal_train = np.concatenate(map(itemgetter(partition_idx),
                self.signal_arrs))
            signal_weight_train = np.concatenate(map(itemgetter(partition_idx),
                self.signal_weight_arrs))
            background_train = np.concatenate(map(itemgetter(partition_idx),
                self.background_arrs))
            background_weight_train = np.concatenate(map(itemgetter(partition_idx),
                self.background_weight_arrs))

            if remove_negative_weights:
                # remove samples from the training sample with a negative weight
                signal_train = signal_train[signal_weight_train >= 0]
                background_train = background_train[background_weight_train >= 0]
                signal_weight_train = signal_weight_train[signal_weight_train >= 0]
                background_weight_train = background_weight_train[background_weight_train >= 0]

            if max_sig is not None and max_sig < len(signal_train):
                subsample = np.random.permutation(len(signal_train))[:max_sig_train]
                signal_train = signal_train[subsample]
                signal_weight_train = signal_weight_train[subsample]

            if max_bkg is not None and max_bkg < len(background_train):
                subsample = np.random.permutation(len(background_train))[:max_bkg_train]
                background_train = background_train[subsample]
                background_weight_train = background_weight_train[subsample]

            if same_size_sig_bkg:
                if len(background_train) > len(signal_train):
                    # random subsample of background so it's the same size as signal
                    subsample = np.random.permutation(
                        len(background_train))[:len(signal_train)]
                    background_train = background_train[subsample]
                    background_weight_train = background_weight_train[subsample]
                elif len(background_train) < len(signal_train):
                    # random subsample of signal so it's the same size as background
                    subsample = np.random.permutation(
                        len(signal_train))[:len(background_train)]
                    signal_train = signal_train[subsample]
                    signal_weight_train = signal_weight_train[subsample]

            if norm_sig_to_bkg:
                # normalize signal to background
                signal_weight_train *= (
                    background_weight_train.sum() / signal_weight_train.sum())

            log.info("Training Samples:")
            log.info("Signal: %d events, %s features" % signal_train.shape)
            log.info("Sum(signal weights): %f" % signal_weight_train.sum())
            log.info("Background: %d events, %s features" % background_train.shape)
            log.info("Sum(background weight): %f" % background_weight_train.sum())

            sample_train = np.concatenate((background_train, signal_train))
            sample_weight_train = np.concatenate(
                (background_weight_train, signal_weight_train))
            labels_train = np.concatenate(
                (np.zeros(len(background_train)), np.ones(len(signal_train))))

            if self.standardize: # TODO use same std for classification
                sample_train = std(sample_train)

            # random permutation of training sample
            perm = np.random.permutation(len(labels_train))
            sample_train = sample_train[perm]
            sample_weight_train = sample_weight_train[perm]
            labels_train = labels_train[perm]

            clf_filename = 'clf_%s%s_%d.pickle' % (
                    self.category, self.output_suffix, partition_idx)

            # train a classifier
            if use_cache and os.path.isfile(clf_filename):
                # use a previously trained classifier
                log.info("using a previously trained classifier...")
                with open(clf_filename, 'r') as f:
                    clf = pickle.load(f)
                log.info(clf)
            else:
                log.info("training a new classifier...")
                log.info("plotting input variables as they are given to the BDT")
                # draw plots of the input variables
                for i, branch in enumerate(self.fields):
                    log.info("plotting %s ..." % branch)
                    branch_data = sample_train[:,i]
                    if 'scale' in variables.VARIABLES[branch]:
                        branch_data *= variables.VARIABLES[branch]['scale']
                    _min, _max = branch_data.min(), branch_data.max()
                    plt.figure()
                    plt.hist(branch_data[labels_train==0],
                            bins=20, range=(_min, _max),
                            weights=sample_weight_train[labels_train==0],
                            label='Background', histtype='stepfilled',
                            alpha=.5)
                    plt.hist(branch_data[labels_train==1],
                            bins=20, range=(_min, _max),
                            weights=sample_weight_train[labels_train==1],
                            label='Signal', histtype='stepfilled', alpha=.5)
                    label = variables.VARIABLES[branch]['title']
                    if 'units' in variables.VARIABLES[branch]:
                        label += ' [%s]' % variables.VARIABLES[branch]['units']
                    plt.xlabel(label)
                    plt.legend()
                    plt.savefig('train_var_%s_%s%s.png' % (
                        self.category, branch, self.output_suffix))

                log.info("plotting sample weights ...")
                _min, _max = sample_weight_train.min(), sample_weight_train.max()
                plt.figure()
                plt.hist(sample_weight_train[labels_train==0],
                        bins=20, range=(_min, _max),
                        label='Background', histtype='stepfilled',
                        alpha=.5)
                plt.hist(sample_weight_train[labels_train==1],
                        bins=20, range=(_min, _max),
                        label='Signal', histtype='stepfilled', alpha=.5)
                plt.xlabel('sample weight')
                plt.legend()
                plt.savefig('train_sample_weight_%s%s.png' % (
                    self.category, self.output_suffix))

                if partition_idx == 0:

                    # grid search params
                    min_leaf_high = int((sample_train.shape[0] / 2.) *
                            (cv_nfold - 1.) / cv_nfold)
                    min_leaf_low = max(10, int(min_leaf_high / 100.))

                    if quick:
                        # quick search for testing
                        min_leaf_step = max((min_leaf_high - min_leaf_low) / 20, 1)
                        MAX_N_ESTIMATORS = 500
                        MIN_N_ESTIMATORS = 10

                    else:
                        # larger search
                        min_leaf_step = max((min_leaf_high - min_leaf_low) / 100, 1)
                        MAX_N_ESTIMATORS = 1000
                        MIN_N_ESTIMATORS = 10

                    MIN_SAMPLES_LEAF = range(
                            min_leaf_low, min_leaf_high, min_leaf_step)

                    grid_params = {
                        'base_estimator__min_samples_leaf': MIN_SAMPLES_LEAF,
                    }

                    #AdaBoostClassifier.staged_score = staged_score

                    clf = AdaBoostClassifier(
                            DecisionTreeClassifier(),
                            compute_importances=True,
                            learning_rate=.5,
                            real=False)

                    grid_clf = BoostGridSearchCV(
                            clf, grid_params,
                            max_n_estimators=MAX_N_ESTIMATORS,
                            min_n_estimators=MIN_N_ESTIMATORS,
                            #n_estimators_step=1,
                            # can use default ClassifierMixin score
                            #score_func=precision_score,
                            cv = StratifiedKFold(labels_train, cv_nfold),
                            n_jobs=20)

                    log.info("")
                    log.info("performing a grid search over these parameter values:")
                    for param, values in grid_params.items():
                        log.info('{0} {1}'.format(param.split('__')[-1], values))
                        log.info('--')
                    log.info("Minimum number of classifiers: %d" % MIN_N_ESTIMATORS)
                    log.info("Maximum number of classifiers: %d" % MAX_N_ESTIMATORS)
                    log.info("")
                    log.info("training new classifiers ...")

                    grid_clf.fit(
                            sample_train, labels_train,
                            sample_weight=sample_weight_train)

                    clf = grid_clf.best_estimator_
                    grid_scores = grid_clf.grid_scores_

                    log.info("Best score: %f" % grid_clf.best_score_)
                    log.info("Best Parameters:")
                    log.info(grid_clf.best_params_)

                    # plot a grid of the scores
                    plot_grid_scores(
                        grid_scores,
                        best_point={
                            'base_estimator__min_samples_leaf':
                            clf.base_estimator.min_samples_leaf,
                            'n_estimators':
                            clf.n_estimators},
                        params={
                            'base_estimator__min_samples_leaf':
                            'min leaf',
                            'n_estimators':
                            'trees'},
                        name=self.category + self.output_suffix + "_%d" % partition_idx)

                    # scale up the min-leaf and retrain on the whole set
                    min_samples_leaf = clf.base_estimator.min_samples_leaf

                    clf = sklearn.clone(clf)
                    clf.base_estimator.min_samples_leaf = int(
                            min_samples_leaf *
                                cv_nfold / float(cv_nfold - 1))

                    clf.fit(sample_train, labels_train,
                            sample_weight=sample_weight_train)
                    print
                    print "After scaling up min_leaf"
                    print clf

                else: # training on the other partition
                    log.info("training a new classifier ...")

                    # use same params as in first partition
                    clf = sklearn.clone(clf)
                    print clf

                    clf.fit(sample_train, labels_train,
                            sample_weight=sample_weight_train)

                with open(clf_filename, 'w') as f:
                    pickle.dump(clf, f)

            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                indices = np.argsort(importances)[::-1]
                log.info("Feature ranking:")
                print r"\begin{tabular}{c|c|c}"
                table = PrettyTable(["Rank", "Variable", "Importance"])
                print r"\hline\hline"
                print r"Rank & Variable & Importance\\"
                for f, idx in enumerate(indices):
                    table.add_row([f + 1,
                        self.fields[idx],
                        '%.3f' % importances[idx]])
                    print r"%d & %s & %.3f\\" % (f + 1,
                        variables.VARIABLES[self.fields[idx]]['title'],
                        importances[idx])
                print r"\end{tabular}"
                print
                print table.get_string(hrules=1)

            self.clfs[(partition_idx + 1) % 2] = clf

    def classify(self, sample, region, cuts=None, systematic='NOMINAL'):

        if self.clfs == None:
            raise RuntimeError("you must train the classifiers first")

        left, right = sample.split(
                category=self.category,
                region=region,
                fields=self.fields,
                cuts=cuts,
                systematic=systematic)

        left_weight = left['weight']
        right_weight = right['weight']
        left = rec_to_ndarray(left, self.fields)
        right = rec_to_ndarray(right, self.fields)

        # each classifier is never used on the partition that trained it
        left_scores = self.clfs[0].decision_function(left)
        right_scores = self.clfs[1].decision_function(right)

        return np.concatenate((left_scores, right_scores)), \
               np.concatenate((left_weight, right_weight))


def purity_score(bdt, X):

    norm = 0.
    total = None
    for w, est ∈ zip(bdt.weights_, bdt.estimators_):
        norm += w
        probs = est.predict_proba(X)
        purity = probs[:, 0]
        if total is None:
            total = purity * w
        else:
            total += purity * w
    total /= norm
    return total


def staged_score(self, X, y, sample_weight, n_estimators=-1):
    """
    calculate maximum signal significance
    """
    bins = 50
    for p in self.staged_predict_proba(X, n_estimators=n_estimators):

        scores = p[:,-1]

        # weighted mean accuracy
        y_pred = scores >= .5
        acc = np.average((y_pred == y), weights=sample_weight)

        min_score, max_score = scores.min(), scores.max()
        b_hist = Hist(bins, min_score, max_score + 0.0001)
        s_hist = b_hist.Clone()

        scores_s, w_s = scores[y==1], sample_weight[y==1]
        scores_b, w_b = scores[y==0], sample_weight[y==0]

        # fill the histograms
        s_hist.fill_array(scores_s, w_s)
        b_hist.fill_array(scores_b, w_b)

        # reverse cumsum
        #bins = list(b_hist.xedges())[:-1]
        s_counts = np.array(s_hist)
        b_counts = np.array(b_hist)
        S = s_counts[::-1].cumsum()[::-1]
        B = b_counts[::-1].cumsum()[::-1]

        # S / sqrt(S + B)
        s_sig = np.divide(list(S), np.sqrt(list(S + B)))

        #max_bin = np.argmax(np.ma.masked_invalid(significance)) #+ 1
        #max_sig = significance[max_bin]
        #max_cut = bins[max_bin]

        s_sig_max = np.max(np.ma.masked_invalid(s_sig))
        yield s_sig_max * acc


def write_score_hists(f, mass, scores_list, hist_template, no_neg_bins=True):

    sys_hists = {}
    for samp, scores_dict in scores_list:
        for sys_term, (scores, weights) in scores_dict.items():
            if sys_term == 'NOMINAL':
                suffix = ''
            else:
                suffix = '_' + '_'.join(sys_term)
            hist = hist_template.Clone(
                    name=samp.name + ('_%d' % mass) + suffix)
            hist.fill_array(scores, weights)
            if sys_term not in sys_hists:
                sys_hists[sys_term] = []
            sys_hists[sys_term].append(hist)
    f.cd()
    for sys_term, hists in sys_hists.items():
        bad_bins = []
        if no_neg_bins:
            # check for negative bins over all systematics and zero them out
            # negative bins cause lots of problem in the limit setting
            # negative bin contents effectively means
            # the same as "no events here..."
            total_hist = sum(hists)
            for bin, content in enumerate(total_hist):
                if content < 0:
                    log.warning("Found negative bin %d (%f) for systematic %s" % (
                            bin, content, sys_term))
                    bad_bins.append(bin)
        for hist in hists:
            for bin in bad_bins:
                # zero out bad bins
                hist[bin] = 0.
            hist.Write()

