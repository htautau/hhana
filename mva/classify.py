import pickle
from operator import itemgetter

import numpy as np

from matplotlib import pyplot as plt

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
from rootpy.io import root_open as ropen
from rootpy.extern.tabulartext import PrettyTable

from .samples import *
from . import log; log = log[__name__]
from . import CACHE_DIR
from .systematics import SYSTEMATICS
from .plotting import (draw, plot_clf,
                       plot_grid_scores, draw_scatter,
                       correlations)
from . import variables
from . import LIMITS_DIR, PLOTS_DIR
from .stats.utils import get_safe_template
from .utils import rec_to_ndarray, std


def print_feature_ranking(clf, fields):

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
                fields[idx],
                '%.3f' % importances[idx]])
            print r"%d & %s & %.3f\\" % (f + 1,
                variables.VARIABLES[fields[idx]]['title'],
                importances[idx])
        print r"\end{tabular}"
        print
        print table.get_string(hrules=1)


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


class ClassificationProblem(object):

    # minimal list of spectators
    SPECTATORS = [
        'mmc_resonance_pt',
        'mass_mmc_tau1_tau2',
    ]

    def __init__(self,
                 fields,
                 category,
                 region,
                 cuts=None,
                 spectators=None,
                 standardize=False,
                 output_suffix=""):

        self.fields = fields
        self.category = category
        self.region = region
        self.spectators = spectators
        self.standardize = standardize
        self.output_suffix = output_suffix
        self.background_label = 0
        self.signal_label = 1

        if spectators is None:
            spectators = []

        # merge in minimal list of spectators
        for spec in ClassificationProblem.SPECTATORS:
            if spec not in spectators and spec not in fields:
                spectators.append(spec)

        self.all_fields = fields + spectators

        assert 'weight' not in fields

        # classifiers for the left and right partitions
        # each trained on the opposite partition
        self.clfs = None

    def train(self,
              signals,
              backgrounds,
              cuts=None,
              max_sig=None,
              max_bkg=None,
              norm_sig_to_bkg=True,
              same_size_sig_bkg=False, # NOTE: if True this crops signal a lot!!
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
        if use_cache:
            # attempt to load existing classifiers
            clfs = [None, None]
            for partition_idx in range(2):

                clf_filename = os.path.join(CACHE_DIR, 'classify',
                        'clf_%s%s_%d.pickle' % (
                        self.category.name, self.output_suffix, partition_idx))

                log.info("attempting to open %s ..." % clf_filename)
                if os.path.isfile(clf_filename):
                    # use a previously trained classifier
                    log.info("found existing classifier in %s" % clf_filename)
                    with open(clf_filename, 'r') as f:
                        clf = pickle.load(f)
                    log.info(clf)
                    print_feature_ranking(clf, self.fields)
                    # check that testing on training sample gives better
                    # performance by swapping the following lines
                    #clfs[partition_idx] = clf
                    clfs[(partition_idx + 1) % 2] = clf
                else:
                    log.warning("could not open %s" % clf_filename)
                    use_cache = False
                    break
            if use_cache:
                self.clfs = clfs
                log.info("using previously trained classifiers")
                return
            else:
                log.warning(
                    "unable to load previously trained "
                    "classifiers; will train new ones")

        signal_recs = []
        signal_arrs = []
        signal_weight_arrs = []

        for signal in signals:
            left, right = signal.partitioned_records(
                category=self.category,
                region=self.region,
                fields=self.all_fields,
                cuts=cuts)
            signal_weight_arrs.append(
               (left['weight'], right['weight']))
            signal_arrs.append(
               (rec_to_ndarray(left, self.fields),
                rec_to_ndarray(right, self.fields)))
            signal_recs.append((left, right))

        background_recs = []
        background_arrs = []
        background_weight_arrs = []

        for background in backgrounds:
            left, right = background.partitioned_records(
                category=self.category,
                region=self.region,
                fields=self.all_fields,
                cuts=cuts)
            background_weight_arrs.append(
               (left['weight'], right['weight']))
            background_arrs.append(
               (rec_to_ndarray(left, self.fields),
                rec_to_ndarray(right, self.fields)))
            background_recs.append((left, right))

        # draw correlation plots
        corr_signal = np.hstack(map(np.hstack, signal_recs))
        corr_signal_weight = np.concatenate(map(np.concatenate,
            signal_weight_arrs))
        corr_background = np.hstack(map(np.hstack, background_recs))
        corr_background_weight = np.concatenate(map(np.concatenate,
            background_weight_arrs))

        correlations(
            signal=rec_to_ndarray(corr_signal, self.all_fields),
            signal_weight=corr_signal_weight,
            background=rec_to_ndarray(corr_background, self.all_fields),
            background_weight=corr_background_weight,
            fields=self.all_fields,
            category=self.category,
            output_suffix=self.output_suffix)

        self.clfs = [None, None]

        for partition_idx in range(2):

            # train a classifier
            # merge arrays and create training samples
            signal_train = np.concatenate(map(itemgetter(partition_idx),
                signal_arrs))
            signal_weight_train = np.concatenate(map(itemgetter(partition_idx),
                signal_weight_arrs))
            background_train = np.concatenate(map(itemgetter(partition_idx),
                background_arrs))
            background_weight_train = np.concatenate(map(itemgetter(partition_idx),
                background_weight_arrs))

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
            log.info("Total: %d events" % (
                signal_train.shape[0] +
                background_train.shape[0]))

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
                plt.savefig(os.path.join(PLOTS_DIR, 'train_var_%s_%s%s.png' % (
                    self.category.name, branch, self.output_suffix)))

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
            plt.savefig(os.path.join(PLOTS_DIR, 'train_sample_weight_%s%s.png' % (
                self.category.name, self.output_suffix)))

            if partition_idx == 0:

                # grid search params
                min_leaf_high = int((sample_train.shape[0] / 2.) *
                        (cv_nfold - 1.) / cv_nfold)
                min_leaf_low = max(10, int(min_leaf_high / 100.))

                if quick:
                    # quick search for testing
                    min_leaf_low = max(10, int(min_leaf_high / 30.))
                    min_leaf_step = max((min_leaf_high - min_leaf_low) / 10, 1)
                    MAX_N_ESTIMATORS = 300
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
                        learning_rate=.5,
                        algorithm='SAMME.R')

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
                log.info("using a %d-fold cross validation" % cv_nfold)
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
                    name=self.category.name + self.output_suffix + "_%d" % partition_idx)

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

            clf_filename = os.path.join(CACHE_DIR, 'classify',
                    'clf_%s%s_%d.pickle' % (
                    self.category.name, self.output_suffix, partition_idx))

            with open(clf_filename, 'w') as f:
                pickle.dump(clf, f)

            print_feature_ranking(clf, self.fields)

            self.clfs[(partition_idx + 1) % 2] = clf

    def classify(self, sample, category, region,
                 cuts=None, systematic='NOMINAL'):

        if self.clfs == None:
            raise RuntimeError("you must train the classifiers first")

        assert(category == self.category)

        left, right = sample.partitioned_records(
                category=category,
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

    def evaluate(self,
                 backgrounds,
                 data,
                 mass_regions,
                 systematics=False,
                 signal_scale=50,
                 unblind=False,
                 bins=20,
                 limitbins=10,
                 limitbinning='flat',
                 quick=False):

        control_region = mass_regions.control_region
        signal_region = mass_regions.signal_region
        train_region = mass_regions.train_region

        year = backgrounds[0].year

        from .samples import Higgs
        signals = []
        for mode in Higgs.MODES:
            signals.append(Higgs(
                year=year,
                mode=mode,
                mass=125,
                systematics=systematics))

        # show 2D plots of all input variables and with BDT output
        log.info("drawing scatter plots of input variables")
        draw_scatter(self.all_fields,
                     self.category,
                     self.region,
                     self.output_suffix,
                     backgrounds,
                     data=data,
                     signals=signals,
                     signal_scale=300.,
                     classifier=self)

        ########################################################################
        # show the background model and 125 GeV signal in the signal region
        log.info("plotting classifier output in the signal region ...")

        # determine min and max scores
        min_score = float('inf')
        max_score = float('-inf')

        # background model scores
        bkg_scores = []
        for bkg in backgrounds:
            scores_dict = bkg.scores(self,
                    category=self.category,
                    region=self.region,
                    cuts=signal_region)

            for sys_term, (scores, weights) in scores_dict.items():
                assert len(scores) == len(weights)
                if len(scores) == 0:
                    continue
                _min = np.min(scores)
                _max = np.max(scores)
                if _min < min_score:
                    min_score = _min
                if _max > max_score:
                    max_score = _max

            bkg_scores.append((bkg, scores_dict))

        sig_scores = []
        # signal scores
        for sig in signals:
            scores_dict = sig.scores(self,
                    category=self.category,
                    region=self.region,
                    cuts=signal_region)

            for sys_term, (scores, weights) in scores_dict.items():
                assert len(scores) == len(weights)
                if len(scores) == 0:
                    continue
                _min = np.min(scores)
                _max = np.max(scores)
                if _min < min_score:
                    min_score = _min
                if _max > max_score:
                    max_score = _max

            sig_scores.append((sig, scores_dict))

        log.info("minimum score: %f" % min_score)
        log.info("maximum score: %f" % max_score)

        # prevent bin threshold effects
        min_score -= 0.00001
        max_score += 0.00001

        # add a bin above max score and below min score for extra beauty
        score_width = max_score - min_score
        bin_width = score_width / bins
        min_score -= bin_width
        max_score += bin_width

        plot_clf(
            background_scores=bkg_scores,
            category=self.category,
            plot_label='Mass Signal Region',
            signal_scores=sig_scores,
            signal_scale=signal_scale,
            draw_data=True,
            name='signal_region' + self.output_suffix,
            bins=bins + 2,
            min_score=min_score,
            max_score=max_score,
            systematics=SYSTEMATICS.values() if systematics else None)

        # plot using the binning used for limit setting
        limit_binning_hist_template = get_safe_template(
                limitbinning, limitbins, bkg_scores, sig_scores)

        plot_clf(
            background_scores=bkg_scores,
            category=self.category,
            plot_label='Mass Signal Region',
            signal_scores=sig_scores,
            signal_scale=signal_scale,
            draw_data=True,
            name='signal_region_%s%s' % (limitbinning, self.output_suffix),
            hist_template=limit_binning_hist_template,
            systematics=SYSTEMATICS.values() if systematics else None)

        ############################################################
        # show the background model and data in the control region
        log.info("plotting classifier output in control region ...")
        log.info(control_region)
        # data scores
        data_scores, _ = data.scores(self,
                category=self.category,
                region=self.region,
                cuts=control_region)

        # determine min and max scores
        min_score = 1e10
        max_score = -1e10
        _min = data_scores.min()
        _max = data_scores.max()
        if _min < min_score:
            min_score = _min
        if _max > max_score:
            max_score = _max

        # background model scores
        bkg_scores = []
        for bkg in backgrounds:
            scores_dict = bkg.scores(self,
                    category=self.category,
                    region=self.region,
                    cuts=control_region)

            for sys_term, (scores, weights) in scores_dict.items():
                assert len(scores) == len(weights)
                if len(scores) == 0:
                    continue
                _min = np.min(scores)
                _max = np.max(scores)
                if _min < min_score:
                    min_score = _min
                if _max > max_score:
                    max_score = _max

            bkg_scores.append((bkg, scores_dict))

        log.info("minimum score: %f" % min_score)
        log.info("maximum score: %f" % max_score)

        # prevent bin threshold effects
        min_score -= 0.00001
        max_score += 0.00001

        # add a bin above max score and below min score for extra beauty
        score_width = max_score - min_score
        bin_width = score_width / bins
        min_score -= bin_width
        max_score += bin_width

        plot_clf(
            background_scores=bkg_scores,
            category=self.category,
            plot_label='Mass Control Region',
            signal_scores=None,
            data_scores=(data, data_scores),
            draw_data=True,
            name='control' + self.output_suffix,
            bins=bins + 2,
            min_score=min_score,
            max_score=max_score,
            systematics=SYSTEMATICS.values() if systematics else None)

        #return bkg_scores, sig_scores_125


def purity_score(bdt, X):

    norm = 0.
    total = None
    for w, est in zip(bdt.estimator_weights_, bdt.estimators_):
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


def histogram_scores(hist_template, scores):

    hist_template = hist_template.Clone()
    hist_template.Reset()

    if isinstance(scores, tuple):
        # data
        scores, weight = scores
        assert (weight == 1).all()
        hist = hist_template.Clone()
        hist.fill_array(scores)
    elif isinstance(scores, dict):
        # non-data with possible systematics
        # nominal case:
        nom_scores, nom_weight = scores['NOMINAL']
        hist = hist_template.Clone()
        hist.fill_array(nom_scores, nom_weight)
        # systematics
        sys_hists = {}
        for sys_term, (sys_scores, sys_weights) in scores.items():
            if sys_term == 'NOMINAL':
                continue
            sys_hist = hist_template.Clone()
            sys_hist.fill_array(sys_scores, sys_weights)
            sys_hists[sys_term] = sys_hist
        hist.systematics = sys_hists
    else:
        raise TypeError("scores not a tuple or dict")
    return hist


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

