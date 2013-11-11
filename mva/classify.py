import os
import pickle
from operator import itemgetter
import types
import shutil
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

# scikit-learn imports
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from rootpy.plotting import Hist
from rootpy.io import root_open
from rootpy.extern.tabulartext import PrettyTable

from cStringIO import StringIO

from .samples import *
from . import log; log = log[__name__]
from . import CACHE_DIR
from . import MMC_MASS, MMC_PT
from .plotting import (
    draw, plot_clf, plot_grid_scores,
    hist_scores, draw_samples_array,
    draw_channel_array, draw_channel,
    efficiency_cut)
from . import variables
from . import PLOTS_DIR
from .stats.utils import get_safe_template
from .np_utils import rec_to_ndarray, std
from .systematics import systematic_name
from .grid_search import BoostGridSearchCV


def print_feature_ranking(clf, fields):

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        log.info("Feature ranking:")
        out = StringIO()
        print >> out
        print >> out
        print >> out, r"\begin{tabular}{c|c|c}"
        table = PrettyTable(["Rank", "Variable", "Importance"])
        print >> out, r"\hline\hline"
        print >> out, r"Rank & Variable & Importance\\"
        for f, idx in enumerate(indices):
            table.add_row([f + 1,
                fields[idx],
                '%.3f' % importances[idx]])
            print >> out, r"%d & %s & %.3f\\" % (f + 1,
                variables.VARIABLES[fields[idx]]['title'],
                importances[idx])
        print >> out, r"\end{tabular}"
        print >> out
        print >> out, table.get_string(hrules=1)
        log.info(out.getvalue())


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


class Classifier(object):

    # minimal list of spectators
    SPECTATORS = [
        MMC_PT,
        MMC_MASS,
    ]

    def __init__(self,
                 fields,
                 category,
                 region,
                 cuts=None,
                 spectators=None,
                 standardize=False,
                 output_suffix="",
                 clf_output_suffix="",
                 partition_key=None,
                 transform=True,
                 mmc=True):

        fields = fields[:]
        if not mmc:
            try:
                fields.remove(MMC_MASS)
            except ValueError:
                pass

        self.fields = fields
        self.category = category
        self.region = region
        self.spectators = spectators
        self.standardize = standardize
        self.output_suffix = output_suffix
        self.clf_output_suffix = clf_output_suffix
        self.partition_key = partition_key
        self.transform = transform
        self.mmc = mmc
        self.background_label = 0
        self.signal_label = 1

        if spectators is None:
            spectators = []

        # merge in minimal list of spectators
        for spec in Classifier.SPECTATORS:
            if spec not in spectators and spec not in fields:
                spectators.append(spec)

        self.all_fields = fields + spectators

        assert 'weight' not in fields

        # classifiers for the left and right partitions
        # each trained on the opposite partition
        self.clfs = None

    def load(self):

        use_cache = True
        # attempt to load existing classifiers
        clfs = [None, None]
        for partition_idx in range(2):

            category_name = self.category.get_parent().name
            clf_filename = os.path.join(CACHE_DIR, 'classify',
                    'clf_%s%s_%d.pickle' % (
                    category_name, self.clf_output_suffix, partition_idx))

            log.info("attempting to open %s ..." % clf_filename)
            if os.path.isfile(clf_filename):
                # use a previously trained classifier
                log.info("found existing classifier in %s" % clf_filename)
                with open(clf_filename, 'r') as f:
                    clf = pickle.load(f)
                out = StringIO()
                print >> out
                print >> out
                print >> out, clf
                log.info(out.getvalue())
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
            return True
        else:
            log.warning(
                "unable to load previously trained "
                "classifiers; train new ones")
            return False

    def train(self,
              signals,
              backgrounds,
              cuts=None,
              max_sig=None,
              max_bkg=None,
              norm_sig_to_bkg=True,
              same_size_sig_bkg=True, # NOTE: if True this crops signal a lot!!
              remove_negative_weights=False,
              grid_search=True,
              cv_nfold=5,
              use_cache=True,
              **clf_params):
        """
        Determine best BDTs on left and right partitions. Each BDT will then be
        used on the other partition.
        """
        if use_cache and not self.clfs:
            if self.load():
                return

        signal_recs = []
        signal_arrs = []
        signal_weight_arrs = []

        for signal in signals:
            left, right = signal.partitioned_records(
                category=self.category,
                region=self.region,
                fields=self.all_fields,
                cuts=cuts,
                key=self.partition_key)
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
                cuts=cuts,
                key=self.partition_key)
            background_weight_arrs.append(
               (left['weight'], right['weight']))
            background_arrs.append(
               (rec_to_ndarray(left, self.fields),
                rec_to_ndarray(right, self.fields)))
            background_recs.append((left, right))

        self.clfs = [None, None]

        for partition_idx in range(2):

            clf_filename = os.path.join(CACHE_DIR, 'classify',
                'clf_%s%s_%d' % (
                self.category.name, self.clf_output_suffix, partition_idx))

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

            #log.info("plotting input variables as they are given to the BDT")
            ## draw plots of the input variables
            #for i, branch in enumerate(self.fields):
            #    log.info("plotting %s ..." % branch)
            #    branch_data = sample_train[:,i]
            #    if 'scale' in variables.VARIABLES[branch]:
            #        branch_data *= variables.VARIABLES[branch]['scale']
            #    _min, _max = branch_data.min(), branch_data.max()
            #    plt.figure()
            #    plt.hist(branch_data[labels_train==0],
            #            bins=20, range=(_min, _max),
            #            weights=sample_weight_train[labels_train==0],
            #            label='Background', histtype='stepfilled',
            #            alpha=.5)
            #    plt.hist(branch_data[labels_train==1],
            #            bins=20, range=(_min, _max),
            #            weights=sample_weight_train[labels_train==1],
            #            label='Signal', histtype='stepfilled', alpha=.5)
            #    label = variables.VARIABLES[branch]['title']
            #    if 'units' in variables.VARIABLES[branch]:
            #        label += ' [%s]' % variables.VARIABLES[branch]['units']
            #    plt.xlabel(label)
            #    plt.legend()
            #    plt.savefig(os.path.join(PLOTS_DIR, 'train_var_%s_%s%s.png' % (
            #        self.category.name, branch, self.output_suffix)))

            #log.info("plotting sample weights ...")
            #_min, _max = sample_weight_train.min(), sample_weight_train.max()
            #plt.figure()
            #plt.hist(sample_weight_train[labels_train==0],
            #        bins=20, range=(_min, _max),
            #        label='Background', histtype='stepfilled',
            #        alpha=.5)
            #plt.hist(sample_weight_train[labels_train==1],
            #        bins=20, range=(_min, _max),
            #        label='Signal', histtype='stepfilled', alpha=.5)
            #plt.xlabel('sample weight')
            #plt.legend()
            #plt.savefig(os.path.join(PLOTS_DIR, 'train_sample_weight_%s%s.png' % (
            #    self.category.name, self.output_suffix)))

            if partition_idx == 0:

                # grid search params
                min_leaf_high = int((sample_train.shape[0] / 8) *
                    (cv_nfold - 1.) / cv_nfold)
                min_leaf_low = max(10, int(min_leaf_high / 100.))

                min_leaf_step = max((min_leaf_high - min_leaf_low) / 50, 1)
                max_n_estimators = 200
                min_n_estimators = 1
                n_estimators_step = 50

                min_samples_leaf = range(
                    min_leaf_low, min_leaf_high, min_leaf_step)

                #n_estimators = range(
                #    min_n_estimators, max_n_estimators, n_estimators_step)

                n_estimators = np.power(2, np.arange(0, 8))

                grid_params = {
                    'base_estimator__min_samples_leaf': min_samples_leaf,
                    #'n_estimators': n_estimators
                }

                #AdaBoostClassifier.staged_score = staged_score

                clf = AdaBoostClassifier(
                    DecisionTreeClassifier(),
                    learning_rate=.1,
                    algorithm='SAMME.R',
                    random_state=0)

                grid_clf = BoostGridSearchCV(
                    clf, grid_params,
                    max_n_estimators=max_n_estimators,
                    min_n_estimators=min_n_estimators,
                    #n_estimators_step=1,
                    # can use default ClassifierMixin score
                    #score_func=precision_score,
                    cv = StratifiedKFold(labels_train, cv_nfold),
                    n_jobs=20)

                #grid_clf = GridSearchCV(
                #    clf, grid_params,
                #    # can use default ClassifierMixin score
                #    #score_func=precision_score,
                #    cv = StratifiedKFold(labels_train, cv_nfold),
                #    n_jobs=20)

                log.info("")
                log.info("using a %d-fold cross validation" % cv_nfold)
                log.info("performing a grid search over these parameter values:")
                for param, values in grid_params.items():
                    log.info('{0} {1}'.format(param.split('__')[-1], values))
                log.info("Minimum number of classifiers: %d" % min_n_estimators)
                log.info("Maximum number of classifiers: %d" % max_n_estimators)
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

                log.info("After scaling up min_leaf")
                out = StringIO()
                print >> out
                print >> out
                print >> out, clf
                log.info(out.getvalue())

            else: # training on the other partition
                log.info("training a new classifier ...")

                # use same params as in first partition
                clf = sklearn.clone(clf)
                out = StringIO()
                print >> out
                print >> out
                print >> out, clf
                log.info(out.getvalue())

                clf.fit(sample_train, labels_train,
                        sample_weight=sample_weight_train)

            if isinstance(clf, AdaBoostClassifier):
                # export to graphviz dot format
                if os.path.isdir(clf_filename):
                    shutil.rmtree(clf_filename)
                os.mkdir(clf_filename)

                for itree, tree in enumerate(clf):
                    export_graphviz(tree,
                        out_file=os.path.join(
                            clf_filename,
                            'tree_{0:d}.dot'.format(itree)),
                        feature_names=self.all_fields)

            with open('{0}.pickle'.format(clf_filename), 'w') as f:
                pickle.dump(clf, f)

            print_feature_ranking(clf, self.fields)

            self.clfs[(partition_idx + 1) % 2] = clf

    def classify(self, sample, category, region,
                 cuts=None, systematic='NOMINAL'):

        if self.clfs == None:
            raise RuntimeError("you must train the classifiers first")

        partitions = sample.partitioned_records(
            category=category,
            region=region,
            fields=self.fields,
            cuts=cuts,
            systematic=systematic,
            num_partitions=2,
            return_idx=True,
            key=self.partition_key)

        score_idx = [[], []]
        for i, partition in enumerate(partitions):
            for rec, idx in partition:
                weight = rec['weight']
                arr = rec_to_ndarray(rec, self.fields)
                # each classifier is never used on the partition that trained it
                scores = self.clfs[i].decision_function(arr)
                score_idx[i].append((idx, scores, weight))

        # must preserve order of scores wrt the other fields!
        # merge the scores and weights according to the idx
        merged_scores = []
        merged_weight = []
        for left, right in zip(*score_idx):
            left_idx, left_scores, left_weight = left
            right_idx, right_scores, right_weight = right
            insert_idx = np.searchsorted(left_idx, right_idx)
            scores = np.insert(left_scores, insert_idx, right_scores)
            weight = np.insert(left_weight, insert_idx, right_weight)
            merged_scores.append(scores)
            merged_weight.append(weight)

        scores = np.concatenate(merged_scores)
        weight = np.concatenate(merged_weight)

        if self.transform:
            log.info("classifier scores are transformed")
            # logistic tranformation used by TMVA (MethodBDT.cxx)
            if isinstance(self.transform, types.FunctionType):
                # user-defined transformation
                scores = self.transform(scores)
            else:
                # default logistic transformation
                scores = -1 + 2.0 / (1.0 +
                    np.exp(-self.clfs[0].n_estimators * scores / 10))

        return scores, weight

    def evaluate(self,
                 analysis,
                 signal_region,
                 control_region,
                 systematics=None,
                 signal_scale=50,
                 unblind=False,
                 mpl=False,
                 fit=None,
                 output_formats=None):

        category = self.category
        region = self.region

        ##########################################################
        # show the background model and data in the control region
        log.info("plotting classifier output in control region ...")
        log.info(control_region)

        _, channel = analysis.clf_channels(self,
            category, region, cuts=control_region,
            bins=category.clf_bins + 2,
            systematics=systematics,
            unblind=True,
            no_signal_fixes=True)

        for logy in (True, False):
            draw_channel(channel,
                category=category,
                plot_label='Mass Control Region',
                data_info=str(analysis.data.info),
                output_name='event_bdt_score_control' + self.output_suffix,
                name='BDT Score',
                systematics=systematics,
                mpl=mpl,
                output_formats=output_formats,
                signal_colour_map=cm.spring,
                plot_signal_significance=True,
                logy=logy,
                fit=fit)

        ###################################################################
        # show the background model and 125 GeV signal in the signal region
        log.info("plotting classifier output in the signal region ...")

        scores, channels = analysis.clf_channels(self,
            category, region, cuts=signal_region,
            mass_points=[125],
            systematics=systematics,
            bins=category.clf_bins + 2,
            unblind=unblind or 0.3,
            no_signal_fixes=True)

        bkg_scores = scores.bkg_scores
        sig_scores = scores.all_sig_scores[125]
        min_score = scores.min_score
        max_score = scores.max_score

        for logy in (True, False):
            draw_channel(
                channels[125],
                category=category,
                plot_label='Mass Signal Region' if signal_region else None,
                signal_scale=signal_scale if not unblind else 1.,
                data_info=str(analysis.data.info),
                output_name='event_bdt_score_signal_region' + self.output_suffix,
                name='BDT Score',
                systematics=systematics,
                mpl=mpl,
                output_formats=output_formats,
                signal_colour_map=cm.spring,
                plot_signal_significance=True,
                logy=logy,
                fit=fit)

        ###############################################################
        log.info("plotting mmc weighted by background BDT distribution")

        bkg_score_hist = Hist(category.limitbins, min_score, max_score)
        hist_scores(bkg_score_hist, bkg_scores)
        _bkg = bkg_score_hist.Clone()
        bkg_score_hist /= sum(bkg_score_hist.y())

        draw_channel_array(
            analysis,
            variables.VARIABLES,
            plots=[MMC_MASS],
            mass=125,
            mode='combined',
            signal_scale=50,
            category=category,
            region=region,
            show_qq=False,
            plot_signal_significance=False,
            systematics=systematics,
            weight_hist=bkg_score_hist,
            clf=self,
            output_suffix="_reweighted_bkg" + self.output_suffix,
            cuts=signal_region,
            mpl=mpl,
            output_formats=output_formats,
            unblind=True,
            fit=fit)

        ###############################################################
        log.info("plotting mmc weighted by signal BDT distribution")

        sig_score_hist = Hist(category.limitbins, min_score, max_score)
        hist_scores(sig_score_hist, sig_scores)
        _sig = sig_score_hist.Clone()
        sig_score_hist /= sum(sig_score_hist.y())

        draw_channel_array(
            analysis,
            variables.VARIABLES,
            plots=[MMC_MASS],
            mass=125,
            mode='combined',
            signal_scale=5,
            category=category,
            region=region,
            show_qq=False,
            plot_signal_significance=False,
            systematics=systematics,
            weight_hist=sig_score_hist,
            clf=self,
            output_suffix="_reweighted_sig" + self.output_suffix,
            cuts=signal_region,
            mpl=mpl,
            output_formats=output_formats,
            unblind=unblind,
            fit=fit)

        ###############################################################
        log.info("plotting mmc weighted by S / B")

        sob_hist = (1 + _sig / _bkg)
        _log = math.log
        for bin in sob_hist.bins(overflow=True):
            bin.value = _log(bin.value)
        log.info(str(list(sob_hist.y())))

        field_channel, figs = draw_channel_array(
            analysis,
            variables.VARIABLES,
            plots=[MMC_MASS],
            templates={MMC_MASS: Hist(30, 50, 200)},
            mass=[125, 150],
            mode='combined',
            signal_scale=1,
            stacked_signal=False,
            signal_colour_map=cm.spring,
            ylabel='ln(1+S/B) Weighted Events',
            category=category,
            region=region,
            show_qq=False,
            plot_signal_significance=False,
            systematics=systematics,
            weight_hist=sob_hist,
            clf=self,
            output_suffix="_reweighted_sob" + self.output_suffix,
            cuts=signal_region,
            mpl=mpl,
            fit=fit,
            output_formats=output_formats,
            unblind=True)
            #bootstrap_data=analysis)

        channel = field_channel[MMC_MASS]
        with root_open('sob.root', 'update') as f:
            for s in channel.samples:
                s.hist.Write()
            channel.data.hist.Write()

        ############################################################
        # show the MMC below a BDT score that unblinds 30% of signal
        # determine BDT score with 30% of 125 signal below:

        """
        signal_score_hist = Hist(1000, -1, 1)
        for s, scores_dict in sig_scores:
            histogram_scores(signal_score_hist, scores_dict, inplace=True)
        max_score = efficiency_cut(signal_score_hist, 0.3)
        log.info("plotting mmc below BDT score of %.2f" % max_score)

        draw_channel_array(
            analysis,
            variables.VARIABLES,
            plots=[MMC_MASS],
            mass=125,
            mode='combined',
            signal_scale=50,
            category=category,
            region=region,
            show_qq=False,
            plot_signal_significance=False,
            systematics=systematics,
            clf=self,
            max_score=max_score,
            output_suffix="_lowbdt" + self.output_suffix,
            cuts=signal_region,
            mpl=mpl,
            output_formats=output_formats,
            unblind=True)
        """
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


def histogram_scores(hist_template, scores,
                     min_score=None, max_score=None,
                     inplace=False):

    if not inplace:
        hist = hist_template.Clone(name=hist_template.name + "_scores")
        hist.Reset()
    else:
        hist = hist_template
    if min_score is not None:
        log.info("cutting out scores below %f" % min_score)
    if max_score is not None:
        log.info("cutting out scores above %f" % max_score)
    if isinstance(scores, np.ndarray):
        if min_score is not None:
            scores = scores[scores > min_score]
        if max_score is not None:
            scores = scores[scores < max_score]
        hist.fill_array(scores)
    elif isinstance(scores, tuple):
        # data
        scores, weight = scores
        if min_score is not None:
            scores_idx = scores > min_score
            scores = scores[scores_idx]
            weight = weight[scores_idx]
        if max_score is not None:
            scores_idx = scores < max_score
            scores = scores[scores_idx]
            weight = weight[scores_idx]
        assert (weight == 1).all()
        hist.fill_array(scores)
    elif isinstance(scores, dict):
        # non-data with possible systematics
        # nominal case:
        nom_scores, nom_weight = scores['NOMINAL']
        if min_score is not None:
            scores_idx = nom_scores > min_score
            nom_scores = nom_scores[scores_idx]
            nom_weight = nom_weight[scores_idx]
        if max_score is not None:
            scores_idx = nom_scores < max_score
            nom_scores = nom_scores[scores_idx]
            nom_weight = nom_weight[scores_idx]
        hist.fill_array(nom_scores, nom_weight)
        # systematics
        sys_hists = {}
        for sys_term, (sys_scores, sys_weight) in scores.items():
            if sys_term == 'NOMINAL':
                continue
            if min_score is not None:
                scores_idx = sys_scores > min_score
                sys_scores = sys_scores[scores_idx]
                sys_weight = sys_weight[scores_idx]
            if max_score is not None:
                scores_idx = sys_scores < max_score
                sys_scores = sys_scores[scores_idx]
                sys_weight = sys_weight[scores_idx]
            sys_hist = hist.Clone(name=hist.name + "_" + systematic_name(sys_term))
            sys_hist.Reset()
            sys_hist.fill_array(sys_scores, sys_weight)
            sys_hists[sys_term] = sys_hist
        hist.systematics = sys_hists
    else:
        raise TypeError("scores not an np.array, tuple or dict")
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
