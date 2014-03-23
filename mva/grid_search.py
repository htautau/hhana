"""
The :mod:`sklearn.grid_search` includes utilities to fine-tune the parameters
of an estimator.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD Style.

import time
from copy import copy
import operator

import numpy as np

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.cross_validation import check_cv, _safe_split
from sklearn.utils.validation import _num_samples, check_arrays
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.utils import safe_mask
from sklearn.grid_search import GridSearchCV, ParameterGrid, _CVScoreTuple
from sklearn.metrics.scorer import check_scoring

__all__ = ['BoostGridSearchCV',]


def fit_grid_point(base_estimator, parameters,
                   X, y, sample_weight,
                   train, test, verbose,
                   **fit_params):
    """Run fit on one set of parameters"""
    if verbose > 1:
        start_time = time.time()
        msg = '%s' % (', '.join('%s=%s' % (k, v)
                                     for k, v in parameters.iteritems()))
        print "[BoostGridSearchCV] %s %s" % (msg, (64 - len(msg)) * '.')

    # update parameters of the classifier after a copy of its base structure
    estimator = clone(base_estimator)
    estimator.set_params(**parameters)

    X_train, y_train, sample_weight_train = _safe_split(
        estimator, X, y, sample_weight, train)
    X_test, y_test, sample_weight_test = _safe_split(
        estimator, X, y, sample_weight, test, train)

    if sample_weight is not None:
        fit_params = fit_params.copy()
        fit_params['sample_weight'] = sample_weight_train

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)

    if verbose > 1:
        end_msg = "%s -%s" % (msg,
                              logger.short_format_time(time.time() -
                                                       start_time))
        print "[BoostGridSearchCV] %s %s" % ((64 - len(end_msg)) * '.', end_msg)
    return estimator, parameters, train, test


def score_each_boost(estimator, parameters,
                     min_n_estimators,
                     X, y, sample_weight,
                     score_func, train, test,
                     verbose):
    """Run fit on one set of parameters

    Returns the score and the instance of the classifier
    """
    if verbose > 1:
        start_time = time.time()
        msg = '%s' % (', '.join('%s=%s' % (k, v)
                                     for k, v in parameters.iteritems()))
        print "[BoostGridSearchCV] %s %s" % (msg, (64 - len(msg)) * '.')

    X_test, y_test, sample_weight_test = _safe_split(
        estimator, X, y, sample_weight, test, train)

    test_score_params = {}
    if sample_weight is not None:
        test_score_params['sample_weight'] = sample_weight_test

    this_n_test_samples = _num_samples(X_test)

    all_scores = []
    all_clf_params = []
    n_test_samples = []

    for i, y_pred in enumerate(estimator.staged_predict(X_test)):
        if i + 1 < min_n_estimators:
            continue
        score = score_func(y_test, y_pred, **test_score_params)
        all_scores.append(score)
        clf_para = copy(parameters)
        clf_para['n_estimators'] = i + 1
        all_clf_params.append(clf_para)
        n_test_samples.append(this_n_test_samples)

    # boosting may have stopped early
    if len(all_scores) < estimator.n_estimators - min_n_estimators + 1:
        last_score = all_scores[-1]
        last_clf_params = all_clf_params[-1]
        for i in range(len(all_scores),
                       estimator.n_estimators - min_n_estimators + 1):
            all_scores.append(last_score)
            clf_para = copy(last_clf_params)
            clf_para['n_estimators'] = i + 1
            all_clf_params.append(clf_para)
            n_test_samples.append(this_n_test_samples)

    if verbose > 1:
        end_msg = "%s -%s" % (msg,
                              logger.short_format_time(time.time() -
                                                       start_time))
        print "[BoostGridSearchCV] %s %s" % ((64 - len(end_msg)) * '.', end_msg)
    return all_scores, all_clf_params, n_test_samples


class BoostGridSearchCV(GridSearchCV):

    def __init__(self, estimator, param_grid,
            max_n_estimators,
            min_n_estimators=1,
            **kwargs):

        if 'n_estimators' in param_grid:
            raise ValueError(
                'do not include n_estimators in param_grid when '
                'using BoostGridSearchCV')
        if min_n_estimators < 1 or min_n_estimators >= max_n_estimators:
            raise ValueError(
                'min_n_estimators must be 1 or greater and less than '
                'max_n_estimators')
        self.max_n_estimators = max_n_estimators
        self.min_n_estimators = min_n_estimators
        super(BoostGridSearchCV, self).__init__(
            estimator=estimator,
            param_grid=param_grid,
            **kwargs)

    def _fit(self, X, y, sample_weight, parameter_iterable):
        """Actual fitting, performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv

        n_samples = _num_samples(X)
        X, y, sample_weight = check_arrays(X, y, sample_weight,
                                           allow_lists=True,
                                           sparse_format='csr')

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
            y = np.asarray(y)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)

        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)

        # first fit at each grid point using the maximum n_estimators
        param_grid = self.param_grid.copy()
        param_grid['n_estimators'] = [self.max_n_estimators]
        grid = ParameterGrid(param_grid)

        pre_dispatch = self.pre_dispatch

        clfs = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(
            delayed(fit_grid_point)(base_estimator, clf_params,
                                    X, y, sample_weight,
                                    train, test,
                                    self.verbose, **self.fit_params)
            for clf_params in grid
            for train, test in cv)

        # now use the already fitted ensembles but trancate to N estimators for
        # N from 1 to n_estimators_max - 1 (inclusive)
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(
            delayed(score_each_boost)(clf, clf_params,
                                      self.min_n_estimators,
                                      X, y, sample_weight,
                                      self.score_func,
                                      train, test,
                                      self.verbose)
            for clf, clf_params, train, test in clfs)

        out = reduce(operator.add, [zip(*stage) for stage in out])
        # out is now a list of triplet: score, estimator_params, n_test_samples

        n_estimators_points = self.max_n_estimators - self.min_n_estimators + 1
        n_fits = len(out)
        n_folds = len(cv)

        grid_scores = list()
        for block in range(0, n_fits, n_folds * n_estimators_points):
            for grid_start in range(block, block + n_estimators_points):
                n_test_samples = 0
                score = 0
                all_scores = list()
                for this_score, parameters, this_n_test_samples in \
                        out[grid_start:
                            grid_start + n_folds * n_estimators_points:
                            n_estimators_points]:
                    all_scores.append(this_score)
                    if self.iid:
                        this_score *= this_n_test_samples
                    score += this_score
                    n_test_samples += this_n_test_samples
                if self.iid:
                    score /= float(n_test_samples)
                else:
                    score /= float(n_folds)
                grid_scores.append(_CVScoreTuple(
                    parameters,
                    score,
                    np.array(all_scores)))

        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            fit_params = self.fit_params
            if sample_weight is not None:
                fit_params = fit_params.copy()
                fit_params['sample_weight'] = sample_weight
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            if y is not None:
                best_estimator.fit(X, y, **fit_params)
            else:
                best_estimator.fit(X, **fit_params)
            self.best_estimator_ = best_estimator
        return self
