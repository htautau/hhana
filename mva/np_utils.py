import numpy as np
import operator


def std(X):

    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


def rec_to_ndarray(rec, fields=None):

    if fields is None:
        fields = rec.dtype.names
    if len(fields) == 1:
        return rec[fields[0]]
    # Creates a copy and recasts data to a consistent datatype
    return np.vstack([rec[field] for field in fields]).T


def rec_stack(recs, fields=None):

    if fields is None:
        fields = list(reduce(operator.and_,
            [set(rec.dtype.names) for rec in recs]))
    return np.hstack([rec[fields] for rec in recs])

