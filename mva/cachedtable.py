from . import log; log = log[__name__]
from tables import Table
import numpy as np
from functools import partial
import os


def nothing(f):
    return f


class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
            log.debug("using cached table")
        except KeyError:
            log.debug("table not cached (first read)")
            res = cache[key] = self.func(*args, **kw)
        return np.copy(res)


if os.getenv('NOCACHE', None):
    memoize_or_nothing = nothing
    log.warning("table cache is disabled")
else:
    memoize_or_nothing = memoize


class CachedTable(Table):

    @classmethod
    def hook(cls, thing):
        thing.__class__ = CachedTable
        return thing

    @memoize_or_nothing
    def readWhere(self, *args, **kwargs):
        return Table.readWhere(self, *args, **kwargs)

    @memoize_or_nothing
    def read_where(self, *args, **kwargs):
        log.debug(str(args))
        return Table.read_where(self, *args, **kwargs)
