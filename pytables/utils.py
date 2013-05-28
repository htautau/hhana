import re
import tables


def obfuscate(rec, s, only_used_cols=False):
    # rename columns
    curr_names = rec.dtype.names
    new_names = ['col_%03d' % d for d in xrange(len(curr_names))]
    new_s = s
    used_cols = {}
    for curr_name, new_name in zip(curr_names, new_names):
        def repl(match):
            used_cols[new_name] = None
            return match.group('left') + new_name + match.group('right')
        new_s = re.sub('(?P<left>(^|\W))(?P<var>%s)(?P<right>(\W|$))' % curr_name, repl, new_s)
    rec.dtype.names = new_names
    if only_used_cols:
        rec = rec[sorted(used_cols.keys())]
    return rec, new_s


def write(rec, outname='test.h5'):
    filters = tables.Filters(complib='lzo',
                             complevel=5)
    h5f_out = tables.openFile(outname, 'w', filters=filters)
    ot = h5f_out.createTable(
            h5f_out.root, 'test',
            rec, 'test')
    h5f_out.close()
