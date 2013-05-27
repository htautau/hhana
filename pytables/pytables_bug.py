from tables.tests import print_versions

print_versions()

from mva.samples import Data
from mva.categories import Category_VBF
import re

d = Data(2012)
s = '(col_164>2.0)&(col_4>50000)&(col_47>30000)&(col_83>35000)&(col_99>25000)&(col_59>20000)&(col_3>60)&(0.8<col_134)&(col_134<2.8)&(col_141)&(col_45<1.5)&((col_115*col_98)==-1)&col_124'

print d.h5data.get_where_list(s).shape
print d.h5data.get_where_list(s, start=0, step=2).shape
print d.h5data.get_where_list(s, start=1, step=2).shape

# get full table as recarray
rec = d.h5data.read()

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
print s
print new_s
rec.dtype.names = new_names

used_cols = sorted(used_cols.keys())

rec = rec[used_cols]

# copy the table to a new file
import tables
filters = tables.Filters(complib='lzo',
                         complevel=5)
h5f_out = tables.openFile('test.h5', 'w', filters=filters)
ot = h5f_out.createTable(
        h5f_out.root, 'test',
        rec, 'test')

print ot.get_where_list(new_s).shape
print ot.get_where_list(new_s, start=0, step=2).shape
print ot.get_where_list(new_s, start=1, step=2).shape

h5f_out.close()
