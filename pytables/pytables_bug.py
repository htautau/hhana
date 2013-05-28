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

print ot.get_where_list(new_s).shape
print ot.get_where_list(new_s, start=0, step=2).shape
print ot.get_where_list(new_s, start=1, step=2).shape

h5f_out.close()
