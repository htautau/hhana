from tables.tests import print_versions
import tables

print_versions()

f = tables.openFile('test.h5', 'r')
t = f.root.test
s = '((((col_164>2.0)&((col_4>50000)&(col_47>30000)))&((((((col_83>35000)&(col_99>25000))&(col_59>20000))&(col_3>60))&((0.8<col_134)&(col_134<2.8)))&(col_141)))&(col_45<1.5))&(((col_115*col_98)==-1)&(col_124))'

print t.get_where_list(s).shape
print t.get_where_list(s, start=0, step=2).shape
print t.get_where_list(s, start=1, step=2).shape
