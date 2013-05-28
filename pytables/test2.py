from tables.tests import print_versions
import tables
print_versions()
f = tables.openFile('test.h5')
f.root.test.readWhere('(((((~(((col_004>30000)&(~((col_045>100)&(~(((col_178>2.0)&((col_004>50000)&(col_049>30000)))&(col_045>40))))))&(~(((col_178>2.0)&((col_004>50000)&(col_049>30000)))&(col_045>40)))))&(~((col_045>100)&(~(((col_178>2.0)&((col_004>50000)&(col_049>30000)))&(col_045>40))))))&(~(((col_178>2.0)&((col_004>50000)&(col_049>30000)))&(col_045>40))))&(((((((col_085>35000)&(col_104>25000))&(col_061>20000))&(col_195>0))&((0.8<col_144)&(col_144<2.8)))&(col_153))&(col_089|(col_016<1.570796))))&(col_047<1.5))&(((col_123*col_103)!=-1)&(col_133))',
    start=0, step=2)
