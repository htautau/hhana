import rootpy
rootpy.log.basic_config_colorized()

from mva.stats.pvalue_plot import pvalue_plot


def test_pvalue_plot():
    mass_points = [100,105,120,125,130,135,140,145,150]
    exp_list = {}
    exp_list[100] = 0.5
    exp_list[105] = 0.25
    exp_list[110] = 0.15
    exp_list[115] = 0.05
    exp_list[120] = 0.03
    exp_list[125] = 0.01
    exp_list[130] = 0.03
    exp_list[135] = 0.05
    exp_list[140] = 0.15
    exp_list[145] = 0.25
    exp_list[150] = 0.5
    
    pvalue_plot(mass_points,exp_list)
#     c.Print('toto.eps')

test_pvalue_plot()
