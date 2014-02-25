import rootpy
rootpy.log.basic_config_colorized()

from mva.stats.pvalue_plot import pvalue_plot

def test_pvalue_plot():
    mass_points = [100,105,120,125,130,135,140,145,150]
    pvalues = [0.5, 0.25, 0.15, 0.05, 0.03, 0.01, 0.03, 0.05, 0.15, 0.25, 0.5]
    pvalue_plot(mass_points, pvalues)

test_pvalue_plot()
