import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rootpy.plotting.root2matplotlib as rplt
from rootpy.plotting import Hist

x = np.linspace(0,2*np.pi,100)

plt.figure(figsize=(8, 8), dpi=100)
#plt.fill(x,np.sin(x),color='blue',alpha=0.5)
plt.fill(x,np.sin(x),color='None', edgecolor='blue', hatch='///')
plt.fill(x,np.sin(2 * x),color='None',linewidth=0,hatch=r'\\', zorder=950)

a = Hist(5, x[0], x[-1])
b = a.Clone()
b.Fill(3, .5)

rplt.fill_between(a, b,
       edgecolor='black',
        linewidth=0,
        facecolor=(0,0,0,0),
        hatch=r'\\\\\\\\',
        zorder=900,
        )
plt.savefig('./test.eps')
plt.savefig('./test.pdf')
