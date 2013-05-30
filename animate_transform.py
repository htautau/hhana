import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-1, 1))

text = ax.text(.5, .8, '',
        fontsize=30,
        transform = ax.transAxes,
        horizontalalignment='center',
        verticalalignment='center')

bkg_scores = np.random.normal(-.3, .2, size=10000)
sig_scores = np.random.normal(.3, .2, size=10000)

bins = 50
_, _, left_patches = ax.hist(bkg_scores, range=(-1, 1),
        bins=bins,
        alpha=.6,
        facecolor='blue',
        normed=True)
_, _, right_patches = ax.hist(sig_scores, range=(-1, 1),
        bins=bins,
        alpha=.6,
        facecolor='red',
        normed=True)

def transform(x, c):
    return 2.0 / (1.0 + np.exp(- c * x)) - 1.0

# initialization function: plot the background of each frame
def init():
    #line.set_data([], [])
    #return line,
    return left_patches, right_patches

# animation function.  This is called sequentially
def animate(i):
    new_bkg, _ = np.histogram(transform(bkg_scores, i / 20.), bins=bins,
            range=(-1, 1),
            normed=True)
    new_sig, _ = np.histogram(transform(sig_scores, i / 20.), bins=bins,
            range=(-1, 1),
            normed=True)

    text.set_text(r'$T(x) = \frac{2}{1 + e^{-%.2f x}} - 1$' % (i / 20.))

    for patch, v in zip(left_patches, new_bkg):
        patch.set_height(v)

    for patch, v in zip(right_patches, new_sig):
        patch.set_height(v)

    ax.set_ylim(0, max(new_bkg.max(), new_sig.max()) * 1.3)
    return left_patches, right_patches

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=10, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('animate_transform.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
