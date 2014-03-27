"""
def transform(x, c=1.):
    if c == 0:
        return x
    return 2.0 / (1.0 + np.exp(- c * x)) - 1.0

b = transform(np.random.normal(-0.2, .2, 5000), 7)
s = transform(np.random.normal(0.2, .2, 100), 7)

def optimize():
    # initial binning defines one bin from min to max value
    binning = [min_x, max_x]
    binning_min = binning[:]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for nbins in xrange(8):
        left, right = binning[0], binning[1]
        edges = np.linspace(left, right, 100, endpoint=False)[1:]
        sigs = [get_sig(s, b, binning, x) for x in edges]
        ax.plot(edges, sigs)
        best_edge = edges[np.argmax(sigs)]
        binning.insert(1, best_edge)

        def func(x):
            return -get_sig(s, b, binning_min, x)

        edge, _ = search(func, binning_min[0], binning_min[1])
        binning_min.insert(1, edge)

    fig.savefig('test.png')

    canvas = Canvas()
    signal = Hist(binning, color='red', drawstyle='hist')
    background = Hist(binning, color='blue', drawstyle='hist')
    fill_hist(signal, *s)
    fill_hist(background, *b)
    signal.Draw()
    background.Draw('same')
    canvas.SaveAs('test2.png')

    print binning
    print binning_min

    print get_sig(s, b, binning)
    print get_sig(s, b, binning_min)


def get_best_bin(s, b, left, right, iter=50):
    probe_edges = np.linspace(left, right, iter, endpoint=False)[1:]
    sigs = [get_sig(s, b, [x, right]) for x in probe_edges]
    best_sig = np.max(sigs[10:])
    best_edge = probe_edges[np.argmax(sigs[10:]) + 10]
    return probe_edges, sigs, best_edge, best_sig
"""

"""
if False:
    _, _, best_edge, best_sig = get_best_edge(s, b, [min_x, max_x])

    right = max_x
    all_edges = [min_x, max_x]
    for nbins in xrange(10):
        _, _, edge, sig = get_best_bin(s, b, min_x, right)
        right = edge
        all_edges.insert(1, edge)
        sig_total = get_sig(s, b, all_edges)
        print sig, sig_total, max_fixed_nbins, max_fixed_sig
        print best_edge, best_sig
        print all_edges
        raw_input()


if False:
    edges, sigs, best_edge, best_sig = get_best_edge(s, b, [min_x, max_x])

    for nbins in range(2, 301):
        _edges = list(np.linspace(min_x, best_edge, nbins)) + [max_x]
        _sig = get_sig(s, b, _edges)

        #edges = [min_x] + list(np.linspace(best_edge, max_x, nbins))
        #sig = get_sig(s, b, edges)

        edges = np.linspace(min_x, max_x, nbins)
        sig = get_sig(s, b, edges)
        print _sig
        print _edges

        print sig
        print edges
        raw_input()
"""
