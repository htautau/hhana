from . import log; log = log[__name__]

import numpy as np
from rootpy.plotting import Hist
from math import sqrt


def get_safe_template(binning, bins, bkg_scores, sig_scores):

    # TODO: use full score range, not just min and max signal score

    llog = log['get_safe_template']
    # determine min and max scores
    """
    min_score = float('inf')
    max_score = float('-inf')

    for bkg, scores_dict in bkg_scores:
        for sys_term, (scores, weights) in scores_dict.items():
            assert len(scores) == len(weights)
            if len(scores) == 0:
                continue
            _min = np.min(scores)
            _max = np.max(scores)
            if _min < min_score:
                min_score = _min
            if _max > max_score:
                max_score = _max
    """

    min_score_signal = float('inf')
    max_score_signal = float('-inf')

    for sig, scores_dict in sig_scores:
        for sys_term, (scores, weights) in scores_dict.items():
                assert len(scores) == len(weights)
                if len(scores) == 0:
                    continue
                _min = np.min(scores)
                _max = np.max(scores)
                if _min < min_score_signal:
                    min_score_signal = _min
                if _max > max_score_signal:
                    max_score_signal = _max

    llog.info("minimum signal score: %f" % min_score_signal)
    llog.info("maximum signal score: %f" % max_score_signal)

    # prevent bin threshold effects
    min_score_signal -= 0.00001
    max_score_signal += 0.00001

    if binning == 'flat':
        llog.info("binning such that background is flat")
        # determine location that maximizes signal significance
        bkg_hist = Hist(100, min_score_signal, max_score_signal)
        sig_hist = bkg_hist.Clone()

        # fill background
        for bkg_sample, scores_dict in bkg_scores:
            score, w = scores_dict['NOMINAL']
            bkg_hist.fill_array(score, w)

        # fill signal
        for sig_sample, scores_dict in sig_scores:
            score, w = scores_dict['NOMINAL']
            sig_hist.fill_array(score, w)

        # determine maximum significance
        sig, max_sig, max_cut = significance(sig_hist, bkg_hist, min_bkg=1)
        llog.info("maximum signal significance of %f at score > %f" % (
                max_sig, max_cut))

        # determine N bins below max_cut or N+1 bins over the whole signal
        # score range such that the background is flat
        # this will require a binary search for each bin boundary since the
        # events are weighted.
        """
        flat_bins = search_flat_bins(
                bkg_scores, min_score_signal, max_score_signal,
                int(sum(bkg_hist) / 20))
        """
        flat_bins = search_flat_bins(
                bkg_scores, min_score_signal, max_cut, 5)
        # one bin above max_cut
        flat_bins.append(max_score_signal)
        hist_template = Hist(flat_bins)

    elif binning == 'onebkg':
        # Define each bin such that it contains at least one background.
        # First histogram background with a very fine binning,
        # then sum from the right to the left up to a total of one
        # event. Use the left edge of that bin as the left edge of the
        # last bin in the final histogram template.
        # Important: also choose the bin edge such that all background
        # components each have at least zero events, since we have
        # samples with negative weights (SS subtraction in the QCD) and
        # MC@NLO samples.

        # TODO: perform rebinning iteratively on all bins

        llog.info("binning such that each bin has at least one background")

        default_bins = list(np.linspace(
                min_score_signal,
                max_score_signal,
                bins + 1))

        nbins = 1000
        total_bkg_hist = Hist(nbins, min_score_signal, max_score_signal)
        bkg_arrays = []
        # fill background
        for bkg_sample, scores_dict in bkg_scores:
            score, w = scores_dict['NOMINAL']
            bkg_hist = total_bkg_hist.Clone()
            bkg_hist.fill_array(score, w)
            # create array from histogram
            bkg_array = np.array(bkg_hist)
            bkg_arrays.append(bkg_array)

        edges = [max_score_signal]
        view_cutoff = nbins

        while True:
            sums = []
            # fill background
            for bkg_array in bkg_arrays:
                # reverse cumsum
                bkg_cumsum = bkg_array[:view_cutoff][::-1].cumsum()[::-1]
                sums.append(bkg_cumsum)

            total_bkg_cumsum = np.add.reduce(sums)

            # determine last element with at least a value of 1.
            # and where each background has at least zero events
            # so that no sample may have negative events in this bin
            all_positive = np.logical_and.reduce([b >= 0. for b in sums])
            #print "all positive"
            #print all_positive
            #print "total >= 1"
            #print total_bkg_cumsum >= 1.

            last_bin_one_bkg = np.where(total_bkg_cumsum >= 1.)[-1][-1]
            #print "last bin index"
            #print last_bin_one_bkg

            # bump last bin down until each background is positive
            last_bin_one_bkg -= all_positive[:last_bin_one_bkg + 1][::-1].argmax()
            #print "last bin index after correction"
            #print last_bin_one_bkg

            # get left bin edge corresponding to this bin
            bin_edge = bkg_hist.xedges(int(last_bin_one_bkg))

            # expected bin location
            bin_index_expected = int(view_cutoff - (nbins / bins))
            if (bin_index_expected <= 0 or
                bin_index_expected <= (nbins / bins)):
                llog.warning("early termination of binning")
                break
            # bump expected bin index down until each background is positive
            bin_index_expected_correct = all_positive[:bin_index_expected + 1][::-1].argmax()
            if bin_index_expected_correct > 0:
                llog.warning(
                    "expected bin index corrected such that all "
                    "backgrounds are positive")
            bin_index_expected -= bin_index_expected_correct
            if (bin_index_expected <= 0 or
                bin_index_expected <= (nbins / bins)):
                llog.warning("early termination of binning after correction")
                break
            bin_edge_expected = total_bkg_hist.xedges(int(bin_index_expected))

            # if this edge is greater than it would otherwise be if we used
            # constant-width binning over the whole range then just use the
            # original binning
            if bin_edge > bin_edge_expected:
                llog.info("expected bin edge %f is OK" % bin_edge_expected)
                bin_edge = bin_edge_expected
                view_cutoff = bin_index_expected

            else:
                llog.info("adjusting bin to contain >= one background")
                llog.info("original edge: %f  new edge: %f " %
                        (bin_edge_expected, bin_edge))
                view_cutoff = last_bin_one_bkg

            edges.append(bin_edge)

        edges.append(min_score_signal)
        #llog.info("edges %s" % str(edges))
        hist_template = Hist(edges[::-1])

    else:
        llog.info("using constant-width bins")
        hist_template = Hist(bins,
                min_score_signal, max_score_signal)
    return hist_template


def uniform_binning(hist):
    """
    For some obscure technical reason, HistFactory can't handle histograms with
    variable width bins. This function takes any 1D histogram and outputs a new
    histogram with constant width bins by using the bin indices of the input
    histogram as the x-axis of the new histogram.
    """
    new_hist = Hist(len(hist), 0, len(hist),
                    name=hist.name + '_uniform_binning')
    # assume yerrh == yerrl (as usual for ROOT histograms)
    for i, (value, error) in enumerate(zip(hist, hist.yerrh())):
        new_hist.SetBinContent(i + 1, value)
        new_hist.SetBinError(i + 1, error)
    return new_hist


def kylefix(hist):
    """

    Return a clone of the input histogram where the empty bins have been filled
    with the average weight and the errors of these bins set to sqrt(<w^2>)


                                            ..
                                  ..'..,c::;;,.. . .,... .
                               .....'....,;''.,'.... .... ;.
                            .... ..,,.'.,:,..;,,,'. .. ...  ,,
                          .;.    .,..';;,'..........,.    ..  ::
                      .','..'.....'.'..';:,,'.''.,','..        .d.
                     ,'. ...'ccodd:''....','.... .''.'..        .:
                   '......,o:;,',lc::;:......       .;;..
                  .; ....lc    . ...,;:clc:;::;'.....
                 .,. .  cc      .    .',,;,,;;lcc,.......
                 ,..   ;:    .          ..   ..'..    .....         .
                 ' .  .:.    .                           ....       .
                 ...  ,;     .                            ...      .
                  .  .c.                                   ..      ,
                  .. .d   ......                           .       o
                  '. ;o'loolcloldoc..          ...''...    ..     '.
                  ';.:0l,;c:','.';lo;.    .'::;;;,;;clll,   .     .
                  dk:kOdlcloo:,.....''...''..   .........:. .    .c
                 ;klxxxclxkkcllxlloc:,..,;clllccloo:::cccc: ,  .',l
                 xd:x o .;llo:;'.''o.:lc';' . .;,'.,'.. .l....;:...:.
                .lc0O:c   ... .....xx'..k:...'..,';;..   l..; c.   ;;
                ..lx.:cllocc:cc;;:0d     oo;.....    .';oO. '.:    ,,
                ; lo       ........       ;,codooodddoc;,.  ..,    .:
                c xo .          .          ..               . ;..  ,
                l dd           .            '               . 'c..'
                  :k           .c,... ....                 .. ;: .
                   '      .     .;'..  ..   .   .          .'';.'.
                    .    ... ......'........'.....,        ..:..
                    .   ...'......'';'''..... .....'       .''
                    ..    .cOxc,'.'..........'....,.      ..c
                     '....'. .:cockdclllcc;....... . .     :.
                     .:'..... .  ....... ..     .    ..   .o
                      .;....'... ..','''.       .   ...   .'
                       .,.',,'.. ..         ..... ...   .  ;.
                         :;c;;;;,',,''..............       ; ..
                         ,:o::;::;'''''''.'...             .. .,.
                         .,'l:;,'.......... .              .'...'.
                          , ;c;'...... ..  .              .'.....;.
                          '...c,..                       ,'...'..';.
                          :c .;:;..            .       ;''..''..;,...
                         :.', .;;'..                 .:';'';,.,;',:;'.
                       .l,,.;. .''..   .           .''';',;,,c,,c;,::l,.
                      'dd,'. l  ...               .c,,;',:;'c;;:;':':llo:'.
                    ':ol;:;. o.                 ',l;;:;,::,cc;c;:c;::;cc:;cl,.
                 .'::lx:l,,. ':               'cccc;;:'c:,l:;l;;l.;x,:;:::;cdol
            ...'.;;;oo:c,;;, '.'            .colcc:;l'cc,l:;o:;o,:l;::l:;:;;cld
         .'';::;,cx:.o;,c;,'...;  .       .:l:occ;,l,cc,o;,o,;o.l:;:lccd:c:c;;:
       .':;:::c:;c:  cl:..;,.:.:........,olxdlcl,:l,c:,o;:l,:c.x';oocoo:occxccc
      .;;;.:ccccl:   .,lcc,';l,.;';:;,.ol. .,co;cl,cc,o,:c'l;,d.'lclo:colo:x:cl
    ..;;:,.lldd;;.    ..,:.lxxo,;;:ollkl      .cc,cc,o,cc,l,:l'.l:ollclo'loclcc
    ;,,;,''ooc;..,,,;oc:,...:dol;:lcdd.         .lc;l,c:,o,cc..,:dl:lo'clco:ll:
    l::':.:co;; ;dxdodllc;,.,.ccc.;d'             .c;c:'o,cl'..:lcll.ldcloclll:
    :;,:,';o:c;';oxxdcoclc:,,;';xlo.        ....... .,;c'::,l.'coc.oodooolllo:l
    ::::..o;:c;:cdxdoooolc;c;',cloO.    . ...;,.'''.  .,o':o ';;:o;ooclololc:k,
    """
    llog = log['kylefix']
    fixed_hist = hist.Clone(name=hist.name + '_kylefix')

    sumW2TotBin = sum([yerr**2 for yerr in hist.yerrh()])
    avWeightBin = hist.GetSumOfWeights() / hist.GetEntries()
    avW2Bin = sumW2TotBin / hist.GetEntries()

    # now fill empty bins with
    # binContent = avWeight     [or avWeightbin]
    # binError = sqrt(avW2)     [or sqrt(avW2Bin)]

    for i in xrange(len(fixed_hist)):
        if fixed_hist[i] < 1E-6:
            llog.warning("filling empty or negative bin %d in %s" %
                         (i, hist.name))
            fixed_hist[i] = avWeightBin
            fixed_hist.SetBinError(i + 1, sqrt(avW2Bin))

    return fixed_hist
