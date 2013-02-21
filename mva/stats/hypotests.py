from . import log; log = log[__name__]
from .asymptotics import AsymptoticsCLs
from ..samples import Higgs


def limits(data, backgrounds,
           category, region=target_region,
           cuts=None,
           unblind=False,
           systematics=False,
           bins=10,
           binning='constant'):

    for mass in Higgs.MASS_POINTS:

        log.info('=' * 20)
        log.info("%d GeV mass hypothesis" % mass)
        # create separate signal. background and data histograms for each
        # mass hypothesis since the binning is optimized for each mass
        # individually.
        # The binning is determined by first locating the BDT cut value at
        # which the signal significance is maximized (S / sqrt(B)).
        # Everything above that cut is put in one bin. Everything below that
        # cut is put into N variable width bins such that the background is
        # flat.
        min_score_signal = 1.
        max_score_signal = -1.
        sig_scores = []
        # signal scores
        for mode in Higgs.MODES:
            sig = Higgs(year=year, mode=mode, mass=mass,
                    systematics=systematics)
            scores_dict = sig.scores(self,
                    region=self.region,
                    cuts=signal_region)

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

            sig_scores.append((sig, scores_dict))

        if mass == 125:
            sig_scores_125 = sig_scores

        log.info("minimum signal score: %f" % min_score_signal)
        log.info("maximum signal score: %f" % max_score_signal)

        # prevent bin threshold effects
        min_score_signal -= 0.00001
        max_score_signal += 0.00001

        # add a bin above max score and below min score for extra beauty
        score_width_signal = max_score_signal - min_score_signal
        bin_width_signal = score_width_signal / bins

        plot_clf(
            background_scores=bkg_scores,
            signal_scores=sig_scores,
            category=self.category,
            category_name=self.category_name,
            plot_label='mass signal region',
            signal_scale=signal_scale,
            name='%d_ROI%s' % (mass, self.output_suffix),
            bins=bins + 2,
            min_score=min_score_signal - bin_width_signal,
            max_score=max_score_signal + bin_width_signal,
            systematics=SYSTEMATICS if systematics else None)

        if limitbinning == 'flat':
            log.info("variable-width bins")
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
            log.info("maximum signal significance of %f at score > %f" % (
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

            plot_clf(
                background_scores=bkg_scores,
                signal_scores=sig_scores,
                category=self.category,
                category_name=self.category_name,
                plot_label='mass signal region',
                signal_scale=signal_scale,
                name='%d_ROI_flat%s' % (mass, self.output_suffix),
                bins=flat_bins,
                plot_signal_significance=False,
                signal_on_top=True,
                systematics=SYSTEMATICS if systematics else None)

            hist_template = Hist(flat_bins)

        elif limitbinning == 'onebkg':
            # Define last bin such that it contains at least one background.
            # First histogram background with a very fine binning,
            # then sum from the right to the left up to a total of one
            # event. Use the left edge of that bin as the left edge of the
            # last bin in the final histogram template.
            # Important: also choose the bin edge such that all background
            # components each have at least zero events, since we have
            # samples with negative weights (SS subtraction in the QCD) and
            # MC@NLO samples.

            log.info("one background in last bin")
            total_bkg_hist = Hist(1000, min_score_signal, max_score_signal)
            sums = []

            # fill background
            for bkg_sample, scores_dict in bkg_scores:
                score, w = scores_dict['NOMINAL']
                bkg_hist = total_bkg_hist.Clone()
                bkg_hist.fill_array(score, w)

                # create array from histogram
                bkg_array = np.array(bkg_hist)

                # reverse cumsum
                bkg_cumsum = bkg_array[::-1].cumsum()[::-1]

                sums.append(bkg_cumsum)

            total_bkg_cumsum = np.add.reduce(sums)

            # determine last element with at least a value of 1.
            # and where each background has at least zero events
            # so that no sample may have negative events in this bin
            all_positive = np.logical_and.reduce([b >= 0. for b in sums])
            last_bin_all_positive = np.argmin(all_positive) - 1

            last_bin = int(min(np.where(bkg_cumsum >= 1.)[-1][-1],
                               last_bin_all_positive))

            # get left bin edge corresponding to this bin
            bin_edge = bkg_hist.xedges(last_bin)

            # if this edge is greater than it would otherwise be if we used
            # constant-width binning over the whole range then just use the
            # original binning
            default_bins = list(np.linspace(
                    min_score_signal,
                    max_score_signal,
                    limitbins + 1))

            if bin_edge > default_bins[-2]:
                log.info("constant-width bins are OK")
                one_bkg_bins = default_bins

            else:
                log.info("adjusting last bin to contain >= one background")
                log.info("original edge: %f  new edge: %f " %
                        (default_bins[-2],
                         bin_edge))

                # now define N-1 constant-width bins to the left of this edge
                left_bins = np.linspace(
                        min_score_signal,
                        bin_edge,
                        limitbins)

                one_bkg_bins = list(left_bins)
                one_bkg_bins.append(max_score_signal)

            plot_clf(
                background_scores=bkg_scores,
                signal_scores=sig_scores,
                category=self.category,
                category_name=self.category_name,
                plot_label='mass signal region',
                signal_scale=signal_scale,
                name='%d_ROI_onebkg%s' % (mass, self.output_suffix),
                bins=one_bkg_bins,
                plot_signal_significance=True,
                systematics=SYSTEMATICS if systematics else None)

            hist_template = Hist(one_bkg_bins)

        else:
            log.info("constant-width bins")
            hist_template = Hist(limitbins,
                    min_score_signal, max_score_signal)

        f.cd()
        if unblind:
            data_hist = hist_template.Clone(name=data.name + '_%s' % mass)
            data_hist.fill_array(data_scores)
            data_hist.Write()
        write_score_hists(f, mass, bkg_scores, hist_template, no_neg_bins=True)
        write_score_hists(f, mass, sig_scores, hist_template, no_neg_bins=True)

