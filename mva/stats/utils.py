from . import log; log = log[__name__]

import numpy as np
from rootpy.plotting import Hist


def get_safe_template(binning, bins, bkg_scores, sig_scores):

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

    log.info("minimum signal score: %f" % min_score_signal)
    log.info("maximum signal score: %f" % max_score_signal)

    # prevent bin threshold effects
    min_score_signal -= 0.00001
    max_score_signal += 0.00001

    if binning == 'flat':
        log.info("binning such that background is flat")
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

        log.info("binning such that each bin has at least one background")

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
                log.warning("early termination of binning")
                break
            # bump expected bin index down until each background is positive
            bin_index_expected_correct = all_positive[:bin_index_expected + 1][::-1].argmax()
            if bin_index_expected_correct > 0:
                log.warning(
                    "expected bin index corrected such that all "
                    "backgrounds are positive")
            bin_index_expected -= bin_index_expected_correct
            if (bin_index_expected <= 0 or
                bin_index_expected <= (nbins / bins)):
                log.warning("early termination of binning after correction")
                break
            bin_edge_expected = total_bkg_hist.xedges(int(bin_index_expected))

            # if this edge is greater than it would otherwise be if we used
            # constant-width binning over the whole range then just use the
            # original binning
            if bin_edge > bin_edge_expected:
                log.info("expected bin edge %f is OK" % bin_edge_expected)
                bin_edge = bin_edge_expected
                view_cutoff = bin_index_expected

            else:
                log.info("adjusting bin to contain >= one background")
                log.info("original edge: %f  new edge: %f " %
                        (bin_edge_expected, bin_edge))
                view_cutoff = last_bin_one_bkg

            edges.append(bin_edge)

        edges.append(min_score_signal)
        log.info("edges %s" % str(edges))
        hist_template = Hist(edges[::-1])

    else:
        log.info("using constant-width bins")
        hist_template = Hist(bins,
                min_score_signal, max_score_signal)
    return hist_template



def kylefix(hist):
    """
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWWMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMN0xoolccc::::;;,'.....';xNMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMKc....    ..... ..       ...;xMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMWxc..      . ...........      .   cWMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMK:'.     .........        ..        .l0MMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMx,.... .......  .....  .......           ;oNMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMM0,......,:c:;'....  ..                      .,WMMMMMMMMMMM
    MMMMMMMMMMMMMMMK,......:k000Oxlc:;'......         .            cWMMMMMMMMMM
    MMMMMMMMMMMMMMx.......x0XNXXK0OOxdoclc;,'......                 xMMMMMMMMMM
    MMMMMMMMMMMMMN'...  .xKXNNXXKKXKK0Okkkxdlc:;;,'.....            oMMMMMMMMMM
    MMMMMMMMMMMMM:.    .o0XXXXXXXXXXXXXKK00Okkxxdolc:;,...          ,MMMMMMMMMM
    MMMMMMMMMMMMM.    .ck0XXXXXXXXXNNNXXXXKK0OOOkxddlc:;,''.        :MMMMMMMMMM
    MMMMMMMMMMMMM;.  .'d0KXXXXXXXNNNNNNNNXXXKKK00Okxdolc:;'..       WMMMMMMMMMM
    MMMMMMMMMMMMMX.  .:kKXXNXXNNNNNNNNNNNNXXXXKK00Okxddlc;'..      'MMMMMMMMMMM
    MMMMMMMMMMMMMW,...l0KXXXXXXXXXNNNNNNNNNNXXXKK0OOkxdoc;,..      cMMMMMMMMMMM
    MMMMMMMMMMMMMM,. .xOo:ccc:;:ldk0KKKXXXXKK000kkkxxddl:;,.       lMMMMMMMMMMM
    MMMMMMMMMMMMMWc. .o,..........'cdxOOkkxol:;,.......';;,'.     .OMMMMMMMMMMM
    MMMMMMMMMMMMMXx'.'c:lollc:,'.',;:oxxoc;'...........  .,'.     dWMMMMMMMMMMM
    MMMMMMMMMMMMNkc.,::ol;;,...'...,lOK0kc,''...''''......''.    .cKMMMMMMMMMMM
    MMMMMMMMMMMMxx:.cox0Oxol,'';,,;, .'. .,'...'.  ....'..  .. ';,':NMMMMMMMMMM
    MMMMMMMMMMMMxO0;ckOXKOdlcccldkO;:0X0:.oc::::;,,;,'',;' .. .;::',OMMMMMMMMMM
    MMMMMMMMMMMMxKOcOkO0OOxdddkOOkloXNNKO;'dxxdlc::::cllo';;' .:cc,,lMMMMMMMMMM
    MMMMMMMMMMMM0KOlKKXXXXKKOkkkkkKXXNXKOxc,cdxxdddddoolllc;,..';l:cOMMMMMMMMMM
    MMMMMMMMMMMMKXxlKKXXXKK0OOkxOKXNNNXKOdolcoxkkkkkkxxddoc;' .';lcoWMMMMMMMMMM
    MMMMMMMMMMMMWXdcKKKKKKK0OkkOXK0OOOkxddxklldxxxxxxxddlc:,.  clccNMMMMMMMMMMM
    MMMMMMMMMMMMMKxl0000KK0000K0l;,,;;,'.,;::oxkkkkxddolc:;'.  :odNMMMMMMMMMMMM
    MMMMMMMMMMMMMWxd0OOO00kO0XXKOdl:,,''...,cxkxddxxolc:;;,'..'cxWMMMMMMMMMMMMM
    MMMMMMMMMMMMMMWxOkOOkxO0KKK000Okolllc::codddoc:llc:;;;,'.'lXMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMxxkkkdoodO0OOOkkkkkkxxxddllccll:lcc;;;,,''0MMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMNdxxkxddl;..;:c:::;;;,,,,,''',;coc:,,,,'.KMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMKoxxxxxOOOxdodlolclc:;''.';:ccol;,,,,'.,MMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMxodddxkOOkOOkkOOOOxdol:::cc:::;,''''..:NMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMM0dddodxkxdl:,,,'','',;::;;,;,''''...';dMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMXldooddxol:,'''.',;;;;,'',,''.....',,:KMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMkcloxxkxkxolc:::::;;;,','''.....',,.  cXMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMOolddkkkkkxddxdol:;,''.......',,,,.   .xMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMNdoccloollcc::c;,'..........',;;,,.    .OMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMNlxl:,,,',,,,''............',,;;,.      .oWMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMN:ddl:,..................'',,,;,.        .,NMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMx',ddlc,'...............',,,,,.        .  .'kWMMMMMMMM
    MMMMMMMMMMMMMMMMMMMNd'  ldolc;'...........''''','.. .  .. .. .....;kMMMMMMM
    MMMMMMMMMMMMMMMMMMk;'.  .oolc;,''.......'',''''..  .  . ....... ....;XMMMMM
    MMMMMMMMMMMMMMMWOc,'.    ;dol:,,',,,'''',,''''................. ......dWMMM
    MMMMMMMMMMMMMNd;,''...   .odol:;,,;,,,,,,,''.................. ........,0MM
    MMMMMMMMMMMKd;,'''....    ;ddocc::;;;;;;,'................... ...........cN
    MMMMMMMWOo;','.,,.....   .'ldoc::;;;:::,'..................................
    MMMMMWk;'''',.'. .....   .,cdl:::;::c:,....................................
    MMMNd'''''',,;.  .... .  .;:oc::;;cll;..'...................  .............
    MMN;..''''','.     ....  .,:llcccldc.    ..................  ..............
    MW;...''''....       . ....,:ccccl:.       ...............  ...............
    W;....''''''........    ....';cl:.           ............   ...............
    c....''''''''''........   ...';.               ..........  ..... ..........
    .....''','','''......... .  ...                   .......  ... ............
    ....''',,',,,'.................                     ....   ................
    """
    """
    double sumW2TotBin_Z=0, avWeightBin_Z=0, avW2Bin_Z=0;
    double sumW2TotBin_QCD=0, avWeightBin_QCD=0, avW2Bin_QCD=0;
    double sumW2TotBin_EW=0, avWeightBin_EW=0, avW2Bin_EW=0;
    double sumW2TotBin_H=0, avWeightBin_H=0, avW2Bin_H=0;

    for(int j=1; j<=hmap_Sig[800001]->GetNbinsX(); ++j){
        sumW2TotBin_Z   += pow( hmap_Sig[800001]->GetBinError(j) , 2); // DON't forget to square!
        sumW2TotBin_QCD += pow( hist_QCD        ->GetBinError(j) , 2);
        sumW2TotBin_EW  += pow( hmap_Sig[800005]->GetBinError(j) , 2);
        sumW2TotBin_H   += pow( hmap_Sig[800000]->GetBinError(j) , 2);
    }
    avWeightBin_Z += hmap_Sig[800001]->GetSumOfWeights() / hmap_Sig[800001]->GetEntries();
    avW2Bin_Z = sumW2TotBin_Z/hmap_Sig[800001]->GetEntries();

    avWeightBin_QCD += hist_QCD->GetSumOfWeights() / hist_QCD->GetEntries();
    avW2Bin_QCD = sumW2TotBin_QCD/hist_QCD->GetEntries();

    avWeightBin_EW += hmap_Sig[800005]->GetSumOfWeights() / hmap_Sig[800005]->GetEntries();
    avW2Bin_EW = sumW2TotBin_EW/hmap_Sig[800005]->GetEntries();

    avWeightBin_H += hmap_Sig[800000]->GetSumOfWeights() / hmap_Sig[800000]->GetEntries();
    avW2Bin_H = sumW2TotBin_H/hmap_Sig[800000]->GetEntries();

    double Z = 999.; double Q = 999.; double EW = 999.;  double H = 999.;

    # now fill empty bins with
    # binContent = avWeight     [or avWeightbin]
    # binError = sqrt(avW2)     [or sqrt(avW2Bin)]

    for(int j=1; j<=hmap_Sig[800001]->GetNbinsX(); ++j){
        Z =  hmap_Sig[800001]->GetBinContent(j);
        if( Z  < 1e-6 )  {
            cout<<"empty Z: "<<hmap_Sig[800001]->GetBinCenter(j)<<endl;
            hmap_Sig[800001]->SetBinContent(j,avWeightBin_Z);
            hmap_Sig[800001]->SetBinError(j,sqrt(avW2Bin_Z));
        }
        Q =  hist_QCD->GetBinContent(j);
        if( Q  < 1e-6 )  {
            hist_QCD->SetBinContent(j,avWeightBin_QCD);
            hist_QCD->SetBinError(j,sqrt(avW2Bin_QCD));
        }
        EW =  hmap_Sig[800005]->GetBinContent(j);
        if( EW  < 1e-6 )  {
            hmap_Sig[800005]->SetBinContent(j,avWeightBin_EW);
            hmap_Sig[800005]->SetBinError(j,sqrt(avW2Bin_EW));
        }
        H =  hmap_Sig[800000]->GetBinContent(j);
        if( H  < 1e-6 )  {
            hmap_Sig[800000]->SetBinContent(j,avWeightBin_H);
            hmap_Sig[800000]->SetBinError(j,sqrt(avW2Bin_H));
        }

    }

    hist_BKG_empty_bin_fix->Add( hist_QCD, 1.0); ///qcd
    hist_BKG_empty_bin_fix->Add( hmap_Sig[800005], 1.0);///ew
    hist_BKG_empty_bin_fix->Add( hmap_Sig[800001], 1.0); ///Z
    """
    return hist
