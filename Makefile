
HHSTUDENT ?= hhskim
HHNTUP ?= ntuples/prod_v29/hhskim
HHNTUP_RUNNING ?= ntuples/running/hhskim

BRANCH := $(shell git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\1/')

.PHONY: dump

default: clean

clean-root:
	rm -f $(HHNTUP)/$(HHSTUDENT).root

clean-h5:
	rm -f $(HHNTUP)/$(HHSTUDENT).h5

clean-ntup: clean-root clean-h5

check-files:
	./checkfile $(HHNTUP_RUNNING)/$(HHSTUDENT)*.root

check-ntup:
	./checkfile $(HHNTUP)/$(HHSTUDENT).root

browse:
	rootpy browse $(HHNTUP)/$(HHSTUDENT).root

roosh:
	roosh $(HHNTUP)/$(HHSTUDENT).root

$(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss.root:
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss_1.root ]; then \
		test -d $(HHNTUP_RUNNING)/data || mkdir $(HHNTUP_RUNNING)/data; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss_*.root $(HHNTUP_RUNNING)/data; \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss.root $(HHNTUP_RUNNING)/data/$(HHSTUDENT).data12-JetTauEtmiss_*.root; \
		test -d $(HHNTUP_RUNNING)/data_log || mkdir $(HHNTUP_RUNNING)/data_log; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data12_*.e[0-9]* $(HHNTUP_RUNNING)/data_log/; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data12_*.o[0-9]* $(HHNTUP_RUNNING)/data_log/; \
		mv $(HHNTUP_RUNNING)/supervisor-$(HHSTUDENT)-$(HHSTUDENT).data12-JetTauEtmiss_*.log $(HHNTUP_RUNNING)/data_log/; \
	fi

$(HHNTUP_RUNNING)/$(HHSTUDENT).data11-JetTauEtmiss.root:
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).data11-JetTauEtmiss_1.root ]; then \
		test -d $(HHNTUP_RUNNING)/data || mkdir $(HHNTUP_RUNNING)/data; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data11-JetTauEtmiss_*.root $(HHNTUP_RUNNING)/data; \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).data11-JetTauEtmiss.root $(HHNTUP_RUNNING)/data/$(HHSTUDENT).data11-JetTauEtmiss_*.root; \
		test -d $(HHNTUP_RUNNING)/data_log || mkdir $(HHNTUP_RUNNING)/data_log; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data11_*.e[0-9]* $(HHNTUP_RUNNING)/data_log/; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data11_*.o[0-9]* $(HHNTUP_RUNNING)/data_log/; \
		mv $(HHNTUP_RUNNING)/supervisor-$(HHSTUDENT)-$(HHSTUDENT).data11-JetTauEtmiss_*.log $(HHNTUP_RUNNING)/data_log/; \
	fi

init-data-12: $(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss.root

init-data-11: $(HHNTUP_RUNNING)/$(HHSTUDENT).data11-JetTauEtmiss.root

init-data: init-data-11 init-data-12

$(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_TES_FAKE_TOTAL_UP.root:
	test -d $(HHNTUP_RUNNING)/embed_tes || mkdir $(HHNTUP_RUNNING)/embed_tes
	
	for TES_TERM in TES_TRUE_MODELING TES_TRUE_SINGLEPARTICLEINTERPOL TES_TRUE_INSITUINTERPOL TES_FAKE_TOTAL; do \
		if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_UP_1.root ]; then \
			mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_*.root $(HHNTUP_RUNNING)/embed_tes; \
		fi; \
		if [ -f $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_UP_1.root ]; then \
			hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_UP.root $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_UP_*.root; \
			hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_DOWN.root $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_DOWN_*.root; \
		fi; \
	done

$(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM.root:
	test -d $(HHNTUP_RUNNING)/embed || mkdir $(HHNTUP_RUNNING)/embed
	
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_1.root ]; then \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_[0-9]*.root $(HHNTUP_RUNNING)/embed; \
	fi
	if [ -f $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-IM_1.root ]; then \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM.root $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-IM_*.root; \
	fi
	
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-UP_1.root ]; then \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-UP_*.root $(HHNTUP_RUNNING)/embed; \
	fi
	if [ -f $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-UP_1.root ]; then \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-UP.root $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-UP_*.root; \
	fi
	
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-DN_1.root ]; then \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-DN_*.root $(HHNTUP_RUNNING)/embed; \
	fi
	if [ -f $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-DN_1.root ]; then \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-DN.root $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-DN_*.root; \
	fi

$(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim_TES_TRUE_FINAL_UP.root:
	test -d $(HHNTUP_RUNNING)/embed_tes || mkdir $(HHNTUP_RUNNING)/embed_tes
	
	for TES_TERM in TES_TRUE_FINAL TES_FAKE_FINAL; do \
		if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim_$${TES_TERM}_UP_1.root ]; then \
			mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim_$${TES_TERM}_*.root $(HHNTUP_RUNNING)/embed_tes; \
		fi; \
		if [ -f $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed11-hh-isol-mfsim_$${TES_TERM}_UP_1.root ]; then \
			hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim_$${TES_TERM}_UP.root $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed11-hh-isol-mfsim_$${TES_TERM}_UP_*.root; \
			hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim_$${TES_TERM}_DOWN.root $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed11-hh-isol-mfsim_$${TES_TERM}_DOWN_*.root; \
		fi; \
	done

$(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim.root:
	test -d $(HHNTUP_RUNNING)/embed || mkdir $(HHNTUP_RUNNING)/embed
	
	for syst in im up dn; do \
		if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfs$${syst}_1.root ]; then \
			mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfs$${syst}_[0-9]*.root $(HHNTUP_RUNNING)/embed; \
		fi; \
		if [ -f $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed11-hh-isol-mfs$${syst}_1.root ]; then \
			hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfs$${syst}.root $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed11-hh-isol-mfs$${syst}_[0-9]*.root; \
		fi; \
	done
	
	for syst in no tight; do \
		if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-$${syst}isol-mfsim_1.root ]; then \
			mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-$${syst}isol-mfsim_[0-9]*.root $(HHNTUP_RUNNING)/embed; \
		fi; \
		if [ -f $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed11-hh-$${syst}isol-mfsim_1.root ]; then \
			hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-$${syst}isol-mfsim.root $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed11-hh-$${syst}isol-mfsim_[0-9]*.root; \
		fi; \
	done

.PHONY: embed-log
embed-log:
	test -d $(HHNTUP_RUNNING)/embed_log || mkdir $(HHNTUP_RUNNING)/embed_log
	-mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed1[1-2]-*.e[0-9]* $(HHNTUP_RUNNING)/embed_log/
	-mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed1[1-2]-*.o[0-9]* $(HHNTUP_RUNNING)/embed_log/
	-mv $(HHNTUP_RUNNING)/supervisor-$(HHSTUDENT)-$(HHSTUDENT).embed1[1-2]-*.log $(HHNTUP_RUNNING)/embed_log/

init-embed-11-nominal: $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim.root

init-embed-11-sys: $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim_TES_TRUE_FINAL_UP.root

init-embed-11: init-embed-11-nominal init-embed-11-sys

init-embed-12-sys: $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_TES_FAKE_TOTAL_UP.root

init-embed-12-nominal: $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM.root

init-embed-12: init-embed-12-nominal init-embed-12-sys embed-log

init-embed: init-embed-11 init-embed-12 embed-log

init-mc:
	test -d $(HHNTUP_RUNNING)/mc_log || mkdir $(HHNTUP_RUNNING)/mc_log
	-mv $(HHNTUP_RUNNING)/$(HHSTUDENT).*mc1[1-2]*.e[1-9]* $(HHNTUP_RUNNING)/mc_log/
	-mv $(HHNTUP_RUNNING)/$(HHSTUDENT).*mc1[1-2]*.o[1-9]* $(HHNTUP_RUNNING)/mc_log/
	-mv $(HHNTUP_RUNNING)/supervisor-$(HHSTUDENT)-$(HHSTUDENT).*mc1[1-2]*.log $(HHNTUP_RUNNING)/mc_log/

init-ntup: init-data init-embed init-mc

$(HHNTUP)/$(HHSTUDENT).root:
	./merge-ntup -s $(HHSTUDENT) -o $(HHNTUP)/$(HHSTUDENT).root $(HHNTUP)/$(HHSTUDENT).*.root

$(HHNTUP)/$(HHSTUDENT).h5: $(HHNTUP)/$(HHSTUDENT).root
	root2hdf5 --complib lzo --complevel 0 --quiet $^

ntup: $(HHNTUP)/$(HHSTUDENT).h5

.PHONY: ntup-update
ntup-update:
	./merge-ntup -s $(HHSTUDENT) -o $(HHNTUP)/$(HHSTUDENT).root $(HHNTUP_RUNNING)/$(HHSTUDENT).*.root
	root2hdf5 --update --complib lzo --complevel 0 --quiet $(HHNTUP)/$(HHSTUDENT).root

.PHONY: $(HHNTUP)/merged_grl_11.xml
$(HHNTUP)/merged_grl_11.xml:
	ls $(HHNTUP)/data/hhskim.data11-*.root | sed 's/$$/:\/lumi/g' | xargs grl or > $@

.PHONY: $(HHNTUP)/merged_grl_12.xml
$(HHNTUP)/merged_grl_12.xml:
	ls $(HHNTUP)/data/hhskim.data12-*.root | sed 's/$$/:\/lumi/g' | xargs grl or > $@

.PHONY: $(HHNTUP)/observed_grl_11.xml
$(HHNTUP)/observed_grl_11.xml: $(HHNTUP)/merged_grl_11.xml ../higgstautau/grl/2011/current.xml 
	grl and $^ > $@

.PHONY: $(HHNTUP)/observed_grl_12.xml
$(HHNTUP)/observed_grl_12.xml: $(HHNTUP)/merged_grl_12.xml ../higgstautau/grl/2012/current.xml 
	grl and $^ > $@

.PHONY: ~/observed_grl_11.xml
~/observed_grl_11.xml: $(HHNTUP)/observed_grl_11.xml
	cp $^ $@

.PHONY: ~/observed_grl_12.xml
~/observed_grl_12.xml: $(HHNTUP)/observed_grl_12.xml
	cp $^ $@

.PHONY: grl-11
grl-11: ~/observed_grl_11.xml

.PHONY: grl-12
grl-12: ~/observed_grl_12.xml

grl: grl-11 grl-12

clean-grl:
	rm -f $(HHNTUP)/observed_grl_11.xml
	rm -f $(HHNTUP)/observed_grl_12.xml
	rm -f ~/observed_grl_11.xml
	rm -f ~/observed_grl_12.xml
	rm -f $(HHNTUP)/merged_grl_11.xml
	rm -f $(HHNTUP)/merged_grl_12.xml

clean-pyc:                                                                      
	find . -name "*.pyc" -exec rm {} \;

clean: clean-pyc

bundle:
	rm -f ~/higgstautau-mva-plots.tar.gz
	find plots/variables/$(BRANCH) -name '*.eps' -print0 | tar -vpcz --null -T - -f ~/higgstautau-mva-plots.tar.gz
	@echo bundle at ~/higgstautau-mva-plots.tar.gz

png-bundle:
	rm -f ~/higgstautau-mva-plots.tar.gz
	find plots/variables/$(BRANCH) -name '*.png' -print0 | tar -vpcz --null -T - -f ~/higgstautau-mva-plots.tar.gz
	@echo bundle at ~/higgstautau-mva-plots.tar.gz

montage:
	montage -tile 4x5 -geometry 400x400+3+3 plots/*.png montage.pdf

test:
	nosetests -s -v mva

dump:
	@./dump -t higgstautauhh -s "taus_pass && (RunNumber==207528)" --select-file etc/embed_select_ac.txt -o RunNumber,EventNumber $(HHNTUP)/$(HHSTUDENT).embed12-HH-IM.root
	@./dump -t higgstautauhh -e 50 -s "taus_pass" -o EventNumber $(HHNTUP)/$(HHSTUDENT).AlpgenJimmy_AUET2CTEQ6L1_ZtautauNp4.mc12a.root
	@./dump -t higgstautauhh -e 50 -s "taus_pass" -o EventNumber $(HHNTUP)/$(HHSTUDENT).AlpgenJimmy_AUET2CTEQ6L1_ZtautauNp0.mc12a.root


.PHONY: norms
norms:
	for year in 2011 2012; do \
		for model in OS_NONISOL nOS nOS_ISOL nOS_NONISOL SS SS_ISOL SS_NONISOL NONISOL; do \
			nohup ./norm --no-systematics --fakes-region $${model} --year $${year} > norm_ebz_$${model}_$${year}.log & \
			nohup ./norm --no-systematics --no-embedding --fakes-region $${model} --year $${year} > norm_mcz_$${model}_$${year}.log & \
		done; \
	done

.PHONY: stats-plots
stats-plots:
	for year in 2011 2012; do \
		for model in OS_NONISOL nOS nOS_ISOL nOS_NONISOL SS SS_ISOL SS_NONISOL NONISOL; do \
			nohup ./ana plot --fakes-region $${model} --no-systematics --year $${year} --output-formats eps png > var_plots_$${year}_$${model}.log & \
			nohup ./ana plot --fakes-region $${model} --no-systematics --year $${year} --categories presel --output-formats eps png > var_plots_presel_$${year}_$${model}.log & \
		done; \
	done

.PHONY: plots
plots:
	nohup ./ana plot --year 2012 --category-names vbf --output-formats eps png > var_plots_vbf_12.log &
	nohup ./ana plot --year 2012 --category-names boosted --output-formats eps png > var_plots_boosted_12.log &
	nohup ./ana plot --year 2012 --category-names rest --output-formats eps png > var_plots_rest_12.log &
	nohup ./ana plot --year 2012 --categories presel --output-formats eps png > var_plots_presel_12.log &
	#nohup ./ana plot --year 2011 --category-names vbf --output-formats eps png > var_plots_vbf_11.log &
	#nohup ./ana plot --year 2011 --category-names boosted --output-formats eps png > var_plots_boosted_11.log &
	#nohup ./ana plot --year 2011 --category-names rest --output-formats eps png > var_plots_rest_11.log &
	#nohup ./ana plot --year 2011 --categories presel --output-formats eps png > var_plots_presel_11.log &

.PHONY: mva-plots
mva-plots:
	nohup ./ana train evaluate --year 2012 --output-formats eps png --category-names vbf > mva_plots_vbf_12.log &
	nohup ./ana train evaluate --year 2012 --output-formats eps png --category-names boosted > mva_plots_boosted_12.log &
	#nohup ./ana train evaluate --year 2011 --output-formats eps png --category-names vbf > mva_plots_vbf_11.log &
	#nohup ./ana train evaluate --year 2011 --output-formats eps png --category-names boosted > mva_plots_boosted_11.log &

.PHONY: mva-control-plots
mva-control-plots:
	nohup ./ana train evaluate --unblind --output-formats eps png --category-names vbf_deta_control --categories mva_deta_controls > vbf_deta_control_plots.log & 
	nohup ./ana train evaluate --unblind --output-formats eps png --category-names boosted_deta_control --categories mva_deta_controls > boosted_deta_control_plots.log & 

.PHONY: train
train:
	@for mass in $$(seq 100 5 150); do \
		run-cluster ./train --mass $${mass}; \
	done

.PHONY: binning
binning:
	@for year in 2011 2012; do \
		for mass in $$(seq 100 5 150); do \
			run-cluster ./optimize-binning --systematics --year $${year} --mass $${mass}; \
		done; \
	done

.PHONY: mva-workspaces
mva-workspaces:
	@for year in 2011 2012; do \
		for mass in $$(seq 100 5 150); do \
			run-cluster ./workspace mva --systematics --years $${year} --masses $${mass}; \
		done; \
	done

.PHONY: cuts-workspaces
cuts-workspaces:
	@for mass in $$(seq 100 5 150); do \
		run-cluster ./workspace cuts --systematics --years 2011 --categories cuts_2011 --masses $${mass}; \
	done;
	@for mass in $$(seq 100 5 150); do \
		run-cluster ./workspace cuts --systematics --years 2012 --categories cuts --masses $${mass}; \
	done;

.PHONY: higgs-pt
higgs-pt:
	./higgs-pt ntuples/running/hhskim/hhskim*tautauhh*.root
