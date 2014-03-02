
HHSTUDENT ?= hhskim
HHNTUP ?= ntuples/prod/hhskim
HHNTUP_RUNNING ?= ntuples/running/hhskim

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

$(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_TES_EOP_UP.root:
	test -d $(HHNTUP_RUNNING)/embed_tes || mkdir $(HHNTUP_RUNNING)/embed_tes
	
	for TES_TERM in TES TES_TRUE TES_FAKE TES_EOP TES_CTB TES_Bias TES_EM TES_LCW TES_PU TES_OTHERS; do \
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

init-embed-11: $(HHNTUP_RUNNING)/$(HHSTUDENT).embed11-hh-isol-mfsim.root

init-embed-12-sys: $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_TES_EOP_UP.root

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
	tar -vpczf ~/higgstautau-mva-plots.tar.gz plots/*.eps 
	@echo bundle at ~/higgstautau-mva-plots.tar.gz

png-bundle:
	rm -f ~/higgstautau-mva-plots.tar.gz
	tar -vpczf ~/higgstautau-mva-plots.tar.gz plots/*.png 
	@echo bundle at ~/higgstautau-mva-plots.tar.gz

montage:
	montage -tile 4x5 -geometry 400x400+3+3 plots/*.png montage.pdf

test:
	nosetests -s -v mva

dump:
	@./dump -t higgstautauhh -s "taus_pass && (RunNumber==207528)" --select-file etc/embed_select_ac.txt -o RunNumber,EventNumber $(HHNTUP)/$(HHSTUDENT).embed12-HH-IM.root
	@./dump -t higgstautauhh -e 50 -s "taus_pass" -o EventNumber $(HHNTUP)/$(HHSTUDENT).AlpgenJimmy_AUET2CTEQ6L1_ZtautauNp4.mc12a.root
	@./dump -t higgstautauhh -e 50 -s "taus_pass" -o EventNumber $(HHNTUP)/$(HHSTUDENT).AlpgenJimmy_AUET2CTEQ6L1_ZtautauNp0.mc12a.root


.PHONY: stats-plots
stats-plots:
	nohup ./ana plot --year 2012 --no-systematics --category-names vbf --output-formats eps png > var_plots_vbf_12.log &
	nohup ./ana plot --year 2012 --no-systematics --category-names boosted --output-formats eps png > var_plots_boosted_12.log &
	nohup ./ana plot --year 2012 --no-systematics --category-names rest --output-formats eps png > var_plots_rest_12.log &
	nohup ./ana plot --year 2012 --no-systematics --categories presel --output-formats eps png > var_plots_presel_12.log &
	nohup ./ana plot --year 2011 --no-systematics --category-names vbf --output-formats eps png > var_plots_vbf_11.log &
	nohup ./ana plot --year 2011 --no-systematics --category-names boosted --output-formats eps png > var_plots_boosted_11.log &
	nohup ./ana plot --year 2011 --no-systematics --category-names rest --output-formats eps png > var_plots_rest_11.log &
	nohup ./ana plot --year 2011 --no-systematics --categories presel --output-formats eps png > var_plots_presel_11.log &

.PHONY: plots
plots:
	nohup ./ana plot --unblind --category-names vbf --output-formats eps png > var_plots_vbf.log &
	nohup ./ana plot --unblind --category-names boosted --output-formats eps png > var_plots_boosted.log &
	nohup ./ana plot --unblind --category-names rest --output-formats eps png > var_plots_rest.log &
	nohup ./ana plot --unblind --categories presel --output-formats eps png > var_plots_presel.log &

.PHONY: stats-bdt-plots
stats-bdt-plots:
	nohup ./ana train evaluate --year 2011 --use-2012-clf --no-systematics --output-formats eps png --category-names vbf > bdt_plots_vbf_11.log &
	nohup ./ana train evaluate --year 2011 --use-2012-clf --no-systematics --output-formats eps png --category-names boosted > bdt_plots_boosted_11.log &
	nohup ./ana train evaluate --year 2012 --no-systematics --output-formats eps png --category-names vbf > bdt_plots_vbf_12.log &
	nohup ./ana train evaluate --year 2012 --no-systematics --output-formats eps png --category-names boosted > bdt_plots_boosted_12.log &

.PHONY: bdt-plots
bdt-plots:
	nohup ./ana train evaluate --unblind --output-formats eps png --category-names vbf > bdt_plots_vbf.log &
	nohup ./ana train evaluate --unblind --output-formats eps png --category-names boosted > bdt_plots_boosted.log &

.PHONY: bdt-control-plots
bdt-control-plots:
	nohup ./ana train evaluate --unblind --output-formats eps png --category-names vbf_deta_control --categories mva_deta_controls > vbf_deta_control_plots.log & 
	nohup ./ana train evaluate --unblind --output-formats eps png --category-names boosted_deta_control --categories mva_deta_controls > boosted_deta_control_plots.log & 

.PHONY: workspace-unblind
workspace-unblind:
	nohup ./ana workspace --unblind --mass-points all > workspace_unblind.log &

.PHONY: workspace
workspace:
	nohup ./ana workspace --mass-points all > workspace.log &
	nohup ./ana workspace --unblind --mu 123 --workspace-suffix unblinded_random_mu --mass-points all > workspace_unblind_random.log &

.PHONY: workspace-const
workspace-const:
	nohup ./ana workspace --constrain-norms --workspace-suffix const_norms --mass-points all > workspace_const_norms.log &
	nohup ./ana workspace --constrain-norms --unblind --mu 123 --workspace-suffix const_norms_unblinded_random_mu --mass-points all > workspace_const_norms_unblind_random.log &

.PHONY: workspace-125
workspace-125:
	nohup ./ana workspace > workspace.log &
	nohup ./ana workspace --unblind --mu 123 --workspace-suffix unblinded_random_mu > workspace_unblind_random.log &
