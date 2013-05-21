
HHSTUDENT ?= HHProcessor
HHNTUP ?= ntuples/prod/HHProcessor
HHNTUP_RUNNING ?= ntuples/running/HHProcessor

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

$(HHNTUP_RUNNING)/HHProcessor.data12-JetTauEtmiss.root:
	test -d $(HHNTUP_RUNNING)/data || mkdir $(HHNTUP_RUNNING)/data
	mv $(HHNTUP_RUNNING)/HHProcessor.data12-JetTauEtmiss_*.root $(HHNTUP_RUNNING)/data
	
	hadd $(HHNTUP_RUNNING)/HHProcessor.data12-JetTauEtmiss.root $(HHNTUP_RUNNING)/data/HHProcessor.data12-JetTauEtmiss_*.root
	
	test -d $(HHNTUP_RUNNING)/data_log || mkdir $(HHNTUP_RUNNING)/data_log
	-mv $(HHNTUP_RUNNING)/HHProcessor.data12_*.e[1-9]* $(HHNTUP_RUNNING)/data_log/
	-mv $(HHNTUP_RUNNING)/HHProcessor.data12_*.o[1-9]* $(HHNTUP_RUNNING)/data_log/
	-mv $(HHNTUP_RUNNING)/supervisor-HHProcessor-HHProcessor.data12-JetTauEtmiss_*.log $(HHNTUP_RUNNING)/data_log/


init-data-12: $(HHNTUP_RUNNING)/HHProcessor.data12-JetTauEtmiss.root

$(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM.root:
	test -d $(HHNTUP_RUNNING)/embed || mkdir $(HHNTUP_RUNNING)/embed
	if [ -f $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_TES_UP_1.root ]; then \
		test -d $(HHNTUP_RUNNING)/embed_tes || mkdir $(HHNTUP_RUNNING)/embed_tes; \
		mv $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_TES_*.root $(HHNTUP_RUNNING)/embed_tes; \
		hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_TES_UP.root $(HHNTUP_RUNNING)/embed_tes/HHProcessor.embed12-HH-IM_TES_UP_*.root; \
		hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_TES_DOWN.root $(HHNTUP_RUNNING)/embed_tes/HHProcessor.embed12-HH-IM_TES_DOWN_*.root; \
	fi
	if [ -f $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_1.root ]; then \
		mv $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_*.root $(HHNTUP_RUNNING)/embed; \
		hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM.root $(HHNTUP_RUNNING)/embed/HHProcessor.embed12-HH-IM_*.root; \
	fi
	if [ -f $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-UP_1.root ]; then \
		mv $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-UP_*.root $(HHNTUP_RUNNING)/embed; \
		hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-UP.root $(HHNTUP_RUNNING)/embed/HHProcessor.embed12-HH-UP_*.root; \
	fi
	if [ -f $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-DN_1.root ]; then \
		mv $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-DN_*.root $(HHNTUP_RUNNING)/embed; \
		hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-DN.root $(HHNTUP_RUNNING)/embed/HHProcessor.embed12-HH-DN_*.root; \
	fi
	test -d $(HHNTUP_RUNNING)/embed_log || mkdir $(HHNTUP_RUNNING)/embed_log
	-mv $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-*.e[1-9]* $(HHNTUP_RUNNING)/embed_log/
	-mv $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-*.o[1-9]* $(HHNTUP_RUNNING)/embed_log/
	-mv $(HHNTUP_RUNNING)/supervisor-HHProcessor-HHProcessor.embed12-HH-*.log $(HHNTUP_RUNNING)/embed_log/

init-embed-12: $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM.root

init-ntup: init-data-12 init-embed-12

$(HHNTUP)/$(HHSTUDENT).root:
	./merge-ntup -s $(HHSTUDENT) $(HHNTUP)

$(HHNTUP)/$(HHSTUDENT).h5: $(HHNTUP)/$(HHSTUDENT).root
	root2hdf5 --complib lzo --complevel 0 --quiet $^

ntup: $(HHNTUP)/$(HHSTUDENT).h5

$(HHNTUP)/merged_grl.xml:
	ls $(HHNTUP)/data/*.root | sed 's/$$/:\/lumi/g' | xargs grl or > $@

$(HHNTUP)/observed_grl.xml: $(HHNTUP)/merged_grl.xml ../higgstautau/grl/2012/current.xml 
	grl and $^ > $@

~/observed_grl.xml: $(HHNTUP)/observed_grl.xml
	cp $^ $@

grl: ~/observed_grl.xml

clean-grl:
	rm -f $(HHNTUP)/observed_grl.xml
	rm -f ~/observed_grl.xml
	rm -f $(HHNTUP)/merged_grl.xml

clean-pyc:                                                                      
	find . -name "*.pyc" | xargs rm -f

clean: clean-pyc


bundle:
	rm -f ~/higgstautau-mva-plots.tar.gz
	tar -vpczf ~/higgstautau-mva-plots.tar.gz plots/*.png

test:
	nosetests -s -v mva

dump:
	@./dump -t higgstautauhh -s "taus_pass && (RunNumber==207528)" --select-file embed_select_ac.txt -o RunNumber,EventNumber $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM.root
	@./dump -t higgstautauhh -e 50 -s "taus_pass" -o EventNumber $(HHNTUP_RUNNING)/HHProcessor.AlpgenJimmy_AUET2CTEQ6L1_ZtautauNp4.mc12a.root
	@./dump -t higgstautauhh -e 50 -s "taus_pass" -o EventNumber $(HHNTUP_RUNNING)/HHProcessor.AlpgenJimmy_AUET2CTEQ6L1_ZtautauNp0.mc12a.root
