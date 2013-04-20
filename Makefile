
HHSTUDENT ?= HHProcessor
HHNTUP ?= ntuples/prod/HHProcessor
HHNTUP_RUNNING ?= ntuples/running/HHProcessor

default: clean

root-clean:
	rm -f $(HHNTUP)/$(HHSTUDENT).root

h5-clean:
	rm -f $(HHNTUP)/$(HHSTUDENT).h5

ntup-clean: root-clean h5-clean

init-log: $(HHNTUP_RUNNING)/log
	mkdir $(HHNTUP_RUNNING)/log
	mv $(HHNTUP_RUNNING)/*.log $(HHNTUP_RUNNING)/log
	mv $(HHNTUP_RUNNING)/*.e* $(HHNTUP_RUNNING)/log
	mv $(HHNTUP_RUNNING)/*.o* $(HHNTUP_RUNNING)/log

init-data-12: $(HHNTUP_RUNNING)/HHProcessor.data12-JetTauEtmiss.root
	mkdir $(HHNTUP_RUNNING)/data
	mv $(HHNTUP_RUNNING)/HHProcessor.data12-JetTauEtmiss_*.root $(HHNTUP_RUNNING)/data
	hadd $(HHNTUP_RUNNING)/HHProcessor.data12-JetTauEtmiss.root $(HHNTUP_RUNNING)/data/HHProcessor.data12-JetTauEtmiss_*.root

init-embed-12: $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM.root
	mkdir $(HHNTUP_RUNNING)/embed_tes
	mv $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_TES_*.root $(HHNTUP_RUNNING)/embed_tes
	hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_TES_UP.root $(HHNTUP_RUNNING)/embed_tes/HHProcessor.embed12-HH-IM_TES_UP_*.root
	hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM_TES_DOWN.root $(HHNTUP_RUNNING)/embed_tes/HHProcessor.embed12-HH-IM_TES_DOWN_*.root
	
	mkdir $(HHNTUP_RUNNING)/embed
	mv $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-*.root $(HHNTUP_RUNNING)/embed
	hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-IM.root $(HHNTUP_RUNNING)/embed/HHProcessor.embed12-HH-IM_*.root
	hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-UP.root $(HHNTUP_RUNNING)/embed/HHProcessor.embed12-HH-UP_*.root
	hadd $(HHNTUP_RUNNING)/HHProcessor.embed12-HH-DN.root $(HHNTUP_RUNNING)/embed/HHProcessor.embed12-HH-DN_*.root

init-ntup: init-log init-data-12 init-embed-12

$(HHNTUP)/$(HHSTUDENT).root:
	./merge-ntup -s $(HHSTUDENT) $(HHNTUP)

$(HHNTUP)/$(HHSTUDENT).h5: $(HHNTUP)/$(HHSTUDENT).root
	root2hdf5 --complib lzo --complevel 5 --quiet $^

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
	tar -vpczf ~/higgstautau-mva-plots.tar.gz *.png plots/analysis/*.png
