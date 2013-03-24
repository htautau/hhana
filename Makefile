
HHSTUDENT ?= HHProcessor
HHNTUP ?= ntuples/prod/HHProcessor

default: clean

ntup-clean:
	rm -f $(HHNTUP)/$(HHSTUDENT).root
	rm -f $(HHNTUP)/$(HHSTUDENT).h5

$(HHNTUP)/$(HHSTUDENT).root:
	./merge-ntup -s $(HHSTUDENT) $(HHNTUP)

$(HHNTUP)/$(HHSTUDENT).h5: $(HHNTUP)/$(HHSTUDENT).root
	root2hdf5 --complib lzo --complevel 5 --quiet --script treesplit.py $^

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
	rm -f higgstautau-mva-plots.tar.gz
	tar -vpczf higgstautau-mva-plots.tar.gz *.png plots/analysis/*.png
