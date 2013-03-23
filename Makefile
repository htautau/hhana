
HHSTUDENT ?= HHProcessor
HHNTUP ?= ntuples/prod/HHProcessor

default: clean

ntup-clean:
	rm -f $(HHNTUP)/$(HHSTUDENT).root
	rm -f $(HHNTUP)/$(HHSTUDENT).h5

$(HHNTUP)/$(HHSTUDENT).root:
	./merge-ntup -s $(HHSTUDENT) $(HHNTUP)

ntup: $(HHNTUP)/$(HHSTUDENT).root
	root2hdf5 --quiet --script treesplit.py $^

$(HHNTUP)/merged_grl.xml:
	ls $(HHNTUP)/data/*.root | sed 's/$$/:\/lumi/g' | xargs grl or > $@

$(HHNTUP)/observed_grl.xml: $(HHNTUP)/merged_grl.xml ../higgstautau/grl/2012/current.xml 
	grl and $^ > $@

grl: $(HHNTUP)/observed_grl.xml

clean-pyc:                                                                      
	find . -name "*.pyc" | xargs rm -f

clean: clean-pyc
