

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

clean-pyc:                                                                      
	find . -name "*.pyc" | xargs rm -f

clean: clean-pyc
