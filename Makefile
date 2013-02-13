

HHSTUDENT ?= HHProcessor
HHNTUP ?= ntuples/hadhad/HHProcessor

default: clean

ntup-clean:
	rm -f $(HHNTUP)/$(HHSTUDENT).root
	rm -f $(HHNTUP)/$(HHSTUDENT).h5

ntup-merge: ntup-clean
	./merge-ntup -s $(HHSTUDENT) $(HHNTUP)
	(cd $(HHNTUP) && root2hd5 $(HHSTUDENT).root)

clean-pyc:                                                                      
	find . -name "*.pyc" | xargs rm -f

clean: clean-pyc
