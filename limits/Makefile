
CHANNEL ?= comb
MASS ?=
CONFIG ?=

default: limit

xml-config: clean-config
	./config/render $(CONFIG) $(CHANNEL)

clean-config:
	rm -f ./config/*.xml

workspace: xml-config
	./hist2workspace.sh $(CHANNEL) $(MASS)
	./fix_permissions.sh

limit: workspace
	./limits.sh $(CHANNEL) $(MASS)
	./fix_permissions.sh

check:
	./check.sh $(CHANNEL) $(MASS)
	./fix_permissions.sh

clean-check:
	rm -f log_check_$(CHANNEL)*

clean-limit:
	rm -rf ./results/$(CHANNEL)/*
	rm -rf ./root-files/$(CHANNEL)/*
	rm -f log_$(CHANNEL)_*.txt
	rm -f hist2workspace_$(CHANNEL).log 

clean: clean-limit clean-check
