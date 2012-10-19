
default: comb

xml-config:
	./config/init.py

clean-config:
	rm -f ./config/*.xml

hh-workspace: xml-config
	./hist2workspace.sh hh

lh-workspace: xml-config
	./hist2workspace.sh lh

comb-workspace: xml-config
	./hist2workspace.sh comb

hh: hh-workspace
	./limits.sh hh

lh: lh-workspace
	./limits.sh lh 

comb: comb-workspace
	./limits.sh comb

hh-check:
	./check.sh hh

clean-hh-check:
	rm -f log_check_hh*

lh-check:
	./check.sh lh

clean-lh-check:
	rm -f log_check_lh*

comb-check:
	./check.sh comb

clean-comb-check:
	rm -f log_check_comb*

clean-hh:
	rm -rf ./results/hh/*
	rm -rf ./root-files/hh/*
	rm -f log_hh_*.txt
	rm -f hist2workspace_hh.log 

clean-lh:
	rm -rf ./results/lh/*
	rm -rf ./root-files/lh/*
	rm -f log_lh_*.txt
	rm -f hist2workspace_lh.log

clean-comb:
	rm -rf ./results/comb/*
	rm -rf ./root-files/comb/*
	rm -f log_comb_*.txt
	rm -f hist2workspace_comb.log

clean-check: clean-hh-check clean-lh-check clean-comb-check

clean: clean-hh clean-lh clean-comb clean-config clean-check
