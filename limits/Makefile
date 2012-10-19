
default: comb

xml-config: clean-config
	./config/init.py

clean-config:
	rm -f ./config/*.xml

hh-workspace: xml-config
	./hist2workspace.sh hh

lh-workspace: xml-config
	./hist2workspace.sh lh

comb-workspace: xml-config
	./hist2workspace.sh comb

hh-workspace-125: xml-config
	./hist2workspace.sh hh 125

lh-workspace-125: xml-config
	./hist2workspace.sh lh 125

comb-workspace-125: xml-config
	./hist2workspace.sh comb 125

hh: hh-workspace
	./limits.sh hh

lh: lh-workspace
	./limits.sh lh 

comb: comb-workspace
	./limits.sh comb

hh-125: hh-workspace-125
	./limits.sh hh 125

lh-125: lh-workspace-125
	./limits.sh lh 125

comb-125: comb-workspace-125
	./limits.sh comb 125

hh-check:
	./check.sh hh

hh-check-125:
	./check.sh hh 125

clean-hh-check:
	rm -f log_check_hh*

lh-check:
	./check.sh lh

lh-check-125:
	./check.sh lh 125

clean-lh-check:
	rm -f log_check_lh*

comb-check:
	./check.sh comb

comb-check-125:
	./check.sh comb 125

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
