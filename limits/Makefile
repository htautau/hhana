
default: all

hh:
	./hist2workspace.sh hh
	./limits.sh hh hadhad

lh:
	./hist2workspace.sh lh
	./limits.sh lh lephad

all:
	./hist2workspace.sh all
	./limits.sh all combination

hh-check:
	./check.sh hh hadhad

clean-hh-check:
	rm -f log_check_hh*

lh-check:
	./check.sh lh lephad

clean-lh-check:
	rm -f log_check_lh*

all-check:
	./check.sh all combination

clean-all-check:
	rm -f log_check_all*

hh-workspace:
	./hist2workspace.sh hh

lh-workspace:
	./hist2workspace.sh lh

all-workspace:
	./hist2workspace.sh all

clean-hh:
	rm -rf results/hadhad/*
	rm -rf root-files/hadhad/*
	rm -f log_hh_*.txt
	rm -f hist2workspace_hh.log 

clean-lh:
	rm -rf results/lephad/*
	rm -rf root-files/lephad/*
	rm -f log_lh_*.txt
	rm -f hist2workspace_lh.log

clean-all:
	rm -rf results/combination/*
	rm -rf root-files/combination/*
	rm -f log_all_*.txt
	rm -f hist2workspace_all.log

clean-check: clean-hh-check clean-lh-check clean-all-check

clean-config:
	rm -f config/*.xml

clean: clean-hh clean-lh clean-all clean-config clean-check
