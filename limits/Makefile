
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

lh-check:
	./check.sh lh lephad

all-check:
	./check.sh all combination

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
