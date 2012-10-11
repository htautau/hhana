
default: limits-all

limits-hh:
	./hist2workspace.sh hh
	./limits.sh hh hadhad

limits-lh:
	./hist2workspace.sh lh
	./limits.sh lh lephad

limits-all:
	./hist2workspace.sh all
	./limits.sh all combination

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
