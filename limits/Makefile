
default: limits-all

limits-hh:
	./hist2workspace-hh.sh
	./limits-hh.sh

limits-lh:
	./hist2workspace-lh.sh
	./limits-lh.sh

limits-all:
	./hist2workspace-all.sh
	./limits-all.sh

clean-hh:
	rm -rf results/hadhad/*
	rm -rf root-files/hadhad/*
	rm -f log*.txt

clean-lh:
	rm -rf results/lephad/*
	rm -rf root-files/lephad/*
	rm -f log*.txt

clean-all:
	clean-lh
	clean-hh
