
default: clean

clean-pyc:                                                                      
	find . -name "*.pyc" | xargs rm -f

clean: clean-pyc
	rm -f nohup.out
	rm -rf *.png
