test:: ; tox -v

clean:: ; rm -f test_*.gif
clean:: ; rm -f test_*.png

cleanall:: clean
cleanall:: ; rm -rf .tox
cleanall:: ; rm -rf .eggs
cleanall:: ; rm -rf kviz.egg-info
cleanall:: ; rm -rf __pycache__
cleanall:: ; rm -rf kviz/__pycache__
cleanall:: ; rm -rf tests/__pycache__
cleanall:: ; rm -rf .coverage
