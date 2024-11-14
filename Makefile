test:: ; tox -v

build:: ; pip3 install --upgrade build
build:: ; python3 -m build

release:: ; pip3 install --upgrade twine
release:: ; python3 -m twine upload dist/*

clean:: ; rm -f test_*.gif
clean:: ; rm -f test_*.png
clean:: ; rm -rf build
clean:: ; rm -rf dist

cleanall:: clean
cleanall:: ; rm -rf .tox
cleanall:: ; rm -rf .eggs
cleanall:: ; rm -rf kviz.egg-info
cleanall:: ; rm -rf __pycache__
cleanall:: ; rm -rf kviz/__pycache__
cleanall:: ; rm -rf tests/__pycache__
cleanall:: ; rm -rf .coverage
