.PHONY : doc
doc :
	mkdir -p _build/docsource
	cp index.rst _build/docsource/
	sphinx-apidoc -f -o _build/docsource oscopetools **/test_*.py
	sphinx-build _build/docsource _build/html -c .
	open _build/html/index.html

.PHONY : clean
clean :
	rm -rf _build/*
