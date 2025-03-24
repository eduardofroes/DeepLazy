.PHONY: execute-test
execute-test:
	python3 setup.py pytest

.PHONY: build
build:
	python3 setup.py sdist bdist_wheel

.PHONY: install
install:
	pip install -e .

.PHONY: publish
publish:
	python3 -m twine upload dist/*

.PHONY: publish_token
publish_token:
	./publish_token.sh
