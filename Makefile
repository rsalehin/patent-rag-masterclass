.PHONY: deps test validate validate-one clean

deps:
	pip install -r requirements.txt
	pip install nbclient nbformat jupyter pytest

test:
	pytest tests/ -q

validate:
	python scripts/validate_all.py

# usage: make validate-one CH=06
validate-one:
	python scripts/validate_all.py $(CH)

clean:
	rm -rf artifacts/ notebooks/.ipynb_checkpoints __pycache__ patentrag/__pycache__
