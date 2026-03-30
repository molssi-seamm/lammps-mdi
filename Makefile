MODULE  := lammps_mdi
PACKAGE := lammps-mdi
SRC     := src/$(MODULE)

# GitHub settings — edit before running 'make git-init'
GITHUB_ORG  := molssi-seamm
GITHUB_REPO := lammps-mdi

.PHONY: help clean clean-build clean-docs clean-pyc clean-test
.PHONY: lint format typing test coverage coverage-html
.PHONY: html docs servedocs
.PHONY: dist check-release release
.PHONY: install uninstall
.PHONY: git-init
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

# -----------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------
clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg'      -exec rm -f  {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc'        -exec rm -f  {} +
	find . -name '*.pyo'        -exec rm -f  {} +
	find . -name '*~'           -exec rm -f  {} +
	find . -name '__pycache__'  -exec rm -fr {} +
	find . -name '_version.py'  -exec rm -f  {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f  .coverage
	rm -fr htmlcov/
	find . -name '.pytype' -exec rm -fr {} +

clean-docs: ## remove Sphinx build artifacts
	rm -f docs/api/$(MODULE).rst
	rm -f docs/api/modules.rst
	$(MAKE) -C docs clean

# -----------------------------------------------------------------------
# Code quality
# -----------------------------------------------------------------------
lint: ## check style with black and flake8 (src layout)
	black --extend-exclude '_version.py' --check --diff $(SRC) tests
	flake8 --color never $(SRC) tests

format: ## reformat with black (src layout)
	black --extend-exclude '_version.py' $(SRC) tests

typing: ## run mypy type checks
	mypy $(SRC)

# -----------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------
test: ## run tests with the default Python
	pytest tests/

coverage: ## check code coverage (terminal report)
	pytest -v --cov=$(MODULE) --cov-report term --color=yes tests/

coverage-html: ## check code coverage (HTML report, opens browser)
	pytest -v --cov=$(MODULE) --cov-report=html:htmlcov --cov-report term --color=yes tests/
	$(BROWSER) htmlcov/index.html

# -----------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------
html: clean-docs ## generate Sphinx HTML documentation including API docs
	sphinx-apidoc -o docs/api $(SRC)
	$(MAKE) -C docs html
	rm -f docs/api/$(MODULE).rst
	rm -f docs/api/modules.rst

docs: html ## build HTML docs and open in browser
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## rebuild docs automatically on .rst changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

# -----------------------------------------------------------------------
# Packaging and release
# -----------------------------------------------------------------------
dist: clean ## build source and wheel packages
	python -m build
	ls -l dist/

check-release: dist ## check the distribution for errors before releasing
	python -m twine check dist/*

release: dist ## build and upload to PyPI
	python -m twine upload dist/*

# -----------------------------------------------------------------------
# Install / uninstall (development)
# -----------------------------------------------------------------------
install: uninstall ## install the package (editable) into the active Python
	pip install -e ".[gpu,dev]"

uninstall: ## uninstall the package
	pip uninstall --yes $(PACKAGE)

# -----------------------------------------------------------------------
# Git and GitHub initialisation
#
# Prerequisites:
#   1. Create the GitHub repository first (empty, no README):
#      https://github.com/organizations/$(GITHUB_ORG)/repositories/new
#      Name: $(GITHUB_REPO)   Visibility: Public
#
#   2. Edit GITHUB_ORG / GITHUB_REPO at the top of this file if needed.
#
#   3. Run:  make git-init
# -----------------------------------------------------------------------
git-init: ## initialise git, create first commit, and push to GitHub
	git init
	git add .
	git commit -m "Initial commit — package skeleton"
	git branch -M main
	git remote add origin git@github.com:$(GITHUB_ORG)/$(GITHUB_REPO).git
	git push -u origin main
	@echo ""
	@echo "Repository pushed to https://github.com/$(GITHUB_ORG)/$(GITHUB_REPO)"
	@echo ""
	@echo "Suggested next steps:"
	@echo "  git checkout -b dev     # create a dev branch for day-to-day work"
	@echo "  git push -u origin dev"
	@echo "  # Then open a PR from dev -> main when ready for a release"
