PROJ_BASE=$(shell pwd)
PYTHONVER=python3.10
PYTHONVENV=$(PROJ_BASE)/venv
VENVPYTHON=$(PYTHONVENV)/bin/$(PYTHONVER)

.PHONY: develop
develop: bootstrap
	@echo "Installing fitdag, with editible modules ('pip install -e . develop')"
	$(VENVPYTHON) -m pip install --editable .
	@echo "\nYou may want to activate the virtual environmnent with 'source venv/bin/activate'\n"

.PHONY: bootstrap
bootstrap:
	@echo "Creating virtual environment 'venv' for development."
	python3 -m virtualenv -p $(PYTHONVER) venv
	@echo "Installing python modules from requirements.txt"
	$(VENVPYTHON) -m pip install --upgrade pip wheel
	$(VENVPYTHON) -m pip install -r requirements.txt

.PHONY: black
black:
	@echo "Formatting with black."
	python3 -m black src/ test/

.PHONY: typecheck
typecheck:
	@echo "Type checking with mypy."
	$(VENVPYTHON) -m mypy src/ test/

.PHONY: test
test:
	@echo "Running test module wit pytest."
	$(VENVPYTHON) -m pytest test/

.PHONY: tight
tight: black typecheck test

.PHONY: clean_build
clean_build:
	@echo "Removing build artifacts"
	rm -rf $(PROJ_BASE)/build
	rm -rf $(PROJ_BASE)/dist
	rm -rf $(PROJ_BASE)/src/*.egg-info
	rm -rf $(PROJ_BASE)/src/**/__pycache__
	rm -rf $(PROJ_BASE)/src/**/version.py

.PHONY: clean
clean: clean_build
	@echo "Removing Python virtual environment 'venv'."
	rm -rf $(PYTHONVENV)

.PHONY: sparkling
sparkling: clean
	rm -rf $(PROJ_BASE)/.mypy_cache
	rm -rf $(PROJ_BASE)/.pytest_cache
