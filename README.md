# Sample Python Research Library

## Running the Code

### Activating the virtual environment

The sample code in this repository has been structured as a research library that can be installed as a python package.
The virtual environment needs to be regenerated, you can use the following steps from the root directory of the project:

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. Install research library: `pip install -e .` (installs current directory to python environment using `setup.py`)

### Running the pipeline

The pipeline for this experiment can be run with:

```bash
python research_library/experiment/lasso/run.py
```

This will generate the data from the order book, create features, and train a lasso regression.
The console output will show what data is being stored at each step. It can all be found in `data/`

### Running tests

I included a `pytest` unit test file with the example from the assessment document. Tests can be run by:

```bash
pytest tests
```
