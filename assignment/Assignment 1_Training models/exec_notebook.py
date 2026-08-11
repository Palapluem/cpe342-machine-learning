"""Execute the notebook using nbconvert's execute API directly."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_path = r'D:\cpe342-machine-learning\assignment\Assignment 1_Training models\Assignment_1_OLS.ipynb'

with open(nb_path, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=300, kernel_name='python3')
print("Executing notebook... (this may take a moment for plots)")
ep.preprocess(nb, {'metadata': {'path': r'D:\cpe342-machine-learning\assignment\Assignment 1_Training models'}})

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Done. Notebook executed and saved with outputs.")
