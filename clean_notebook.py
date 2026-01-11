import nbformat

path = "Classifying_Math_Problems.ipynb"
nb = nbformat.read(path, as_version=4)

if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

nbformat.write(nb, path)
print("metadata.widgets removed")

python clean_notebook.py
