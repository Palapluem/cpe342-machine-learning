import json

f = 'Assignment_1_OLS.ipynb'
with open(f, encoding='utf-8') as fp:
    nb = json.load(fp)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i in range(len(source)):
            source[i] = source[i].replace(r"\hat", r"\\hat").replace(r"\sqrt", r"\\sqrt")

with open(f, 'w', encoding='utf-8') as fp:
    json.dump(nb, fp, indent=1)

print("Notebook strings fixed.")
