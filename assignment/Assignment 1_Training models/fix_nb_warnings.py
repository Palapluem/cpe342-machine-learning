import json
import re

f = 'Assignment_1_OLS.ipynb'
with open(f, encoding='utf-8') as fp:
    nb = json.load(fp)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i in range(len(source)):
            # Remove matplotlib.use('Agg')
            source[i] = re.sub(r"matplotlib\.use\(['\"]Agg['\"]\)", "", source[i])
            # Comment out plt.savefig
            source[i] = re.sub(r"(plt\.savefig\s*\()", r"# \1", source[i])
            # Comment out any print("Saved...")
            source[i] = re.sub(r"(print\s*\(\s*['\"]Saved.*['\"]\s*\))", r"# \1", source[i])

with open(f, 'w', encoding='utf-8') as fp:
    json.dump(nb, fp, indent=1)

print("Notebook code modified for inline display.")
