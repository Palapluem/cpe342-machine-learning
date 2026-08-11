import json

f = 'Assignment_1_OLS.ipynb'
with open(f, encoding='utf-8') as fp:
    nb = json.load(fp)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i in range(len(source)):
            # Replace checkmarks
            source[i] = source[i].replace("✓", "[OK]")
            
            # Find the lollipop plot generation code and update its X-axis to use 'months' (M1-M10) instead of Y_hat
            if "ax2.vlines(Y_hat" in source[i]:
                source[i] = source[i].replace("Y_hat", "months")
            if "ax2.scatter(Y_hat" in source[i]:
                source[i] = source[i].replace("Y_hat", "months")
            if "ax2.axhline(0" in source[i]:
                source[i] = "    ax2.axhline(0, color='#1F2937', linestyle='--', linewidth=1.5)\n"
            if "ax2.annotate" in source[i] and "Y_hat" in source[i]:
                source[i] = source[i].replace("Y_hat[i]", "months[i]")
            if "ax2.set_xlabel" in source[i] and "Fitted Values" in source[i]:
                source[i] = "    ax2.set_xlabel('Observation (Month)', fontproperties=prop_b)\n    ax2.set_xticks(months)\n    ax2.set_xticklabels([f'M{m}' for m in months], fontproperties=prop_r)\n"

with open(f, 'w', encoding='utf-8') as fp:
    json.dump(nb, fp, indent=1)

print("Notebook updated.")
