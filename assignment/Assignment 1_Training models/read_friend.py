import sys
sys.stdout.reconfigure(encoding='utf-8')
import fitz

pdf_path = r'D:\cpe342-machine-learning\assignment\Assignment 1_Training models\assignment_1_ols_visualize.pdf'
doc = fitz.open(pdf_path)
for i, page in enumerate(doc):
    if i < 9:  # pages 1-9
        print(f'--- Page {i+1} ---')
        print(page.get_text())
