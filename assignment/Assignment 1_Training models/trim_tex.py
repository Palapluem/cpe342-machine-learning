import re

path = r'D:\cpe342-machine-learning\assignment\Assignment 1_Training models\CPE342_Assignment 1_main.tex'
with open(path, encoding='utf-8') as f:
    content = f.read()

# Cut everything after \end{document}
cut = content.find(r'\end{document}')
if cut != -1:
    content = content[:cut + len(r'\end{document}')] + '\n'

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"File trimmed. Total lines: {content.count(chr(10))}")
