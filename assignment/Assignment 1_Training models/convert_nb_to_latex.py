import json
import os

nb_path = r'D:\cpe342-machine-learning\assignment\Assignment 1_Training models\Assignment_1_OLS.ipynb'
out_path = r'D:\cpe342-machine-learning\assignment\Assignment 1_Training models\notebook_appendix.tex'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

tex_lines = []
tex_lines.append(r"% Auto-generated notebook appendix")
tex_lines.append(r"\begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,title=\textbf{Jupyter Notebook: Assignment\_1\_OLS.ipynb}]")
tex_lines.append(r"โค้ดและผลลัพธ์ทั้งหมดด้านล่างเป็นการรันจริงจาก Jupyter Notebook")
tex_lines.append(r"\end{tcolorbox}")
tex_lines.append("")

cell_num = 1
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        # A very simplistic markdown conversion for basic things if needed, but we'll just put it as text.
        tex_lines.append(r"\vspace{0.5em}")
        tex_lines.append(r"\noindent\textbf{Markdown:}")
        tex_lines.append(r"\begin{lstlisting}[style=output]")
        tex_lines.append(source.strip())
        tex_lines.append(r"\end{lstlisting}")
    elif cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if not source.strip(): continue
        
        tex_lines.append(r"\vspace{1em}")
        tex_lines.append(rf"\noindent\textbf{{In [{cell_num}]:}}")
        tex_lines.append(r"\begin{lstlisting}[style=pycode]")
        tex_lines.append(source.strip())
        tex_lines.append(r"\end{lstlisting}")
        
        outputs = cell.get('outputs', [])
        if outputs:
            tex_lines.append(r"\smallskip")
            tex_lines.append(rf"\noindent\textbf{{Out[{cell_num}]:}}")
            tex_lines.append(r"\begin{lstlisting}[style=output]")
            for out in outputs:
                if out['output_type'] == 'stream':
                    tex_lines.append("".join(out['text']).strip())
                elif out['output_type'] == 'execute_result':
                    if 'text/plain' in out['data']:
                        tex_lines.append("".join(out['data']['text/plain']).strip())
            tex_lines.append(r"\end{lstlisting}")
        cell_num += 1

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(tex_lines))

print("notebook_appendix.tex generated.")
