import json
import os
import base64
import re

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
img_count = 1

def escape_tex(text):
    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('{', r'\{').replace('}', r'\}')
    text = text.replace('_', r'\_').replace('%', r'\%').replace('$', r'\$').replace('#', r'\#').replace('&', r'\&')
    text = text.replace('^', r'\textasciicircum{}').replace('~', r'\textasciitilde{}')
    return text

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source']).strip()
        if source.startswith("# CPE") or source.startswith("## 6. Summary") or "|" in source:
            continue # Skip title and table markdown cells as they are redundant
        
        # Clean markdown formatting for normal reading
        clean_source = re.sub(r'[*#>`]', '', source)
        tex_lines.append(r"\vspace{0.5em}")
        tex_lines.append(r"\noindent " + escape_tex(clean_source) + r"\\")
        
    elif cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if not source.strip(): continue
        
        # Fix raw strings in source if we update the notebook
        source = source.replace("\hat", r"\\hat").replace("\sqrt", r"\\sqrt")

        tex_lines.append(r"\vspace{1em}")
        tex_lines.append(rf"\noindent\textbf{{In [{cell_num}]:}}")
        tex_lines.append(r"\begin{lstlisting}[style=pycode]")
        tex_lines.append(source.strip())
        tex_lines.append(r"\end{lstlisting}")
        
        outputs = cell.get('outputs', [])
        if outputs:
            has_text = False
            for out in outputs:
                if out['output_type'] == 'stream' or (out['output_type'] == 'execute_result' and 'text/plain' in out.get('data', {})):
                    has_text = True
            
            if has_text:
                tex_lines.append(r"\smallskip")
                tex_lines.append(rf"\noindent\textbf{{Out[{cell_num}]:}}")
                tex_lines.append(r"\begin{lstlisting}[style=output]")
                for out in outputs:
                    if out['output_type'] == 'stream':
                        tex_lines.append("".join(out['text']).strip())
                    elif out['output_type'] == 'execute_result' and 'text/plain' in out.get('data', {}):
                        tex_lines.append("".join(out['data']['text/plain']).strip())
                tex_lines.append(r"\end{lstlisting}")
            
            # Handle images
            for out in outputs:
                data = out.get('data', {})
                if 'image/png' in data:
                    img_data = base64.b64decode(data['image/png'])
                    img_filename = f'nb_out_plot_{img_count}.png'
                    with open(os.path.join(os.path.dirname(out_path), img_filename), 'wb') as img_f:
                        img_f.write(img_data)
                    tex_lines.append(r"\begin{center}")
                    tex_lines.append(rf"\includegraphics[width=0.7\textwidth]{{{img_filename}}}")
                    tex_lines.append(r"\end{center}")
                    img_count += 1

        cell_num += 1

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(tex_lines))

print("notebook_appendix.tex generated with images.")
