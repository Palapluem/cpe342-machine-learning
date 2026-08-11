# -*- coding: utf-8 -*-
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
tex_lines.append("โค้ดและผลลัพธ์ทั้งหมดด้านล่างเป็นการรันจริงจาก Jupyter Notebook")
tex_lines.append(r"\end{tcolorbox}")
tex_lines.append("")

def escape_latex(text):
    math_blocks = []
    def save_math(match):
        math_blocks.append(match.group(0))
        return f"MMMMATH{len(math_blocks)-1}MMMM"
    
    text = re.sub(r'(?<!\\)\$\$.*?(?<!\\)\$\$', save_math, text, flags=re.DOTALL)
    text = re.sub(r'(?<!\\)\$.*?(?<!\\)\$', save_math, text, flags=re.DOTALL)
    
    text = text.replace(r'\$', '$')
    
    text = text.replace('\\', '\\textbackslash ')
    text = text.replace('%', r'\%')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('&', r'\&')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('~', r'\textasciitilde ')
    text = text.replace('^', r'\textasciicircum ')
    text = text.replace('α', r'$\alpha$')
    text = text.replace('β', r'$\beta$')
    text = text.replace('≈', r'$\approx$')
    
    text = text.replace('\\textbackslash ', '\\')
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    
    for i, block in enumerate(math_blocks):
        text = text.replace(f"MMMMATH{i}MMMM", block)
        
    return text

cell_num = 1
img_count = 1

for i, cell in enumerate(nb['cells']):
    if i == 0 and cell['cell_type'] == 'markdown':
        continue
        
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source']).strip()
        if "6. Summary of Results" in source:
            continue
            
        if source:
            tex_lines.append(r"\vspace{1em}")
            tex_lines.append(r"\begin{tcolorbox}[colback=gray!5!white,colframe=gray!50!black,boxrule=0.5pt,arc=2pt,left=2pt,right=2pt,top=2pt,bottom=2pt]")
            
            in_list = False
            for line in source.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('- '):
                    if not in_list:
                        tex_lines.append(r"\begin{itemize}")
                        in_list = True
                    tex_lines.append(r"\item " + escape_latex(line[2:]))
                else:
                    if in_list:
                        tex_lines.append(r"\end{itemize}")
                        in_list = False
                    
                    if line.startswith('### '):
                        tex_lines.append(r"\textbf{" + escape_latex(line[4:]) + r"}\par\smallskip")
                    elif line.startswith('## '):
                        tex_lines.append(r"\textbf{\large " + escape_latex(line[3:]) + r"}\par\smallskip")
                    elif line.startswith('# '):
                        tex_lines.append(r"\textbf{\Large " + escape_latex(line[2:]) + r"}\par\smallskip")
                    else:
                        tex_lines.append(escape_latex(line) + r"\\")
            if in_list:
                tex_lines.append(r"\end{itemize}")
                
            tex_lines.append(r"\end{tcolorbox}")
        
    elif cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if not source.strip(): continue
        
        # fix warning in python by using raw string
        source = source.replace(r"\hat", r"\\hat").replace(r"\sqrt", r"\\sqrt")

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
            
            for out in outputs:
                data = out.get('data', {})
                if 'image/png' in data:
                    img_data = base64.b64decode(data['image/png'])
                    img_filename = f'nb_out_plot_{img_count}.png'
                    with open(os.path.join(os.path.dirname(out_path), img_filename), 'wb') as img_f:
                        img_f.write(img_data)
                    tex_lines.append(r"\begin{center}")
                    tex_lines.append(rf"\includegraphics[width=0.6\textwidth]{{{img_filename}}}")
                    tex_lines.append(r"\end{center}")
                    img_count += 1

        cell_num += 1

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(tex_lines))

print("notebook_appendix.tex generated with proper markdown to latex.")
