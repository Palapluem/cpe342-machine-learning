# -*- coding: utf-8 -*-
"""
better_convert_nb_2.py
Converts Assignment_2_GD.ipynb → notebook_appendix_2.tex
"""
import json, os, base64, re, unicodedata

nb_path  = r'D:\cpe342-machine-learning\assignment\Assignment 2_Training models via iterative approach\Assignment_2_GD.ipynb'
out_path = r'D:\cpe342-machine-learning\assignment\Assignment 2_Training models via iterative approach\notebook_appendix_2.tex'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

tex_lines = []
tex_lines.append(r"% Auto-generated notebook appendix (Assignment 2)")
tex_lines.append(r"\begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,title=\textbf{Jupyter Notebook: Assignment\_2\_GD.ipynb}]")
tex_lines.append("โค้ด Python และผลลัพธ์ทั้งหมดจาก Jupyter Notebook (รันสมบูรณ์)")
tex_lines.append(r"\end{tcolorbox}")
tex_lines.append("")

def strip_non_ascii_safe(text):
    result = []
    for ch in text:
        code = ord(ch)
        cat  = unicodedata.category(ch)
        if code < 128:
            result.append(ch)
        elif 0x0E00 <= code <= 0x0E7F:   # Thai
            result.append(ch)
        elif cat.startswith('L') or cat.startswith('N') or cat in ('Po', 'Ps', 'Pe', 'Pi', 'Pf', 'Pd'):
            result.append(ch)
        else:
            result.append('-')
    return ''.join(result)

def truncate_long_lines(text, max_chars=100):
    lines = text.split('\n')
    out = []
    for line in lines:
        if len(line) > max_chars:
            out.append(line[:max_chars] + ' ...')
        else:
            out.append(line)
    return '\n'.join(out)

def escape_latex(text):
    math_blocks = []
    def save_math(match):
        math_blocks.append(match.group(0))
        return f"MMMMATH{len(math_blocks)-1}MMMM"

    text = re.sub(r'(?<!\\)\$\$.*?(?<!\\)\$\$', save_math, text, flags=re.DOTALL)
    text = re.sub(r'(?<!\\)\$.*?(?<!\\)\$',     save_math, text, flags=re.DOTALL)

    text = text.replace(r'\$', '$')
    text = text.replace('\\', '\\textbackslash ')
    text = text.replace('%',  r'\%')
    text = text.replace('$',  r'\$')
    text = text.replace('#',  r'\#')
    text = text.replace('_',  r'\_')
    text = text.replace('&',  r'\&')
    text = text.replace('{',  r'\{')
    text = text.replace('}',  r'\}')
    text = text.replace('~',  r'\textasciitilde ')
    text = text.replace('^',  r'\textasciicircum ')
    text = text.replace('α',  r'$\alpha$')
    text = text.replace('β',  r'$\beta$')
    text = text.replace('η',  r'$\eta$')
    text = text.replace('≈',  r'$\approx$')
    text = text.replace('²',  r'${}^2$')

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
        if source:
            tex_lines.append(r"\vspace{1em}")
            tex_lines.append(r"\begin{tcolorbox}[colback=gray!5!white,colframe=gray!50!black,boxrule=0.5pt,arc=2pt,left=2pt,right=2pt,top=2pt,bottom=2pt]")
            in_list = False
            for line in source.split('\n'):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if line_stripped.startswith('- '):
                    if not in_list:
                        tex_lines.append(r"\begin{itemize}")
                        in_list = True
                    tex_lines.append(r"\item " + escape_latex(line_stripped[2:]))
                else:
                    if in_list:
                        tex_lines.append(r"\end{itemize}")
                        in_list = False
                    if line_stripped.startswith('### '):
                        tex_lines.append(r"\textbf{" + escape_latex(line_stripped[4:]) + r"}\par\smallskip")
                    elif line_stripped.startswith('## '):
                        tex_lines.append(r"\textbf{\large " + escape_latex(line_stripped[3:]) + r"}\par\smallskip")
                    elif line_stripped.startswith('# '):
                        tex_lines.append(r"\textbf{\Large " + escape_latex(line_stripped[2:]) + r"}\par\smallskip")
                    else:
                        tex_lines.append(escape_latex(line_stripped))
            if in_list:
                tex_lines.append(r"\end{itemize}")
            tex_lines.append(r"\end{tcolorbox}")

    elif cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if not source.strip():
            continue

        source = source.replace(r"\hat",  r"\\hat")
        source = source.replace(r"\sqrt", r"\\sqrt")
        source = strip_non_ascii_safe(source)

        tex_lines.append(r"\vspace{1em}")
        tex_lines.append(rf"\noindent\textbf{{In [{cell_num}]:}}")
        tex_lines.append(r"\begin{lstlisting}[style=pycode]")
        tex_lines.append(source.strip())
        tex_lines.append(r"\end{lstlisting}")

        outputs = cell.get('outputs', [])
        if outputs:
            has_text = any(
                o['output_type'] == 'stream' or
                (o['output_type'] == 'execute_result' and 'text/plain' in o.get('data', {}))
                for o in outputs
            )
            if has_text:
                tex_lines.append(r"\smallskip")
                tex_lines.append(rf"\noindent\textbf{{Out[{cell_num}]:}}")
                tex_lines.append(r"\begin{lstlisting}[style=output]")
                for out in outputs:
                    if out['output_type'] == 'stream':
                        raw = "".join(out['text']).strip()
                        raw = strip_non_ascii_safe(raw)
                        raw = truncate_long_lines(raw, max_chars=95)
                        tex_lines.append(raw)
                    elif out['output_type'] == 'execute_result' and 'text/plain' in out.get('data', {}):
                        raw = "".join(out['data']['text/plain']).strip()
                        raw = strip_non_ascii_safe(raw)
                        raw = truncate_long_lines(raw, max_chars=95)
                        tex_lines.append(raw)
                tex_lines.append(r"\end{lstlisting}")

            for out in outputs:
                data = out.get('data', {})
                if 'image/png' in data:
                    img_data = base64.b64decode(data['image/png'])
                    img_filename = f'nb2_out_plot_{img_count}.png'
                    with open(os.path.join(os.path.dirname(out_path), img_filename), 'wb') as img_f:
                        img_f.write(img_data)
                    tex_lines.append(r"\begin{center}")
                    tex_lines.append(rf"\includegraphics[width=0.75\textwidth]{{{img_filename}}}")
                    tex_lines.append(r"\end{center}")
                    img_count += 1

        cell_num += 1

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(tex_lines))
