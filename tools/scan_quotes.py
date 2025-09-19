#!/usr/bin/env python3

# safe scanner to report triple-quote balance and ignore CUDA kernel bodies
import re, sys, io

path = sys.argv[1]
txt = io.open(path, 'r', encoding='utf-8', errors='strict').read()

# Detect starts/ends of CUDA kernels so we don't treat their content as Python
kernel_spans = []
for m in re.finditer(r'(?:cp\.RawKernel\s*\(\s*r?("""|\'\'\')|kernel_source\s*=\s*r?("""|\'\'\'))', txt):
    q = m.group(1) or m.group(2)
    start = m.end()
    end = txt.find(q, start)
    if end == -1:
        print(f'UNTERMINATED CUDA literal starting at byte {m.start()}')
        sys.exit(1)
    kernel_spans.append((start, end+len(q)))

def in_kernel(i):
    for a,b in kernel_spans:
        if a <= i < b: return True
    return False

opens = []
for m in re.finditer(r'("""|\'\'\')', txt):
    if in_kernel(m.start()):
        continue
    tok = m.group(1)
    if opens and opens[-1] == tok:
        opens.pop()
    else:
        opens.append(tok)

if opens:
    print('UNBALANCED triple quotes (outside CUDA blocks):', opens)
    sys.exit(1)
print('OK: triple quotes balanced outside CUDA blocks.')