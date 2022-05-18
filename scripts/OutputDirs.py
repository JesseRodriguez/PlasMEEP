import os

cwd = os.getcwd()
outputs = cwd+'/../outputs'
plots = cwd+'/../outputs/plots'
output_dirs = [plots]

if not os.path.isdir(outputs):
    os.mkdir(outputs)

for dir in output_dirs:
    if not os.path.isdir(dir):
        os.mkdir(dir)