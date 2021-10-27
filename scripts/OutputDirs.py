import os

cwd = os.getcwd()
outputs = cwd+'/../outputs'
params = cwd+'/../outputs/params'
plots = cwd+'/../outputs/plots'
run_params = cwd+'/../outputs/run_params'
output_dirs = [params, plots, run_params]

if not os.path.isdir(outputs):
    os.mkdir(outputs)

for dir in output_dirs:
    if not os.path.isdir(dir):
        os.mkdir(dir)