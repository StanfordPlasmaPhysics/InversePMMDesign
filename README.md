### InversePMMDesign

Library built on top of Ceviche (https://github.com/fancompute/ceviche) tailored for creating plasma metamaterial devices
_____

### Quick Tutorial

1. First, use the setup.py file to make sure you have all the dependencies and 'install' PMM. It would behoove you to do this in a seperate conda environment. _Note: The ceviche field solver uses the PARDISO sparse matrix solver; which is part of Intel's MKL poackage (pyMKL, a requirement for ceviche, is a wrapper for this) and is responsible for the fast parallel performance of the field solvers. If you are running on an HPC system, you will likely need to load an MKL module._
~~~
    (base)$ git clone https://github.com/StanfordPlasmaPhysics/InversePMMDesign
    (base)$ cd InversePMMDesign
    (base)$ conda create --name PMM python=3.7.10
    (base)$ conda activate PMM
    (PMM)$  python setup.py install
    (PMM)$  pip install -e .
~~~

2. Next, get all the output directories ready
~~~
    (PMM)$ cd scripts
    (PMM)$ python OutputDirs.py
~~~

3. Now you're ready. Adjust the resolution and other parameters in the optimizations scripts and create some PMMs. Run everything with scripts/ as your working directory.
~~~
    (PMM)$ python BentWaveguide10x10.py
~~~
