### PlasMEEP

Library built as a plasma-themed wrapper for Meep (https://meep.readthedocs.io) that speeds up the process of simulating the electromagnetics of simple plasma-based devices and systems
_____

### Quick Tutorial

1. First, use the setup.py file to make sure you have all the dependencies and 'install' PlasMEEP. It would behoove you to do this in a seperate conda environment.
~~~
    (base)$ git clone https://github.com/JesseRodriguez/PlasMEEP
    (base)$ cd PlasMEEP
    (base)$ conda create --name plasmeep python=3.7.10
    (base)$ conda activate plasmeep
    (PMM)$  python setup.py install
    (PMM)$  pip install -e .
~~~

2. Next, get all the output directories ready
~~~
    (PMM)$ cd scripts
    (PMM)$ python OutputDirs.py
~~~

3. Now you're ready. Build and simulate some plasma devices. Run everything with scripts/ as your working directory.
~~~
    (PMM)$ python Bands_MagPlasma.py
~~~
