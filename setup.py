from setuptools import setup, find_packages

setup(name="plasmeep",
      version="0.0",
      packages=find_packages(),
      description="Plasma-themed wrapper for meep.",
      author="Jesse A. Rodriguez",
      author_email="jrodrig@stanford.edu",
      download_url="https://github.com/JesseRodriguez/PlasMEEP",
      install_requires=[
          'autograd',
          'h5py',
          'jax',
          'jaxlib',
          'matplotlib',
          'numpy',
          'parameterized',
          'pytest',
          'scipy'
          ]
      )