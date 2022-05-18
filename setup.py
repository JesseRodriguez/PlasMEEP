from setuptools import setup, find_packages

setup(name="plasmeep",
      version="0.0",
      packages=find_packages(),
      description="Plasma-themed wrapper for meep.",
      author="Jesse A. Rodriguez",
      author_email="jrodrig@stanford.edu",
      download_url="https://github.com/JesseRodriguez/PlasMEEP",
      install_requires=[
          'plasmeep',
          'pymeep-extras',
          ]
      )