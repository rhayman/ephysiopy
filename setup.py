import setuptools
import os
from ephysiopy import __about__

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=__about__.__project__,
    version=__about__.__version__,
    author=__about__.__author__,
    author_email="robin.hayman@gmail.com",
    description="Analysis of electrophysiology data",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/rhayman/ephysiopy",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"" : ["*.pkl", "*.txt"]},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn<0.21",
        "astropy",
        "scikit-image",
        "pandas",
        "h5py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
