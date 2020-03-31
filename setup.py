import setuptools
import os
from ephysiopy import __version__

with open(os.path.join(os.getcwd(), "ephysiopy", "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ephysiopy",
    version=__version__,
    author="Robin Hayman",
    author_email="robin.hayman@gmail.com",
    description="Analysis of electrophysiology data",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "mahotas",
        "h5py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
