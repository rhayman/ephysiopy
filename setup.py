import setuptools
import ephysiopy.__about__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=ephysiopy.__about__.__project__,
    version=ephysiopy.__about__.__version__,
    author=ephysiopy.__about__.__author__,
    author_email="robin.hayman@gmail.com",
    description="Analysis of electrophysiology data",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/rhayman/ephysiopy",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"": ["*.txt"]},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
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
