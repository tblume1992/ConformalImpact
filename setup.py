# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ConformalImpact",
    version="0.0.1",
    author="Tyler Blume",
    url="https://github.com/tblume1992/ConformalImpact",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description = "Confrmal Based Impact Analysis.",
    keywords = ['forecasting', 'time series', 'seasonality', 'trend'],
      install_requires=[           
                        'numpy',
                        'pandas',
                        'tqdm',
                        'numba',
                        'matplotlib',
			'mfles',
                        ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


