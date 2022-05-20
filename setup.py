import setuptools
from src import __version__, __author__, __author_email__

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fo:
    install_requires = fo.read().split('\n')

setuptools.setup(
    name="anomix",
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    description="Mixture Models for anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TRSS-Research/anomix",
    project_urls={},
    entry_points={},
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src") ,
    install_requires=install_requires,
    python_requires=">=3.6",
)
