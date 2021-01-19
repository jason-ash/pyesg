"""Package setup"""
from setuptools import setup, find_packages


with open("pyesg/version.py") as f:
    __version__ = None
    exec(f.read(), globals())  # pylint: disable=exec-used


with open("README.md") as f:
    README = f.read()

setup(
    name="pyesg",
    version=__version__,
    description="Economic Scenario Generator for Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jason-ash/pyesg",
    author="Jason Ash",
    author_email="jason@ashanalytics.com",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy"],
    extras_require={
        "dev": [
            "black==19.10b0",
            "coverage==5.2.1",
            "hypothesis==5.24.4",
            "pre-commit==2.6.0",
            "pylint==2.6.0",
            "mypy==0.782",
        ]
    },
    include_package_data=True,
    package_data={
        "pyesg": ["../README.md", "../LICENSE.md", "../MANIFEST.in", "datasets/*"]
    },
    python_requires=">=3.6",  # f-strings
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    zip_safe=False,
)
