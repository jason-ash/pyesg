"""Package setup"""
from setuptools import setup, find_packages


with open("README.md") as f:
    README = f.read()

setup(
    name="pyesg",
    version="0.1.4",
    description="Economic Scenario Generator for Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jason-ash/pyesg",
    author="Jason Ash",
    author_email="jason@ashanalytics.com",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy"],
    extras_require={"dev": ["coverage", "hypothesis", "pre-commit", "pylint", "mypy"]},
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
