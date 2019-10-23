"""Package setup"""
from setuptools import setup, find_packages

setup(
    name="pyesg",
    version="0.1",
    description="Economic Scenario Generator in Python",
    long_description=open("README.md").read(),
    url="https://github.com/jason-ash/pyesg",
    author="Jason Ash",
    author_email="jason@ashanalytics.com",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy"],
    extras_require={"dev": ["pre-commit"]},
    include_package_data=True,
    test_suite="tests",
    zip_safe=False,
)
