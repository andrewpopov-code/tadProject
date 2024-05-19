from setuptools import setup, find_packages

setup(
    name="tda-project",
    version="2024.0.0",
    package_dir={'': "src"},
    packages=find_packages("src"),
    python_requires=">=3.9"
)
