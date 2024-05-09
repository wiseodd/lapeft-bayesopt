from setuptools import setup

setup(
    name="lapeft_bayesopt",
    version="1.0",
    description="Bayesian optimization with LLMs, PEFT, and the Laplace approximation",
    url="https://github.com/wiseodd/lapeft-bayesopt",
    author="Agustinus Kristiadi",
    author_email="agustinus@kristia.de",
    license="MIT",
    packages=["lapeft_bayesopt"],
    install_requires=[
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pandas",
        "transformers",
        "datasets",
        "peft",
        "tqdm",
        "rdkit",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
