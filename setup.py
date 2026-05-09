from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="saann",
    version="0.2.0",
    author="Alessio Branda",
    description="Self-automated Artificial Neural Network: a from-scratch implementation of an ANN with multi-layer perceptron architecture, now with CNN support and GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ale99bra/SaANN.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.0,<3.0",
        "pandas>=3.0,<4.0",
        "matplotlib>=3.10,<4.0",
        "Pillow>=12.0,<13.0",
        "setuptools>=75.0,<76.0",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x>=14.0,<15.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)