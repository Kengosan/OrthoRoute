from setuptools import setup, find_packages

setup(
    name="orthoroute",
    version="0.1.0",
    description="GPU-accelerated PCB autorouter using CUDA",
    author="Brian Benchoff",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "cupy-cuda12x",  # or cupy-cuda11x depending on CUDA version
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
