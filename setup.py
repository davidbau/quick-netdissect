import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="netdissect",
    version="0.0.2",
    author="David Bau",
    author_email="davidbau@csail.mit.edu",
    description="Network Dissection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidbau/quick-netdissect",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ),
    python_requires=">=3.5.2",
    install_requires=[
        "numpy>=1.14.5",
        "Pillow>=5.1.0",
        "scipy>=1.1.0",
        "torch>=0.4.1",
        "torchvision>=0.2.1",
        "tqdm>=4.23.4",
    ],
)
