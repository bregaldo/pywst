import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="pywst",
    version="1.0.1",
    author="Bruno Régaldo-Saint Blancard",
    author_email="bregaldo@flatironinstitute.org",
    description="WST and RWST analyses tools for astrophysical data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bregaldo/pywst",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
