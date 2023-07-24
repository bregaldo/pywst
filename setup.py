import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywst",
    version="1.0",
    author="Bruno Régaldo-Saint Blancard",
    author_email="bruno.regaldo@phys.ens.fr",
    description="WST and RWST analyses tools for astrophysical data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bregaldo/pywst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
