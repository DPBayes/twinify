import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='twinify',
    version='0.1.2',
    author="twinify Developers",
    author_email="lukas.m.prediger@aalto.fi",
    description="A software package for privacy-preserving generation of a synthetic twin to a given sensitive data set.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DPBayes/twinify",
    packages=setuptools.find_packages(include=['twinify', 'twinify.*']),
    python_requires='>=3.7, <3.9',
    install_requires=[
        'pandas',
        'matplotlib < 3.5',
        'jaxlib >= 0.1.48, <0.1.57',
        'jax >= 0.1.70, <0.1.77',
        'numpyro @ git+https://github.com/pyro-ppl/numpyro.git@b74167942cf0364b21a54f4e14ab6bb3673d18c2#egg=numpyro',
        'dppp @ git+https://github.com/DPBayes/dppp.git@e168eb4201bd460754defe3638404820cd8fe191#egg=dppp',
    ],
    extras_require = {
        'examples': [
            'openpyxl',
            'xlrd',
            'scikit-learn'
        ],
    },
    entry_points = {
        'console_scripts': ['twinify=twinify.__main__:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research"
     ],
)
