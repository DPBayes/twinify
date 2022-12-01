import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

# read version number
import importlib
spec = importlib.util.spec_from_file_location("version_module", "twinify/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)

setuptools.setup(
    name='twinify',
    version = version_module.VERSION,
    author="twinify Developers",
    author_email="lukas.m.prediger@aalto.fi",
    description="A software package for privacy-preserving generation of a synthetic twin to a given sensitive data set.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DPBayes/twinify",
    packages=setuptools.find_packages(include=['twinify', 'twinify.*']),
    python_requires='>=3.7',
    install_requires=[
        'pandas >= 1.3.4, < 2.0',
        'd3p >= 0.2.0, < 1.0',
        'tqdm >= 4.62, < 5.0',
        'numpy >= 1.21, < 2.0',
        'graphviz >= 0.20.1, < 1.0.0',
        'arviz >= 0.12.1, < 1.0.0',
        'networkx >= 2.6.0, < 3.0.0',
        'disjoint-set >= 0.7.0, < 1.0.0'
    ],
    extras_require = {
        'examples': [
            'xlrd < 2.0',
            'scikit-learn',
        ],
        'dev': [
            'pytest'
        ],
        'compatible-dependencies': "d3p[compatible-dependencies]",
        'cpu': "d3p[cpu]",
        'cuda': "d3p[cuda]",
    },
    entry_points = {
        'console_scripts': [
            'twinify=twinify.__main__:main',
            'twinify-tools=twinify.tools.__main__:main'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research"
     ],
)
