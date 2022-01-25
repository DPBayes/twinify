import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

# read version number
import importlib
spec = importlib.util.spec_from_file_location("version_module", "twinify/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)

_available_cuda_versions = ['101', '102', '110', '111']

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
        'pandas',
        'd3p @ git+https://github.com/DPBayes/d3p.git@clipping_bias_mitigation#egg=d3p',
    ],
    extras_require = {
        'examples': [
            'xlrd < 2.0',
            'scikit-learn'
        ],
        'compatible-dependencies': "d3p[compatible-dependencies]",
        'tpu': "d3p[tpu]",
        'cpu': "d3p[cpu]",
        'cuda': "d3p[cuda]", # after numpyro v0.8.0 (and some time after jax v0.2.13)
        **{
            f'cuda{version}': [f'd3p[cuda{version}]']
            for version in _available_cuda_versions
        }
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
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research"
     ],
)
