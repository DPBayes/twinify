# Documentation

### Tooling

Our documentation uses Sphinx Book Theme: https://sphinx-book-theme.readthedocs.io/en/stable. <br>
Configuration for the documentation can be found in `conf.py`. <br>
You define pages in either [reStructuredText](https://docutils.sourceforge.io/rst.html) or in [Markdown](https://www.markdownguide.org/). <br>
You can also use Jupyter notebooks as documentation, like in `examples/NapsuMQ example.ipynb`. You can place them in `examples` folder. `Makefile` or `make.bat` copies `examples` folder under folder `docs/_examples` during building. <br>

### How to build docs locally

Make sure you have `dev` requirements and `Make` installed.

```bash
python -m pip install .[dev]
```

Go to `docs` folder and build documentation. *Optionally* you can clean the documentation before building, this helps if the documentation seems to behave weirdly with left sidebar links etc. 

```bash
cd docs

# optionally
make clean

make github
```

### How to build docs for Github pages

Documentation should update Github pages automatically after `master` branch has a push. **Workflow requires that unit-tests pass.** The workflow is defined in `.github/workflows/docs.yml`.