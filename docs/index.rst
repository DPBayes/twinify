Welcome to twinify's documentation!
===================================

twinify is a software package for **privacy-preserving generation of a synthetic twin** to a given sensitive tabular **data set.**

On a high level, twinify follows the differentially private data sharing process introduced by `Jälkö et al. <https://arxiv.org/pdf/1912.04439.pdf>`_ Depending on the nature of your data, twinify implements either the NAPSU-MQ approach described by `Räisä et al. <https://arxiv.org/abs/2205.14485>`_ or finds an approximate parameter posterior for any probabilistic model you formulated using differentially private variational inference (DPVI). For the latter, twinify also offers automatic modelling for easy building of models fitting the data. If you have existing experience with NumPyro you can also implement your own model directly.

Documentation
-----------------
.. toctree::
   :maxdepth: 2
   :titlesonly:

   installation
   quickstart
   cli_doc
   api_usage
   examples
   api
   faq
   about_us
   acknowledgements


References
-----------

1. J. Jälkö, E. Lagerspetz, J. Haukka, S. Tarkoma, A. Honkela, and S. Kaski. "Privacy-preserving data sharing via probabilistic modelling". In *Patterns (2021)*, p. 100271.
2. O. Räisä, J. Jälkö, S. Kaski, and A. Honkela. "Noise-Aware Statistical Inference with Differentially Private Synthetic Data". arXiv: 2205.14485. 2022.

BibTeX
------
::

   @article{jalko19,
       title={Privacy-preserving data sharing via probabilistic modelling},
       author={Joonas Jälkö and Eemil Lagerspetz and Jari Haukka and Sasu Tarkoma and Samuel Kaski and Antti Honkela},
       year={2021},
       journal={Patterns},
       volume={2},
       number={7},
       publisher={Elsevier}
   }


   @article{raisa22,
       title={Noise-Aware Statistical Inference with Differentially Private Synthetic Data},
       author={Ossi Räisä and Joonas Jälkö and Samuel Kaski and Antti Honkela},
       year={2022},
       publisher = {arXiv},
       url = {https://arxiv.org/abs/2205.14485}
   }


Versioning
------------

twinify version numbers adhere to `Semantic Versioning <https://semver.org/>`_. Changes
between releases are tracked in ``ChangeLog.txt``.

License
----------

twinify's code base is licensed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

Some files of the accompanying documentation and examples may be licensed differently. You can find an annotation
about which license applies in the beginning of each file using `SPDX <https://spdx.dev/>`_ tags (or in a separate file named ``<file>.license`` for files where
this information cannot be directly embedded).