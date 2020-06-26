# Example: Covid-19 Data Replication with Twinify

An example of using Twinify to replicate results of a classification task to predict SARS-Cov-2 infection in individuals. For a full description and results, see `covid19_example_report.pdf`.

## How to Reproduce

First make sure you have installed
Twinify with `examples` option `pip install twinify[examples]`. You will
also need the `R` language runtime and packages `tidyverse`, `readxl`, `gbm` and `caret`.

To execute all steps to reproduce the example, run
```
bash run_example.sh
```

in your terminal. Be aware that this may take a long time.

## Contents (Files and Folders)

- `data_preprocessing`
  - `download_dataset.py`: Downloads and stores the data set
  - `extract_feature_names.R`: Extracts the features used in the original analysis.
  - `covid19_features.txt`: List of features created by `extract_feature_names.R`.
- `models`
  - `create_model_txt.py`: Creates the `model.txt` for Twinify`s automatic modelling mode.
  - `model.txt`: Created by `create_model_txt.py`.
  - `run_params.txt`: Additional parameters for Twinify invocation
- `twinify_scripts`: Scripts for running all instances of Twinify (different seeds, different privacy levels)
  - `run_twinify.sh`: Script to run Twinify to replicate covid19 example data given a seed and epsilon.
  - `run_twinify_nonprivate.sh`: Script to run Twinify to replicate covid19 example data non-privately given a seed.
  - `seeds_and_eps.txt`: Seed and epsilon values used in the experiments.
  - `run_all.sh`: Runs Twinify for all combinations of seed and privacy levels reported.
  - `run_slurm_batch.sh` and `run_slurm_batch_nonprivate.sh`: Scripts to run Twinify for all combinations of seed and privacy levels leveraging slurm computation clusters, if present. Not used by default in `run_all.sh`.
- `results`: Scripts for analysing the results and producing the plots in the report
  - `extract_train_and_test_set.R`: Splits the original data set in train and test set for classifier training as in the original analysis.
  - `gbm_analysis.py`: Trains GBM classifiers for all data instances.
  - `plot_gbm_results.py`: Produces the report plots based on the results of `gbm_analysis.py`
  - `run_analysis.sh`: Runs all the above scripts in order to performs the full analysis of results.
- `run_example.sh`: Executes all steps to reproduce the full report.

## License

Code in this example is not covered by the Apache License 2.0 under which Twinify itself is published but instead the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. The full license text is available in `LICENSE.txt`.

**Note: The CC BY-NC-SA 4.0 does not apply to the main Twinify code base. It covers only to this example, i.e. everything contained in the `examples/covid19_analysis` folder of the repository with the exception of the `examples/covid19_analysis/slurm_scripts/twinify_nonprivate.py` (covered by the Apache License 2.0).**