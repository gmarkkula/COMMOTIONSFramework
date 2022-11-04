This folder contains code that uses the modules in the root of this repository to implement and test a range of models of interactions between one car driver and one pedestrian, travelling longitudinally on perpendicular intersecting paths. 

The results of these tests have been written up as a paper with working title "Explaining human interactions on the road requires large-scale integration of psychological theory", preprint available [here](https://doi.org/10.31234/osf.io/hdxbs). The scripts in this folder allow running all tests and generating all figures in the paper. Note that the current version of the preprint was generated based on [release v0.3.1](https://github.com/gmarkkula/COMMOTIONSFramework/releases/tag/v0.3.1) of this repository.

* The scripts `do_#_[...].py` implement the sequence of model analysis steps, generating and saving results (as Pickle `.pkl` files) in the `results` folder, and in some cases also generating and saving supplementary figures in the `figs` folder:
    * `do_2_analyse_deterministic_fits.py` generates Figures S5 and S10-S12 (the code on this master branch will generate a coarser-grid version of Figure S10; for simplicity the finer-grid version was generated on a [separate branch](https://github.com/gmarkkula/COMMOTIONSFramework/tree/finer-grid-deterministic-tests-oVAoBEvoAI)).
    * `do_4_analyse_short_stopping_check.py` generates Figure S6.
    * `do_6_analyse_probabilistic_fits.py` generates Figures S13-S15.
    * `do_9_analyse_combined_fits.py` generates Figure S16.
    * `do_12_analyse_HIKER_fits.py` generates Figure S18.
* The scripts `do_fig#.py` generate the figures provided in the main text of the paper.
* The scripts `do_figS#.py` generate the supplementary figures not generated from the `do_#_[...].py` scripts.
* The scripts `do_#.sh` are for running the more computationally intensive jobs on the University of Leeds [ARC4 HPC cluster](https://arcdocs.leeds.ac.uk/).
* The modules `parameter_search.py` and `sc_fitting.py` provide functionality for running model simulations for a number of given scenarios across a grid of model parameters, and recording various metrics from thes simulations. The module `sc_plot.py` provides some figure-related constants and functionality.
* The `data` folder contains the data from the two controlled experiments described in the paper. 



The `results` folder contains all intermediate results files saved while generating the results presented in the paper. This means that all of the scripts generating figures can be run without first running the entire `do_#_[...].py` sequence of scripts from scratch. Note, however, that some `do_fig#.py` and `do_figS#.py` scripts run simulations for plotting purposes, saving intermediate results files which were too large for including in this repository. This means that these scripts can take a while to run, and in the case of stochastic model variants, relying on random numbers generated at runtime, it also means that the generated figures will not look exactly the same as in the paper.

Some of the scripts use the `multiprocessing` module to run model simulations over multiple cores, which can cause problems in some environments. Typically there is a variable `PARALLEL` that can be set to `False` to run serially instead.

The bulk of the results in this repository and in the paper was generated on the ARC4 cluster mentioned above, running Python 3.7.4, with package versions from the [Anaconda 2019.10 release](https://docs.anaconda.com/anaconda/reference/release-notes/#anaconda-2019-10-october-15-2019) (as described on [this ARC4 page](https://arcdocs.leeds.ac.uk/software/compilers/anaconda.html), accessed 2022-05-26). Some results, and all figures, were generated on the lead author's own computer, running Python 3.9.2, with package versions from the [WinPython 2021-01 release](https://winpython.github.io/); in cases where the same analyses were tested in both places the obtained results were identical between the two.

---

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png
