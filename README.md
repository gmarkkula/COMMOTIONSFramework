This repository contains a Python 3 implementation of a modular framework for modeling road user interactions, developed in the EPSRC-funded project [COMMOTIONS](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/S005056/1). 

* This root folder contains some base modules, `commotions.py` providing some general basics, and `sc_scenario*.py` providing an implementation of the framework for "straight crossing paths" scenarios, with two agents on perpendicular paths.

* The main current entry point to this repository is the "*SCPaper*" folder, which makes use of the modules in this root folder to implement a number of tests of different model variants obtainable from the framework, in vehicle-pedestrian interactions, including comparisons of model predictions to empirical data. This work has been written up as a paper, and the "*SCPaper*" folder contains the code needed to regenerate all of the results and figures in that paper.

* The "*diary notebooks*" folder contains Jupyter notebooks which are effectively diary entries, documenting the work on developing and testing the code in this repository, providing further insight into the motivations behind the many modelling decisions made along the way.

* The "*handwritten notes*" folder contains scans of some mathematical derivations referenced in the code and diary notebooks.
