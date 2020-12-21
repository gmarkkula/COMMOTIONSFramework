This repository contains some sketches and work in progress code for the COMMOTIONS modelling framework. **Please note that this is currently a project-internal repository, not to be shared with others without asking Gustav first**.

# Current todos in focus toward v0.2

* ~~New architecture for the behaviour estimation functionality (allowing also for assumptions of impact of own actions on behaviour of the other agent)~~
* ~~Improved functions for calculating speeds/accelerations for passing in front / behind another agent, to be used both for estimating behaviour of another agent, and in the intended updated interaction terms in the value functions~~
* Add more advanced logic re the "constant" behaviour - it is still needed when there is no `oBE*` assumption
* ~~Updating the collision course calculations in `sc_scenario.py` to match those in `sc_scenario_helper.py`, i.e., conflict space based rather than based on distances between agent coordinates.~~
* Updated, affordance-based interaction terms in the value functions, and separate values for both potential outcomes
* Improved consideration of time in the value functions


# Explanation of repository contents

* The "*first simple tests*" folder contains an implementation of the "baseline model" (`b1DD-b2SP-b3VD`) described in the "COMMOTIONS modelling: Scope and framework" document (v 2020-05-08) - for pedestrians agents (speed-controlling agents) only. Entry point: the Jupyter notebook.

* The "*acceleration ctrl tests*" folder contains a copy-pasted version of the code in "first simple tests", with further code added to allow simulation also of acceleration-controlling agents (i.e., car-like agents rather than pedestrian-like agents), as well as a combination of one agent controlling speed and one controlling acceleration. Entry point: the Jupyter notebook.

* The "*Golman et al tests*" folder contains basically the same as the publicly shared repository https://github.com/gmarkkula/GolmanEtAlTypeModel, but with some added project-internal tests.

* The "*ped Xing tests*" folder contains a copy (made 2020-10-27) of the implementation in the root folder, which has been simplified to make it easier to understand and use as a starting point for simulating pedestrian crossing decisions. Entry point: the Jupyter notebook.

* This root folder contains a work in progress towards a more complete implementation of the framework described in the scoping document. As of 2020-10-15, it implements `b1DD-b2SP-b3VD-oEA-oAN-oBEvs-oBEao` for both pedestrian (speed-controlling) and 
driver (acceleration-controlling) agents. Entry point: the `test_sc_scenario.ipynb` notebook.

# Old todo-notes for v0.1, saved for now

Some things on Gustav's todolist as next steps for the framework implementation:
* ~~Update code to use same mathematical notation as scoping doc.~~
* ~~Extend with functionality for turning model assumptions on and off.~~
* ~~Allow user of SCSimulation and SCAgent to set parameters.~~
* ~~Extend to allow different agent types, specifically to allow also driver (acceleration-controlling) agents.~~
* **Testing the optional assumptions implemented so far: Enable them one at a time and check that they behave sensibly.**
    * ~~oEA~~
    * ~~oBEao~~
    * ~~oAN~~
    * oBEvs
* Add functionality for resetting/rerunning the same simulation without creating new objects? (relevant now that probabilistic features have been added)
* Allow for separate $\Delta T$ between agents (note that $\Delta T$ is used in multiple places in the code; e.g., also to calculate behaviour accelerations.)
* Extend to allow the user of SCSimulation/SCAgent to also provide the parameters to use when estimating value of a predicted state for another agent (currently the default parameters are used).
* Work on value function term for being on a collision course to consider also the agent's affordances for getting themselves out of the collision course. Some notes in the notebook in "acceleration ctrl tests".
* When multiple other-agent behaviours $b$ correspond to the same accelerations for the other agent, these get double-counted in the current implementation. The solution might be to consider behaviours in an acceleration space, either explicitly by replacing the behaviour grid with an acceleration grid, or implicitly somehow by for ex downweighting behaviours if they are too close to each other in acceleration space.
* Consider adding back in the option in yielding acceleration estimation to also allow for adapted acceleration where the other agent passes just behind the ego agent - currently (2020-07-08) disabling this because it wasn't clear if it was correctly implemented, and I first needed to figure out the basics of oBEao (which is done now).


