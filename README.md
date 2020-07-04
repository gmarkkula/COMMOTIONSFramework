This repository contains some sketches and work in progress code for the COMMOTIONS modelling framework. **Please note that this is currently a project-internal repository, not to be shared with others without asking Gustav first**.

* The "*first simple tests*" folder contains an implementation of the "baseline model" (`b1DD-b2SP-b3VD`) described in the "COMMOTIONS modelling: Scope and framework" document (v 2020-05-08) - for pedestrians agents (speed-controlling agents) only. Entry point: the Jupyter notebook.

* The "*acceleration ctrl tests*" folder contains a copy-pasted version of the code in "first simple tests", with further code added to allow simulation also of acceleration-controlling agents (i.e., car-like agents rather than pedestrian-like agents), as well as a combination of one agent controlling speed and one controlling acceleration. Entry point: the Jupyter notebook.

* The "*Golman et al tests*" folder contains basically the same as the publicly shared repository https://github.com/gmarkkula/GolmanEtAlTypeModel, but with some added project-internal tests.

* This root folder contains a work in progress towards a more complete implementation of the framework described in the scoping document. As of 2020-06-02, it implements `b1DD-b2SP-b3VD-oEA-oBEvs-oBEa` for pedestrian (speed-controlling) agents. Entry point: the `test_sc_scenario.ipynb` notebook.

Some things on Gustav's todolist as next steps for the framework implementation:
* ~~Update code to use same mathematical notation as scoping doc.~~
* Extend to allow different agent types, specifically to allow also driver (acceleration-controlling) agents. 
    * Note 2020-07-01: implemented in "acceleration ctrl tests" - but still to be implemented in the framework implementation in the root folder.
    * Note 2020-07-04: Made some progress in the root framework implementation, remaining todo:
        * Use different value functions for agents of different control types
        * Allow for separate $\Delta T$ between agents
        * ...?
* Work on value function term for being on a collision course to consider also the agent's affordances for getting themselves out of the collision course. Some notes in the notebook in "acceleration ctrl tests".
* Extend with functionality for turning model assumptions on and off.
* When multiple other-agent behaviours $b$ correspond to the same accelerations for the other agent, these get double-counted in the current implementation. The solution might be to consider behaviours in an acceleration space, either explicitly by replacing the behaviour grid with an acceleration grid, or implicitly somehow by for ex downweighting behaviours if they are too close to each other in acceleration space.


