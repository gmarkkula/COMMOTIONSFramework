This repository contains some sketches and work in progress code for the COMMOTIONS modelling framework. **Please note that this is currently a project-internal repository, not to be shared with others without asking Gustav first**.

* The "*first simple tests*" folder contains an implementation of the "baseline model" (`b1DD-b2SP-b3VD`) described in the "COMMOTIONS modelling: Scope and framework" document (v 2020-05-08) - for pedestrians agents (speed-controlling agents) only. Entry point: the Jupyter notebook.

* The "*acceleration ctrl tests*" folder contains a copy-pasted version of the code in "first simple tests", with further code added to allow simulation also of acceleration-controlling agents (i.e., car-like agents rather than pedestrian-like agents), as well as a combination of one agent controlling speed and one controlling acceleration. Entry point: the Jupyter notebook.

* The "*Golman et al tests*" folder contains basically the same as the publicly shared repository https://github.com/gmarkkula/GolmanEtAlTypeModel, but with some added project-internal tests.

* This root folder contains a work in progress towards a more complete implementation of the framework described in the scoping document. As of 2020-06-02, it implements `b1DD-b2SP-b3VD-oEA-oBEvs-oBEa` for pedestrian (speed-controlling) agents. Entry point: Run `test_scp_model.py`. See also [`notation.ipynb`](notation.ipynb) for some notes complementing the model scoping document.

Some things on Gustav's todolist as next steps for the framework implementation:
* ~~Update code to use same mathematical notation as scoping doc.~~
* Extend to allow different agent types, specifically to allow also driver (acceleration-controlling) agents. (Note 2020-07-01: implemented in "acceleration ctrl tests" - but still to be implemented in the framework implementation in the root folder.)
* Work on value function term for being on a collision course to consider also the agent's affordances for getting themselves out of the collision course. Some notes in the notebook in "acceleration ctrl tests".
* Extend with functionality for turning model assumptions on and off.


