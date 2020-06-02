This repository contains some sketches and work in progress code for the COMMOTIONS modelling framework. **Please note that this is currently a project-internal repository, not to be shared with others without asking Gustav first**.

* The "*first simple tests*" folder contains an implementation of the "baseline model" (`b1DD-b2SP-b3VD`) described in the "COMMOTIONS modelling: Scope and framework" document (v 2020-05-08) - for pedestrians agents (speed-controlling agents) only. Entry point: the Jupyter notebook.

* The "*Golman et al tests*" folder contains basically the same as the publicly shared repository https://github.com/gmarkkula/GolmanEtAlTypeModel, but with some added project-internal tests.

* This root folder contains a work in progress towards a more complete implementation of the framework described in the scoping document. As of today's (2020-06-02) commit, it implements `b1DD-b2SP-b3VD-oEA-oBEvs-oBEa` for pedestrian (speed-controlling) agents. Please note that the code and comments have not yet been completely harmonised with the exact mathematical notation used in the scoping document. Entry point: Run `test_scp_model.py`.

Some things on Gustav's todolist as next steps for the framework implementation:
* Update code to use same mathematical notation as scoping doc.
* Extend to allow different agent types, specifically to allow also driver (acceleration-controlling) agents.


