This repository contains some sketches and work in progress code for the COMMOTIONS modelling framework. **Please note that this is currently a project-internal repository, not to be shared with others without asking Gustav first**.

# Current todos in focus toward v0.2

* ~~New architecture for the behaviour estimation functionality (allowing also for assumptions of impact of own actions on behaviour of the other agent)~~
* ~~Improved functions for calculating speeds/accelerations for passing in front / behind another agent, to be used both for estimating behaviour of another agent, and in the intended updated interaction terms in the value functions~~
* ~~Add more advanced logic re the "constant" behaviour - it is still needed when there is no `oBE*` assumption.~~
* ~~Updating the collision course calculations in `sc_scenario.py` to match those in `sc_scenario_helper.py`, i.e., conflict space based rather than based on distances between agent coordinates.~~
* Updated, affordance-based interaction terms in the value functions, and separate values for both potential outcomes. This is ongoing. Currently open issues, in some form of falling order of priority (from 2021-01-16 diary entry, ticking some of them off after later updates):

    * ~~One addition that would be helpful for debugging quite a few of the issues below would be some form of solution for "inspecting" what the model is saying at specific time steps, wrt state predictions, access order accelerations, values etc.~~
    * ~~If both agents start from standstill, they currently don't get going at all...?! (they did in previous commits today 2021-01-16 I believe, but not in this final one...)~~
    * The action values become -inf on and off when different access orders are deemed impossible by the current implementation. In principle this is as intended, but the way it looks currently doesn't seem completely right. (Another possible culprit is `sc_scenario_helper.get_value_of_const_jerk_interval()`, where I am getting `invalid value encountered in double scalars` warnings...)
    * ~~When yielding to (near-)standstill just at the edge of the pedestrian's crossing area, the car sometimes continues sliding into it and just keeps going (see further below for an example). I think the problem here lies in how collisions are treated; basically once the collision is a fact the value function does not care whether the car continues going. One idea might be to allow some sort of tolerance for small incursions into the pedestrian crossing area at very low speeds?~~ [I now instead think this is as it should be as long as there are any oBE assumptions - when the pedestrian stops, the vehicle *should* decide to cross in front of it - and there was never any collision.]
    * ~~Strange estimated behaviour probabilities, whereby both speed(/acceleration) increases and decreases can cause an estimate that the other agent will pass first to change into an estimate that the other agent will pass second. See below for example.~~
    * ~~I have not yet added calculation of delays for waiting time and regaining speed - these are made use of in `SCAgent.get_access_order_values_for_agent_v02()` but are currently just set to zero in `sc_scenario_helper.get_access_order_implications()`.~~
    * The estimation of future acceleration to regain free speed in `sc_scenario_helper.get_access_order_implications()` for acceleration-controlling agents is currently hardcoded to "acceleration needed to reach free speed in 10 s" (the calculation previously used there was sort of ok for v0.1, but probably rather incorrect for v0.2 - and I was seeing weird results from using it).
    * I have added a simple placeholder $\pm$100 saturation for value $V$ of an action - pending proper squashing of the value function as experimented with elsewhere.
    * Might be good to improve the delay discounting of value for slow returns to free speed - see note in 2021-06-01 diary entry, under the "testing hypotheses" heading. A good way to check if any modifications have improved things in practice would be to see if the small deceleration pulse that is noted in that diary entry goes away.
    * Imperfect handling of special cases in `sc_scenario_helper.get_access_order_implications()`:
        * When agent A is in movement within the conflict space and agent B is stationary in front of it, the value estimates for A effectively break - they all go to very large values. The reason is that the current implementation of `sc_scenario_helper.get_access_order_implications()` suggests that the "pass in second" outcome can be achieved by an infinitely long zero acceleration, which looks very valuable given the current approximate way of adding up the value contributions. See the 2021-06-01 notes under "A pedestrian will start..." for a test case.
        * If agent B is reversing, the current implementation of this function will not be able to compute the deceleration for agent A to pass behind B - logically it would make sense to describe it as a stopping deceleration (followed by an infinite waiting time). Check further below under "A pedestrian will start..." for a test case.




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


