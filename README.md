This repository contains some sketches and work in progress code for the COMMOTIONS modelling framework. **Please note that this is currently a project-internal repository, not to be shared with others without asking Gustav first**.


# Explanation of repository contents

* The "*first simple tests*" folder contains an implementation of the "baseline model" (`b1DD-b2SP-b3VD`) described in the "COMMOTIONS modelling: Scope and framework" document (v 2020-05-08) - for pedestrians agents (speed-controlling agents) only. Entry point: the Jupyter notebook.

* The "*acceleration ctrl tests*" folder contains a copy-pasted version of the code in "first simple tests", with further code added to allow simulation also of acceleration-controlling agents (i.e., car-like agents rather than pedestrian-like agents), as well as a combination of one agent controlling speed and one controlling acceleration. Entry point: the Jupyter notebook.

* The "*Golman et al tests*" folder contains basically the same as the publicly shared repository https://github.com/gmarkkula/GolmanEtAlTypeModel, but with some added project-internal tests.

* The "*ped Xing tests*" folder contains a copy (made 2020-10-27) of the implementation in the root folder, which has been simplified to make it easier to understand and use as a starting point for simulating pedestrian crossing decisions. Entry point: the Jupyter notebook.

* This root folder contains a work in progress towards a more complete implementation of the framework described in the scoping document. As of 2020-10-15, it implements `b1DD-b2SP-b3VD-oEA-oAN-oBEvs-oBEao` for both pedestrian (speed-controlling) and driver (acceleration-controlling) agents. As of v0.2, 2021-06-01, it does so with updated "affordance-based" value functions that are not fully documented in the COMMOTIONS model scoping document. Entry point: the diary notes mentioned below.

* The "*diary notebooks*" folder contains Jupyter notebooks which document the work on developing and testing the COMMOTIONS framework, including some maths that is not (yet) covered in the model scoping document.


# Todo notes and next steps


* In focus for the next paper:
    * Further thinking about and testing of `oBEao`, to possibly tweak implementation and/or default parameterisation to give more sensible behaviour. See the 2021-06-01 diary notes.
    * Test whether smaller crossing RTs can be obtained for smaller initial TTAs in a pedestrian crossing situation, with v1 value functions --> If yes, consider moving v2 value functions to some sort of parking lot, reverting to v1 functions for the time being.
    * Revert to the previous oBEvs functionality, where the value for the other agent of a behaviour was estimated based on the current state, not the predicted state as a function of my own action.
    * Add priority rules functionality to v1 value functions. 
* Add functionality for resetting/rerunning the same simulation without creating new objects? (relevant now that probabilistic features have been added)
* Allow for separate $\Delta T$ between agents (note that $\Delta T$ is used in multiple places in the code; e.g., also to calculate behaviour accelerations.)
* Extend to allow the user of SCSimulation/SCAgent to also provide the parameters to use when estimating value of a predicted state for another agent (currently the default parameters are used).
* The  updated, affordance-based interaction terms in the "v0.2" value functions, still has some open issues that are probably not critical, but which should be kept in mind, and can be worked on as and when appropriate:
    * The action values become -inf on and off when different access orders are deemed impossible by the current implementation. In principle this is as intended, but the way it looks currently doesn't seem completely right. (Another possible culprit is `sc_scenario_helper.get_value_of_const_jerk_interval()`, where I am getting `invalid value encountered in double scalars` warnings...)
    * The estimation of future acceleration to regain free speed in `sc_scenario_helper.get_access_order_implications()` for acceleration-controlling agents is currently hardcoded to "acceleration needed to reach free speed in 10 s" (the calculation previously used there was sort of ok for v0.1, but probably rather incorrect for v0.2 - and I was seeing weird results from using it).
    * I have added a simple placeholder $\pm$100 saturation for value $V$ of an action - pending proper squashing of the value function as experimented with elsewhere.
    * Might be good to improve the delay discounting of value for slow returns to free speed - see note in 2021-06-01 diary entry, under the "testing hypotheses" heading. A good way to check if any modifications have improved things in practice would be to see if the small deceleration pulse that is noted in that diary entry goes away.
    * Imperfect handling of special cases in `sc_scenario_helper.get_access_order_implications()`:
        * When agent A is in movement within the conflict space and agent B is stationary in front of it, the value estimates for A effectively break - they all go to very large values. The reason is that the current implementation of `sc_scenario_helper.get_access_order_implications()` suggests that the "pass in second" outcome can be achieved by an infinitely long zero acceleration, which looks very valuable given the current approximate way of adding up the value contributions. See the 2021-06-01 notes under "A pedestrian will start..." for a test case.
        * If agent B is reversing, the current implementation of this function will not be able to compute the deceleration for agent A to pass behind B - logically it would make sense to describe it as a stopping deceleration (followed by an infinite waiting time). Check further below under "A pedestrian will start..." for a test case.



