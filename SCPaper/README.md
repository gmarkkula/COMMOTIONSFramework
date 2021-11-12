# Working towards a paper on the straight crossing (SC) scenario, with a focus on behaviour estimation

## Next steps

* ~~Get the code in `COMMOTIONSFramework` ready for what is described below:~~
    * ~~Implement the "more constrained general formulation" equations (allowing for model formulations of complexity up to formulation $[3.1]$) - and make sure to closely follow the same mathematical formulations and notation as in this document, for each of the six key parts of the (full) model:~~
        * ~~Accumulated/filtered action value estimates~~
        * ~~Action value estimates given behaviours and actions~~
        * ~~Behaviour probabilities given actions~~
        * ~~Behaviour evidence given actions~~
        * ~~Behaviour evidence from observation of the other agent~~
        * ~~Accumulated/filtered behaviour values given actions~~
        * ~~Momentary behaviour value estimates given actions~~
    * ~~Remove any extraneous stuff~~
    * ~~Implement `oAI` - or rather its absence since it's currently assumed by default.~~
    * ~~Refactor from `V02VALUEFCN` to the optional assumption `oVA`.~~
    * ~~Refactor to no longer assume that the other agent has default parameters - instead assume same as ego parameters - requires changing implementation of `SCAgent` to store `params_k` for both `ctrl_type`s.~~
    * ~~Test and tweak `oBEo` to make sure it is possible to get sensible behaviour out of it.~~
        * ~~Implement acceleration-aware version of the access order implications functions.~~
    * ~~Implement proper value squashing~~
    * ~~Update the `oBEo` and `oBEv` implementations to align with the latest formulations from the 2021-09-10 diary notes.~~    
    * ~~Make sure both types of agents stop as they should if the other agent is static in the middle of their path.~~ (verified for the 2021-09-16 commit, not documented anywhere) 
    * ~~Go through the various targeted phenomena and try to see if my rough expectations for whether the model will be able to achieve them seem to hold.~~ (verified to a reasonable level, except for `oPN`)
* ~~Implement code for model fitting~~
    * ~~Generic parameter search class~~
        * ~~Basic list and grid searching~~
        * ~~Results saving~~
    * ~~Classes for SCPaper specific parameter searching~~
        * ~~Need to add support for fixing Tprime = T and similar stuff~~
* Model expansion/fitting, in some order (see 2021-10-07 diary notes for explanation of some of the planned new assumptions):
    * ~~Implement and test `oVAl` to see how it changes the deterministic fits.~~
    * ~~Run with `oEA`, to get parameterisations that might be possible to carry over straight to probabilistic fitting.~~ (not done, decided against this approach)
    * Implement `oSN*` and `oPF`
        * Implementation in place
        * Verify that it doesn't change previous, deterministic model behaviour
        * Document implementation in this README
        * Make a diary entry showing example results for some various model alternatives. (Also closer look at "flip-flopping" value estimates with the `oSN*oPF` perception?)
    * Implement `oAN` as actual accumulator noise rather than value noise.
    * Get to the point where I am ready to run probabilistic fits (but don't really run them yet):
        * Include also one or more scenarios where both agents are active, for example two encounter scenarios, one with pedestrian priority and one without, to verify correct order of access and absence of collisions.
        * Make sure to include tests both with and without `oPF`, to see if it is need for the "pedestrian hesitation and speedup" phenomenon.
    * Circle back and rerun the deterministic fits, since some of the implementation for the probabilistic fits may have changed these results slightly (see e.g. 2021-11-09 diary notes).
        * Maybe first on my own computer...?
        * ... And then on a faster computer, with an expanded grid?
    * Run the actual probabilistic fits - probably again in multiple stages with expanding grid.
* Test the best model candidates on the Keio or HIKER pedestrian crossing data. (without fitting)
* Optional stuff
    * ~~Parallelisation in `parameter_search.py`?~~ 
    * Add support for agent width and length.




## Model formulations

### Value estimation over time

At the core is estimation of the value of an action $a$ for the ego-agent $\mathrm{A}$ at time step $k$:

$$ 
\hat{V}_{\mathrm{A},a}(k) = \mathscr{F}_{T, \sigma_\mathrm{V}}(\hat{V}_{\mathrm{A},a}, \tilde{V}_{\mathrm{A},a}, k)
$$

where $\mathscr{F}_{T, \sigma_\mathrm{V}}$ is a low-pass filter, with added accumulation noise:

$$
\mathscr{F}_{T, \sigma_\mathrm{V}}(\hat{X}, \tilde{X}, k) \equiv \left( 1 - \frac{\Delta t}{T} \right) \hat{X}(k-1) + \frac{\Delta t}{T} \tilde{X}(k) + \epsilon(k) \sigma_\mathrm{V} \sqrt{\Delta t} 
$$

where $\Delta t$ is the simulation time step, $\epsilon(k) \sim \mathscr{N}(0, 1)$ is Gaussian noise (drawn separately for each separate low-pass filter $\mathscr{F}$ introduced below).

Two optional assumptions defined based on the above:
* `oEA`, evidence accumulation: $T > \Delta t$
* `oAN`, accumulation noise: $\sigma_\mathrm{V} > 0$

### Noisy value estimates

Consider three alternative formulations of $\tilde{V}_{\mathrm{A},a}$:

$$
\begin{align*}
\mathrm{[1]} & & \tilde{V}_{\mathrm{A},a}(k) & \equiv V_\mathrm{A}[\tilde{\mathbf{x}}(k) | a] \\
\mathrm{(2)} & & \tilde{V}_{\mathrm{A},a}(k) & \equiv \sum_b P_{b}(k) V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a, b)] \\
\mathrm{(3)} & & \tilde{V}_{\mathrm{A},a}(k) & \equiv \sum_b P_{b|a}(k) V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a, b)]
\end{align*}
$$

with the $[]$ versus $()$ indicating that a model variant is fully defined versus requires further specification, respectively.

Above, $\tilde{\mathbf{x}}(k)$ is a (possibly noisy) sensory snapshot of the world state at time $k$, and $V_A[\tilde{\mathbf{x}}(k) | \mathscr{C}]$ is the estimated value for agent $A$ of this world state, conditional upon $\mathscr{C}$.


### Estimating the other agent's possible behaviours

The behaviours $b$ of the other agent at time step $k$ are estimated as the accelerations needed to pass either in front of the ego agent or behind it, given the position and speed of both agents at time step $k$, i.e., assuming that the ego agent maintains it current speed. So there is a (sort of) assumption here that the ego agent doesn't assume that the other agent can perceive the ego agent's acceleration.

Model formulation $[1]$ can effectively be obtained as a special case of $(2)$ or $(3)$, by instead assuming that the only possible behaviour $b$ for the other agent is to keep acceleration at zero.


### Affordance-based value function

(For now only describing the affordance-based value function, i.e., with `oVA` enabled, not the original value function.)

The value function $V_X[\tilde{\mathbf{x}}(k) | \mathscr{C}]$ is the estimated value for agent $X$ of this world state, conditional upon $\mathscr{C}$. 

If $\mathscr{C}$ contains an outcome $\Omega \in \{ \Omega_{1st}, \Omega_{2nd} \}$ (passing the conflict space first or second), then the value function is calculated as follows:

1. Estimating the predicted world state $\tilde{\mathbf{x}}'$ a time $T_\mathrm{P}$ into the future, given the current state $\tilde{\mathbf{x}}(k)$ and $\mathscr{C}$, and some combination of 
    * An action $a$ initiated at time $k$ by the ego agent A
    * A behaviour $b$ exhibited by the other agent B from $k$ onward
    * A behaviour $c$ exhibited by the ego agent A from time $k + T_\mathrm{P}/\Delta t$ onward (not 100% sure about this last one - isn't really implemented yet).
2. Calculating the acceleration needed for agent $X$ to achieve outcome $\Omega$, given the position and speed of agent $X$ in $\tilde{\mathbf{x}}'$, and the position, speed, and possibly acceleration of agent $\lnot X$ in $\tilde{\mathbf{x}}'$. Acceleration is considered if `oVAa` is enabled.
3. Calculating the value of the resulting future trajectory of $X$, as described in `COMMOTIONSFramework` diary entry 2021-05-19b (maybe to be described here also later).
4. If `oVAl` is enabled and the agents are on a collision course at $k + T_\mathrm{P}/\Delta t$, also adding a negative looming aversion term:

$$
g_{\dot{\theta}} = 
    \begin{cases}
        - g_\mathrm{free} \frac{  \dot{\theta} - \dot{\theta}_0 }{  \dot{\theta}_1 - \dot{\theta}_0 } & \text{if } \dot{\theta} > \dot{\theta}_0 \\
        0 & \text{otherwise}
    \end{cases}
$$
where $g_\mathrm{free}$ is the value rate of being at one's free speed without any interaction, $\dot{\theta}_0$ is the looming value above which looming starts to be aversive, and $\dot{\theta}_1$ is the looming value at which the magnitude of the negative looming value equals $V_\mathrm{free}$.

If $X = A$ and $\mathscr{C}$ does *not* contain an outcome $\Omega$, then the value estimate is simply the maximum value of the two possible outcomes:
$$
V_A[\tilde{\mathbf{x}}(k) | \mathscr{C}] = \max_{\Omega} V_A[\tilde{\mathbf{x}}(k) | \mathscr{C} \cup \Omega]
$$

If $X = B$ and $\mathscr{C}$ does *not* contain an outcome $\Omega$, then the value estimate is the value given the outcome indicated by the behaviour $b$ (if we are estimating a value for $B$ then there should be a behaviour $b$ in $\mathscr{C}$):

$$
V_B[\tilde{\mathbf{x}}(k) | \mathscr{C}] = \max_{\Omega} V_A[\tilde{\mathbf{x}}(k) | \mathscr{C} \cup \Omega(b)]
$$

As mentioned above, the value function is affected by the following optional assumption:
* `oVAa`, acceleration-aware (affordance-based) value estimation.
* `oVAl`, looming aversion.

### Estimating probabilities of the other agent's possible behaviours

In alternative (2) above, the probability $P_b$ of the other agent's (agent $\mathrm{B}$'s) behaviour is estimated using a softmax function:

$$
P_b(k) = \mathscr{S}\left[ \{A_{b'}\}, b, k \right] \equiv \frac{e^{A_b(k-1)}}{\sum_{b'} e^{A_{b'}(k-1)}}
$$

over the values (at the previous time step, note bene), of the evidences $\{A_{b'}\}$ across the behaviours:

$$
A_b(k) = \beta_\mathrm{V} \hat{A}_{\mathrm{V},b}(k) + \beta_\mathrm{O} \hat{A}_{\mathrm{O},b}(k)
$$

where $\hat{A}_{\mathrm{O},b}$ is evidence accumulated from behaviour observation (see below), and $\hat{A}_{\mathrm{V},b}$ is evidence accumulated from value estimation, as:

$$
\hat{A}_{\mathrm{V},b}(k) = \mathscr{F}_{T', \sigma'_\mathrm{V}}(\hat{A}_{\mathrm{V},b}, \tilde{A}_{\mathrm{V},b}, k)
$$

where one might consider fixing $T' = T$ and $\sigma'_\mathrm{V} = \sigma_\mathrm{V}$.

Two optional assumptions defined based on the above:

* `oBEv`, behaviour estimation based on value estimates: $\beta_\mathrm{V} > 0$ 
* `oBEo`, behaviour estimation based on observation: $\beta_\mathrm{O} = 1$ 

As described in the 2021-09-10 diary notes, we may define a parameter $P_\dag$ (the probability of choosing a maximally negative value action over a neutral value action) from which to derive $\beta_\mathrm{V}$.

---

All of this is exactly the same in alternative (3) above, but the probabilities and evidences now also depend on the own action $a$:

$$
P_{b|a}(k) = \mathscr{S}\left[ \{A_{b'|a}\}, b, k \right] 
$$

$$
A_{b|a}(k) = \beta_\mathrm{V} \hat{A}_{\mathrm{V},b|a}(k) + \beta_\mathrm{O} \hat{A}_{\mathrm{O},b}(k)
$$

$$
\hat{A}_{\mathrm{V},b|a}(k) = \mathscr{F}_{T', \sigma'_\mathrm{V}}(\hat{A}_{\mathrm{V},b|a}, \tilde{A}_{\mathrm{V},b}, k)
$$

---

### Observing the behaviour of the other agent

The evidence for behaviour $b$ accumulated from behaviour observation is updated as:
$$
\hat{A}_{\mathrm{O},b}(k) = \mathscr{O}_{T_\mathrm{Of},T_\mathrm{O1}} \left[ \hat{A}_{\mathrm{O},b}, \tilde{\mathbf{x}}, b, k \right] \equiv \left( 1 - \frac{\Delta t}{T_\mathrm{Of}} \right) \hat{A}_{\mathrm{O},b}(k-1) + \frac{\Delta t}{T_\mathrm{O1}}\ln{p[\tilde{\mathbf{x}}(k) | \tilde{\mathbf{x}}(k-1), b]}
$$

where $T_\mathrm{Of}$ is a "forgetting" time constant, and $T_\mathrm{O1}$ the time needed for the human to "take one sample" (which obviously can be different from the simulation time stamp $\Delta t$). 

The specific formulation above is chosen because with $T_\mathrm{Of} \rightarrow \infty$ and `oBEv` disabled (such that we can fix $\beta_\mathrm{O} = 1$), we get, for $\Delta t = T_\mathrm{O1}$:

$$
\begin{align*}
P_{b|a}(k)  & = \mathscr{S}\left[ \{A_{b'|a}\}, b, k \right] \\
            & = \mathscr{S}\left[ \{\hat{A}_{\mathrm{O},b'}\}, b, k \right] \\
            & = \frac{e^{\hat{A}_{\mathrm{O},b}(k)}}{\sum_{b'} e^{\hat{A}_{\mathrm{O},b'}(k)}} = \{\mathrm{definition\;above} \} \\
            & = \frac{e^{ \hat{A}_{\mathrm{O},b}(k-1)} p[\tilde{\mathbf{x}}(k) | \tilde{\mathbf{x}}(k-1), b] }{\sum_{b'} e^{\hat{A}_{\mathrm{O},b'}(k-1)} p[\tilde{\mathbf{x}}(k) | \tilde{\mathbf{x}}(k-1), b'] } \\
            & = \frac{ \frac{e^{\hat{A}_{\mathrm{O},b}(k-1)}}{\sum_{b''} e^{\hat{A}_{\mathrm{O},b''}(k-1)}} p[\tilde{\mathbf{x}}(k) | \tilde{\mathbf{x}}(k-1), b] }{\sum_{b'} \frac{e^{\hat{A}_{\mathrm{O},b'}(k-1)}}{\sum_{b''} e^{\hat{A}_{\mathrm{O},b''}(k-1)}} p[\tilde{\mathbf{x}}(k) | \tilde{\mathbf{x}}(k-1), b'] } \\
            & = \frac{ P_{b|a}(k-1) \cdot p[\tilde{\mathbf{x}}(k) | \tilde{\mathbf{x}}(k-1), b] }{\sum_{b'} P_{b'|a}(k-1) \cdot  p[\tilde{\mathbf{x}}(k) | \tilde{\mathbf{x}}(k-1), b'] } 
\end{align*}
$$
which is a standard Bayesian update.

If there is no observation noise in the model that helps determine the probability of the observation (the $p[...]$), we can assume a standard deviation $\sigma_\mathrm{O}$ of a normal distribution, as an additional parameter. I have implemented this observation in terms of a longitudinal position, and that should be logically compatible with the optional assumption `oPN` (see further below) also.


### Estimating values of the other agent's possible behaviours

To further specify model formulation (2), we can consider two alternative formulations of $\tilde{A}_{\mathrm{V},b}$, i.e., two different ways of extending model formulation (2) above:

$$
\begin{align*}
\mathrm{[2.1]} & & \tilde{A}_{\mathrm{V},b}(k) & \equiv V_\mathrm{B}[\tilde{\mathbf{x}}(k) | b] \\
\mathrm{(2.2)} & & \tilde{A}_{\mathrm{V},b}(k) & \equiv \sum_c P_{c}(k) V_\mathrm{B}[\tilde{\mathbf{x}}(k) | (b, c)] \\
\end{align*}
$$

where $c$ refers to the possible behaviours (not actions) of the ego agent $\mathrm{A}$, and analogously to above:

$$
P_c(k) = \mathscr{S}\left[ \{A_{c'}\}, c, k \right] 
$$

$$
A_c(k) = \beta_\mathrm{V}' \hat{A}_{\mathrm{V},c}(k) + \beta_\mathrm{O}' \hat{A}_{\mathrm{O},c}(k)
$$

$$
\hat{A}_{\mathrm{V},c}(k) = \mathscr{F}_{T'', \sigma''_\mathrm{V}}(\hat{A}_{\mathrm{V},c}, \tilde{A}_{\mathrm{V},c}, k)
$$

Again, one might consider fixing $T'' = T$ and $\sigma''_\mathrm{V} = \sigma_\mathrm{V}$.

Two optional assumptions based on the above:

* `oBEev`, ego behaviour estimation based on value estimates: $\beta_\mathrm{V}' > 0$
* `oBEeo`, ego behaviour estimation based on self-observation: $\beta_\mathrm{O}' > 0$ 

---

Also here, all of the above can again be used to instead further specify model formulation (3), by introducing the dependency on the own action $a$:

$$
\begin{align*}
\mathrm{[3.1]} & & \tilde{A}_{\mathrm{V},b|a}(k) & \equiv V_\mathrm{B}[\tilde{\mathbf{x}}(k) | (b, a)] \\
\mathrm{(3.2)} & & \tilde{A}_{\mathrm{V},b|a}(k) & \equiv \sum_c P_{c|a}(k) V_\mathrm{B}[\tilde{\mathbf{x}}(k) | (b, a, c)] \\
\end{align*}
$$

$$
P_{c|a}(k) = \mathscr{S}\left[ \{A_{c'|a}\}, c, k \right] 
$$

$$
A_{c|a}(k) = \beta_\mathrm{V}' \hat{A}_{\mathrm{V},c|a}(k) + \beta_\mathrm{O}' \hat{A}_{\mathrm{O},c|a}(k)
$$

$$
\hat{A}_{\mathrm{V},c|a}(k) = \mathscr{F}_{T'', \sigma''_\mathrm{V}}(\hat{A}_{\mathrm{V},c|a}, \tilde{A}_{\mathrm{V},c|a}, k)
$$

---

### Self-observation

In the above formulations, for the $2.x$ model formulations the accumulated evidence for one's own behaviours can be written completely analogously to the previously definied update of evidence for the other agent's behaviours:

$$
\hat{A}_{\mathrm{O},c}(k) = \mathscr{O}_{T'_\mathrm{Of},T'_\mathrm{O1}} \left[ \hat{A}_{\mathrm{O},c}, \tilde{\mathbf{x}}, c, k \right]
$$

and one may choose $T'_\mathrm{Of} = T_\mathrm{Of}$ and $T'_\mathrm{O1} = T_\mathrm{O1}$. 

However, for the $3.x$ model formulations, the observation probability should refer to the future instead, considering one's own action:

$$
\hat{A}_{\mathrm{O},c|a}(k) = \mathscr{O}'_{T_\mathrm{Of},T_\mathrm{Of}} \left[ \hat{A}_{\mathrm{O},c|a}, \tilde{\mathbf{x}}, c, a, k \right] \equiv \left( 1 - \frac{\Delta t}{T_\mathrm{Of}} \right) \hat{A}_{\mathrm{O},c|a}(k-1) + \frac{\Delta t}{T_\mathrm{O1}}\ln{p[\tilde{\mathbf{x}}(k+\frac{T_\mathrm{p}}{\Delta t}|a) | \tilde{\mathbf{x}}(k-1), c]}
$$


### Estimating values of the ego agent's possible behaviours

Consider two alternative formulations of $\tilde{A}_{\mathrm{V},c}$, i.e., two different ways of further specifying model formulation (2.2) above:

$$
\begin{align*}
\mathrm{[2.2.1]} & & \tilde{A}_{\mathrm{V},c}(k) & \equiv V_\mathrm{A}[\tilde{\mathbf{x}}(k) | c] \\
\mathrm{[2.2.2]} & & \tilde{A}_{\mathrm{V},c}(k) & \equiv \sum_b P_{b}(k) V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (c, b)] \\
\end{align*}
$$

Note that model (2.2.2) with `oBEv` and `oBEev` but without `oBEo` and `oBEeo` is an implementation of "my" variant of the Golman et al dual accumulator between the values/evidences $A_b$ and $A_c$.

---

Again, we can instead further specify model formulation (3.2) by including the dependency on own action $a$:

$$
\begin{align*}
\mathrm{[3.2.1]} & & \tilde{A}_{\mathrm{V},c|a}(k) & \equiv V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a, c)] \\
\mathrm{[3.2.2]} & & \tilde{A}_{\mathrm{V},c|a}(k) & \equiv \sum_b P_{b|a}(k) V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a, c, b)] \\
\end{align*}
$$

---

## General formulation

Model formulation $[3.2.2]$ is actually general enough to sort of allow all the other formulations:

$$ 
\hat{V}_{\mathrm{A},a}(k) = \mathscr{F}_{T, \sigma_\mathrm{V}}(\hat{V}_{\mathrm{A},a}, \tilde{V}_{\mathrm{A},a}, k)
$$

$$
\tilde{V}_{\mathrm{A},a}(k) = \sum_b P_{b|a}(k) V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a, b)]
$$

$$
P_{b|a}(k) = \mathscr{S}\left[ \{A_{b'|a}\}, b, k \right] 
$$

$$
A_{b|a}(k) = \beta_\mathrm{V} \hat{A}_{\mathrm{V},b|a}(k) + \beta_\mathrm{O} \hat{A}_{\mathrm{O},b}(k)
$$

$$
\hat{A}_{\mathrm{O},b}(k) = \mathscr{O}_{T_\mathrm{Of},T_\mathrm{O1}} \left[ \hat{A}_{\mathrm{O},b}, \tilde{\mathbf{x}}, b, k \right]
$$

$$
\hat{A}_{\mathrm{V},b|a}(k) = \mathscr{F}_{T', \sigma'_\mathrm{V}}(\hat{A}_{\mathrm{V},b|a}, \tilde{A}_{\mathrm{V},b}, k)
$$

$$ \tilde{A}_{\mathrm{V},b|a}(k) = \sum_c P_{c|a}(k) V_\mathrm{B}[\tilde{\mathbf{x}}(k) | (b, a, c)] 
$$

$$
P_{c|a}(k) = \mathscr{S}\left[ \{A_{c'|a}\}, c, k \right] 
$$

$$
A_{c|a}(k) = \beta_\mathrm{V}' \hat{A}_{\mathrm{V},c|a}(k) + \beta_\mathrm{O}' \hat{A}_{\mathrm{O},c|a}(k)
$$

$$
\hat{A}_{\mathrm{O},c|a}(k) = \mathscr{O}'_{T'_\mathrm{Of},T'_\mathrm{O1}} \left[ \hat{A}_{\mathrm{O},c|a}, \tilde{\mathbf{x}}, c, a, k \right]
$$

$$
\hat{A}_{\mathrm{V},c|a}(k) = \mathscr{F}_{T'', \sigma''_\mathrm{V}}(\hat{A}_{\mathrm{V},c|a}, \tilde{A}_{\mathrm{V},c|a}, k)
$$

$$
\tilde{A}_{\mathrm{V},c|a}(k) = \sum_b P_{b|a}(k) V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a, c, b)]
$$

To use this general set of equations rather than different ones for different model formulations, we can introduce another optional assumption:
* `oAI`, action impact estimation.

If this assumption is not present, then we set, for all $a$: 

$$
V_\mathrm{B}[\tilde{\mathbf{x}}(k) | (b, a, c)] = V_\mathrm{B}[\tilde{\mathbf{x}}(k) | (b, a_0, c)]
$$
$$
V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a, c, b)] = V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a_0, c, b)]
$$
$$
\mathscr{O}'_{T_\mathrm{Of},T_\mathrm{O1}} \left[ \hat{A}_{\mathrm{O},c|a}, \tilde{\mathbf{x}}, c, a, k \right] = \mathscr{O}_{T_\mathrm{Of},T_\mathrm{O1}} \left[ \hat{A}_{\mathrm{O},c|a}, \tilde{\mathbf{x}}, c, k \right]
$$
where $a_0$ is "no action". The above means that these value estimates and self-observation evidences are no longer dependent on the own action $a$ being considered.

We also define the derived assumptions

* `dBE`, behaviour estimation, assumed if `oBEv` and/or `oBEo` are assumed.
* `dBEe`, ego behaviour estimation, assumed if `dBE` is assumed and (`oBEev` and/or `oBEeo` are assumed).

And then if `dBE` is not assumed, we only allow the single behaviour $b =$ constant speed, and if `dBEe` is not assumed, we only allow the single ego behaviour $c =$ constant speed.

We can then achieve all of the model formulations mentioned earlier, except $[2.2.1]$ and $[3.2.1]$ (which is probably not a big loss), as follows:

| Model formulation to achieve | oBEv | oBEo | oBEev | oBEeo | oAI |
|-------------------|------|------|-------|-------|-----|
| $[1]$             |      |      |       |       |     |
| $[2.1]$           |\[R\] |\[R\] |       |       |     |
| $[2.2.2]$         | R    | O    |\[R\]  |\[R\]  |     |
| $[3.1]$           | R    | O    |       |       | R   |
| $[3.2.2]$         | R    | O    |\[R\]  |\[R\]  | R   |

Above, R means required, O optional, \[R\] means at least one required.


## More constrained general formulation

Since I am not planning to get into the `dBEe` stuff in this paper, we can use formulation $[3.1]$ as a simpler and more constrained, but still relatively general formulation:

Accumulated/filtered action value estimates:
$$ 
\hat{V}_{\mathrm{A},a}(k) = \mathscr{F}_{T, \sigma'_\mathrm{V}}(\hat{V}_{\mathrm{A},a}, \tilde{V}_{\mathrm{A},a}, k)
$$

Action value estimates given behaviours and actions:
$$
\tilde{V}_{\mathrm{A},a}(k) = \sum_b P_{b|a}(k) V_\mathrm{A}[\tilde{\mathbf{x}}(k) | (a, b)]
$$

Behaviour probabilities given actions:
$$
P_{b|a}(k) = \mathscr{S}\left[ \{A_{b'|a}\}, b, k \right] 
$$

Behaviour evidence (or "activation") given actions:
$$
A_{b|a}(k) = \beta_\mathrm{V} \hat{A}_{\mathrm{V},b|a}(k) + \beta_\mathrm{O} \hat{A}_{\mathrm{O},b}(k)
$$

Behaviour evidence from observation of the other agent:
$$
\hat{A}_{\mathrm{O},b}(k) = \mathscr{O}_{T_\mathrm{Of},T_\mathrm{O1}} \left[ \hat{A}_{\mathrm{O},b}, \tilde{\mathbf{x}}, b, k \right]
$$

Accumulated/filtered behaviour values given actions:
$$
\hat{A}_{\mathrm{V},b|a}(k) = \mathscr{F}_{T', \sigma'_\mathrm{V}}(\hat{A}_{\mathrm{V},b|a}, \tilde{A}_{\mathrm{V},b}, k)
$$

Behaviour value estimates given actions:
$$
\tilde{A}_{\mathrm{V},b|a}(k) = V_\mathrm{B}[\tilde{\mathbf{x}}(k) | (b, a)] 
$$

As described in the section just above, by introducing `oAI` and `dBE` (but not `dBEe`) we can get the following smaller table of achievable model formulations:

| Model formulation to achieve | oBEv | oBEo | oAI |
|------------------------------|------|------|-----|
| $[1]$                        |      |      |     |
| $[2.1]$                      |\[R\] |\[R\] |     |
| $[3.1]$                      | R    | O    | R   |


## Other optional assumptions, independent of model formulation

### Value function type

In the base model, the value functions $V_X$ are as in (Lin et al., 2021). An alternative formulation to also test is `oVA`, the more advanced affordance-based value functions I have referred to as "v2 value functions".

The value function should also include a term representing priority rules.


### Perception noise

In the base model, the sensory observations $\tilde{\mathbf{x}}$ are noise-free. An alternative formulation to also test is `oPN`, where noise is added based on (an attempt at) perceptually plausible assumptions, by converting the Cartesian representation to optical angle $\theta$ and bearing angle $\phi$, adding constant-intensity noise to these, computing $\dot{\theta}$ and $\dot{\phi}$, then converting back to a Cartesian representation.


## Model variants to target in this paper

Among the large number of possible models implied by the table above, at the moment I think the main types of models to target in the paper should be the six ones implied by the table under "More constrained general formulation" above, all with `oEA` enabled to allow sensible generalisation to variants with noise:

| oBEv | oBEo | oAI | Model formulation  | Description                      | Non-value-function parameters |
|------|------|-----|--------------------|----------------------------------|-------------------------------|
|      |      |     | $[1]$              | Base model                       | $T$                           |
| x    |      |     | $[2.1]$            | Value-based beh. est.            | $T$, $P_\dag$             |
|      | x    |     | $[2.1]$            | Observation-based beh. est.      | $T$, $T_\mathrm{Of}$, $\sigma_\mathrm{O}$ |
| x    | x    |     | $[2.1]$            | Value and observation-based beh. est. | $T$, $P_\dag$, $T_\mathrm{Of}$, $\sigma_\mathrm{O}$|
| x    |      | x   | $[3.1]$            | Action impact est. and value-based beh. est.            | $T$, $P_\dag$                  |
| x    | x    | x   | $[3.1]$            | Action impact est. and value and observation-based beh. est. | $T$, $P_\dag$, $T_\mathrm{Of}$, $\sigma_\mathrm{O}$ |


Each of these six should be tested with all combinations of:
* `oVA` present or not.
* `oAN` or `oPN` (adding another free parameter)

This makes for a total of $6 \times 2 \times 2 = 24$ different models.

The most complex models will have something like seven or eight free parameters (one/two for the value function with/without `oVA`, two for noisy evidence accumulation, and four for behaviour estimation). With `oVA`, the only value function parameter we want to keep free is probably $T_\delta$, whereas without `oVA` we probably need to keep both $k_c$ and $k_{sc}$ free.


## Behaviour phenomena to target in this paper

Overall focus should be straight pedestrian-vehicle interactions.


### By deterministic model fitting, one-sided simulation

My thinking is to narrow down on as many parameters as possible in deterministic, one-sided simulation, to test some basic phenomena:
 
* Fixed-acceleration pedestrian stationary at kerb, simulate vehicle starting from free speed at initial TTA such that the pedestrian stepping out would be a problem for the vehicle:
    * `pAP` - **accelerating to assert priority**: If $V_\nu = 0$, the vehicle reaches a peak speed more than 10 % above free speed.
    * `pSS` - **short-stopping**: if $V_\nu < -V_\mathrm{free}$, the vehicle reaches a peak deceleration more than 10 % larger than needed to stop before the conflict space.
        * Hypothesis: Both require `oBEv`, `oAI`, `oVA`.
* Fixed-acceleration vehicle, simulate pedestrian starting from free speed. Check for **deceleration to gauge uncertain situation**, where the pedestrian's speed drops more than 10 % below free speed...
    * `pDUc` - ... when the vehicle keeps a constant speed, from an initial TTA which would result in a 2 s PET if the pedestrian just walked straight across.
        * Hypothesis: Not achieved by any deterministic model.
    * `pDUd` - ... when the vehicle decelerates to stop before the conflict space, from an initial TTA which would result in a collision at the crossing point if the pedestrian just walked straight across. 
        * Hypothesis: Achieved by all deterministic models.
* Fixed-acceleration vehicle decelerating to stop before the conflict space, using the kinematics from Fig 6 in Jami's paper, simulate pedestrian starting from zero speed at kerb:
    * `pGAd` - **yielding-sensitive gap acceptance**: The pedestrian starts walking before the car has come to a full stop.
        * Hypothesis: Requires `oBEo`

This goes up to six free parameters for the most complex model, so with some quite hard limitations of the grid search, for example:

* $T_\delta \in \{10, 20, 40\}$ s
* $V_0 \in \{5, 10, 20\} \cdot V_\mathrm{free}$
* $T \in \{0, 0.2, 0.4, ..., 1\}$ s
* $T_\mathrm{Of} \in \{0.5, 1, 2, \infty\}$ s
* $P_\dag \in \{0.1, 0.01, 0.001\}$ 
* $\sigma_\mathrm{O} \in$ 5 values 

yielding something $3^3 \times 6 \times 4 \times 5 = 3,240$ parameterisations, given that it seems possible to run the five simulations listed above 1-2 seconds with the current simulation, the most advanced model should not take more than a few hours to test.


### By deterministic model fitting, two-sided simulation

Testing (some of) the best parameterisations from the previous step, to maybe see the phenomena mentioned above in actual interactions, verify that there are no problems with collisions, ... ? To be further specified.


### By probabilistic model fitting 

Then the parameterisations that achieve all of DP1-4 can be further tested by probabilistic simulations with also `oAN` or `oPN`, to test their ability to reasonably fit the Keio dataset:

* PP1: Constant-speed scenarios, collapsed across speed conditions
* PP2: Constant-speed scenarios, not collapsed across speed conditions
* PP3: Deceleration scenarios

The hypothesis here is that `oPN` is required to achieve PP2, and `oBEo` to achieve PP3.

Assuming that we need to run about 30 simulations per scenario, i.e. about 300 (I think) per model parameterisation, it will take about five minutes to test one model parameterisation, so if we sample five noise values then we will need about 25 minutes per parameterisation we bring over from the deterministic tests - if we get lots of "good" parameterisations from there we may need to sample those randomly.


### As follow-up analyses
* Introduce priority rules term in value function, and show how this can cause vehicle yielding
* Then introduce an obstacle after the intersection, and test if this can increase the tendency for the vehicle to yield (Gorrini et al, 2018)
* In simulations where it is initially unclear whether the pedestrian will pass first (due to a limit case of gap acceptance, or due to vehicle yielding), look for an "appraisal phase" (Gorrini et al, 2018) - I think this will require `oBEv` or `oBEo`.
* Assuming that an appraisal phase occurs, check under what circumstances (what initial TTAs) the appraisal phase happens - no direct data on this in (Gorrini et al, 2018), but there are clear common sense expectations
* See if Gorrini et al's (2018) results for speed of older pedestrians can be reproduced by just dialing up $k_\mathrm{dv}$.
* Investigate the effect of increasing/decreasing the gain of the priority rules term.
* Showing that the car can exhibit short stopping with oBEv enabled (see 2021-06-01 diary notes)