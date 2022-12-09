### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 746961b0-f4b6-11ea-3289-03b36dffbea7
begin
	using PlutoUI

	md"""
	# Spacecraft Collision Avoidance
	##### A Partially Observable Markov Decision Process

	A model of Spacecraft Collision Avoidance using POMDPs.jl
	"""
end

# ╔═╡ d0a58780-f4d2-11ea-155d-f55c848f91a8
using POMDPs, QuickPOMDPs, POMDPModelTools, BeliefUpdaters, Parameters

# ╔═╡ fd7f872d-7ef2-4987-af96-4ca4573f29fc
using POMDPPolicies

# ╔═╡ 17bbb35d-b74d-47a7-8349-904338127977
using QMDP

# ╔═╡ 9cdc9132-f524-11ea-2051-41beccfeb0e4
using FIB

# ╔═╡ b0f1a551-9eec-4b7d-8fa5-afcbc6a86cd9
using PointBasedValueIteration

# ╔═╡ df3fd98d-40df-4850-8410-992610bbad10
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

# ╔═╡ 315ede12-f60d-11ea-076e-e1b8b460aa9e
using BasicPOMCP

# ╔═╡ 6dc0ddc0-f60f-11ea-2a57-158b8a68be4e
using D3Trees

# ╔═╡ a88c0bf0-f4c0-11ea-0e61-853ac9a0c0cb
md"## 1. Partially Observable MDP (POMDP)"

# ╔═╡ 32c56c10-f4d2-11ea-3c79-3dc8b852c182
md"""
A partially observable Markov decision process (POMDP) is a 7-tuple consisting of:

$\langle \mathcal{S}, \mathcal{A}, {\color{blue}\mathcal{O}}, T, R, {\color{blue}O}, \gamma \rangle$

Variable           | Description          | `POMDPs` Interface
:----------------- | :------------------- | :-------------------:
$\mathcal{S}$      | State space          | `POMDPs.states`
$\mathcal{A}$      | Action space         | `POMDPs.actions`
$\mathcal{O}$      | Observation space    | `POMDPs.observations`
$T$                | Transition function  | `POMDPs.transision`
$R$                | Reward function      | `POMDPs.reward`
$O$                | Observation function | `POMDPs.observation`
$\gamma \in [0,1]$ | Discount factor      | `POMDPs.discount`

Notice the addition of the observation space $\mathcal{O}$ and observation function $O$, which differ from the MDP 5-tuple.

Indeed, the agent receives _observations_ of the current state rather than the true state—and using past observations, it builds a _belief_ of the underlying state (this can be represented as a probability distribution over true states).
"""

# ╔═╡ a13e36e0-f4d2-11ea-28cf-d18a43e34c3e
md"### 1.1 Set of State variables $\mathcal{S}$
The *state* set in our problem is composed of both $satellite$ and $debris$ respective state *variables*.

**Orbit location $satellite$**

-  $a_{sat}$: size of semi-major axis of $orbit_{sat}$
-  $e_{sat}$: eccentricity of $orbit_{sat}$
-  $\omega_{sat}:$ argument of $orbit_{sat}$ (angle from the ascending node radius to the periapsis radius in a counter clockwise direction)
-  $\nu_{sat}:$ position of $orbit_{sat}$ (angle from the periapsis radius to the object)

⇒ gives the **position of the satellite** on a **2D orbit plane**

**Orbit location $debris$**

-  $a_{deb}$: size of semi-major axis of $orbit_{deb}$
-  $e_{deb}$: eccentricity of $orbit_{deb}$
-  $\omega_{deb}:$ argument of $orbit_{deb}$ (angle from the ascending node radius to the periapsis radius in a counter clockwise direction)
-  $\nu_{deb}:$ position of $orbit_{deb}$ (angle from the periapsis radius to the object

⇒ gives the **position of the debris** on a **2D orbit plane**

**Speed**

-  $V_{sat}$
-  $V_{deb}$

**Advisory**

-  $s_{adv}$: current advisory

**Other parameters**

-  $s_{fuel}$: fuel level

$$\begin{align}
\mathcal{S_{sat}} &= \{\ a_{sat},e_{sat},\omega_{sat},\nu_{sat},V_{sat},s_{adv},s_{fuel}\}\\
\mathcal{S_{deb}} &= \{\ a_{deb},e_{deb},\omega_{deb},\nu_{deb},V_{deb}\}
\end{align}$$
"

# ╔═╡ dbc77e50-f529-11ea-0d79-71196165ac17


# ╔═╡ 222c7568-d9b3-4148-9800-c42372969c14
md"""### 1.2 Set of Actions $\mathcal{A}$
The *actions* set in our problem is composed of clear $advisories$ to be given as **instructions** to the satellite.

$$\begin{align}
\mathcal{A} = \{\rm &Clear\ of\ Conflict,\\
&Monitor\ Speed\ V\,\\
&Accelerate,\\
&Decelerate\}\\

\end{align}$$
"""

# ╔═╡ 33b27b26-2b32-4620-9212-262fb30fcbbd
md"## 2. Julia Model"

# ╔═╡ 67a7ce56-5cff-432c-96a2-08098bb8af44
md"""
### 2.1 Parameters
We define *hardcoded* parameters to be tweaked.
"""

# ╔═╡ bd6724d1-3067-4285-9ee0-c0b1363e8243
@with_kw struct SpacecraftCollisionAvoidanceParameters
	# Rewards
	r_danger::Real = -10
	r_accelerate::Real = -5
	r_decelerate::Real = -5
	
	# Transition probability
	p_to_collide::Real = 0.1

	# Observation probabilities
	p_CDM_when_safe::Real = 0.1
	p_CDM_when_danger::Real = 0.9
end

# ╔═╡ 647c5372-a632-42ae-ab32-ea6ad491b0b6
params = SpacecraftCollisionAvoidanceParameters();

# ╔═╡ b2a53c8e-f4db-11ea-08ba-67b9158f39b3
md"""
We get the sparse categorical distribution (`SparseCat`) from `POMDPModelTools`, the `PreviousObservationUpdater` from `BeliefUpdaters`, and `@with_kw` from `Parameters`.
"""

# ╔═╡ be1258b0-f4db-11ea-390e-2bcc849111d0
md"""
### 2.2 State, Action, and Observation Spaces
We define enumerations for our states, actions, and observations using Julia's built-in `@enum`.
"""

# ╔═╡ f49ffe90-f4dc-11ea-1ecb-9d6e6e66d3d4
begin
	@enum State SAFEₛ DANGERₛ
	@enum Action CLEARofCONFLICTₐ ACCELERATEₐ DECELERATEₐ
	@enum Observation NoCDMₒ CDMₒ
end

# ╔═╡ 9df137d0-f61c-11ea-0dd6-67535f3b0d52
md"We define our state, action, and observation *spaces*."

# ╔═╡ c720f8a0-f61e-11ea-155d-c13361437a85
md"##### State Space"

# ╔═╡ b03708f0-f61e-11ea-38c8-5945da744bff
𝒮 = [SAFEₛ, DANGERₛ]

# ╔═╡ ce359010-f61e-11ea-2f71-a1fc0b6d5300
md"##### Action Space"

# ╔═╡ e97a7a20-f4d9-11ea-0aca-659f1ede1fd9
𝒜 = [CLEARofCONFLICTₐ, ACCELERATEₐ, DECELERATEₐ, ]

# ╔═╡ d1b6ee9e-f61e-11ea-0619-d13585355550
md"##### Observation Space"

# ╔═╡ f4427ca2-f4d9-11ea-2822-e16314168c58
𝒪 = [NoCDMₒ, CDMₒ]

# ╔═╡ 2e6aff30-f61d-11ea-1f71-bb0a7c3aad2e
md"""
##### Initial State
For our initial state distribution, the baby is deterministically full (i.e. not hungry).
"""

# ╔═╡ 02937ed0-f4da-11ea-1f82-cb56e99e5e20
initialstate_distr = Deterministic(SAFEₛ);

# ╔═╡ eb932850-f4d6-11ea-3102-cbbf0e9d8189
md"""
### 2.3 Transition Function
The transition dynamics are $T(s^\prime \mid s, a)$:

$$\begin{align}
T(\rm danger \mid danger, accelerate) &= 0\%\\
T(\rm safe \mid safe, accelerate) &= 100\%
\end{align}$$

$$\begin{align}
T(\rm danger \mid safe, accelerate) &= 0\%\\
T(\rm safe \mid safe, accelerate) &= 100\%
\end{align}$$

$$\begin{align}
T(\rm danger \mid danger, decelerate) &= 0\%\\
T(\rm safe \mid safe, decelerate) &= 100\%
\end{align}$$

$$\begin{align}
T(\rm danger \mid safe, decelerate) &= 0\%\\
T(\rm safe \mid safe, decelerate) &= 100\%
\end{align}$$

$$\begin{align}
T(\rm danger \mid danger, clearofconflict)&= 100\%\\
T(\rm safe \mid danger, clearofconflict)&= 0\%\\
\end{align}$$

$$\begin{align}
T(\rm danger \mid safe, clearofconflict) &= 10\%\\
T(\rm safe \mid safe, clearofconflict) &= 90\%
\end{align}$$

Note we include the implied complements for completeness.
"""

# ╔═╡ 3d57d840-f4d5-11ea-2744-c3e456949d67
function T(s::State, a::Action)
	p_h::Real = params.p_to_collide

	if a == ACCELERATEₐ
		return SparseCat([DANGERₛ, SAFEₛ], [0, 1])
		
	elseif a == DECELERATEₐ
		return SparseCat([DANGERₛ, SAFEₛ], [0, 1])
		
	elseif s == DANGERₛ && a == CLEARofCONFLICTₐ
		return SparseCat([DANGERₛ, SAFEₛ], [1, 0])
		
	elseif s == SAFEₛ && a == CLEARofCONFLICTₐ
		return SparseCat([DANGERₛ, SAFEₛ], [p_h, 1-p_h])
	end
end

# ╔═╡ d00d9b00-f4d7-11ea-3a5c-fdad48fabf71
md"""
### 2.4 Observation Function
The observation function, or observation model, $O(o \mid s^\prime)$ is given by:

$$\begin{align}
O(\rm CDM \mid danger) &= 90\%\\
O(\rm NoCDM \mid danger) &= 10\%
\end{align}$$

$$\begin{align}
O(\rm CDM \mid safe) &= 10\%\\
O(\rm NoCDM \mid safe) &= 90\%
\end{align}$$
"""

# ╔═╡ 61655130-f4d6-11ea-3aaf-53233c68b6a5
function O(s::State, a::Action, s′::State)
	if s′ == DANGERₛ
		return SparseCat([CDMₒ, NoCDMₒ],
			             [params.p_CDM_when_safe, 1-params.p_CDM_when_danger])
	elseif s′ == SAFEₛ
		return SparseCat([CDMₒ, NoCDMₒ],
			             [params.p_CDM_when_safe, 1-params.p_CDM_when_danger])
	end
end

# ╔═╡ a78db25b-c324-4ffb-b39b-383768b0919c
O(a::Action, s′::State) = O(SAFEₛ, a, s′) # first s::State is unused

# ╔═╡ 648d16b0-f4d9-11ea-0a53-39c0bfe2b4e1
md"""
### 2.5 Reward Function
The reward function is addative, meaning we get a reward of $r_\text{collision}$ whenever the spacecraft is in danger *plus* $r_\text{feed}$ whenever we accelerate the baby.
"""

# ╔═╡ 153496b0-f4d9-11ea-1cde-bbf92733afe3
function R(s::State, a::Action)
	return (s == DANGERₛ ? params.r_danger : 0) + (a == ACCELERATEₐ ? params.r_accelerate : 0) + (a == DECELERATEₐ ? params.r_decelerate : 0)
end

# ╔═╡ b664c3b0-f52a-11ea-1e44-71034541ace4
md"
### 2.6 Discount Factor
For an infinite horizon problem, we set the discount factor $\gamma \in [0,1]$ to a value where $\gamma < 1$ to discount future rewards.
"

# ╔═╡ aa58d350-f4d9-11ea-3532-8be52f0d6f2b
γ = 0.9;

# ╔═╡ b35776ca-6f61-47ee-ab37-48da09bbfb2b
md"""
### 2.7 POMDP Structure using `QuickPOMDPs`
We again using `QuickPOMDPs.jl` to succinctly instantiate the crying baby POMDP.
"""

# ╔═╡ 0aa6d08a-8d41-44d5-a1e5-85a6bcb92e81
abstract type SpacecraftCollisionAvoidance <: POMDP{State, Action, Observation} end

# ╔═╡ a858eddc-716b-49ac-864f-04c46b816ab6
pomdp = QuickPOMDP(SpacecraftCollisionAvoidance,
    states       = 𝒮,
    actions      = 𝒜,
	observations = 𝒪,
    transition   = T,
    reward       = R,
	observation  = O,
    discount     = γ,
    initialstate = initialstate_distr);

# ╔═╡ 704ea980-f4db-11ea-01db-233562722c4d
md"""
### 2.8 Policy
We create a simple `Policy` type with an associated `POMDPs.action` function which always feeds the baby when we it's crying.

The `POMDPs.action(π, s)` function maps the current state $s$ (or belief state $b(s)$ for POMDPs) to an action $a$ given a policy $\pi$.

$$\begin{align}
\pi(s) &= a\tag{for MDPs}\\
\pi(b) &= a\tag{for POMDPs}
\end{align}$$
"""

# ╔═╡ d2a3f220-f52b-11ea-2360-bf797a6f9374
md"""
For the simple case, we define 2 policies when we observe the CDM:
"""

# ╔═╡ f3b9f270-f52b-11ea-2f2e-ef56d5522ffb
struct AccelerateWhenCDM <: Policy end

# ╔═╡ 9373ebc5-efde-4a92-ad2c-a92ad58d7a80
# ATTENTION: only defining policy to ACCELERATE in simple model => will have to enhance by implementing a policy to DECELERATE
# struct DecelerateWhenCDM <: Policy end

# ╔═╡ ea5c5ff0-f52c-11ea-2d8f-73cdc0137343
md"And two policies that make the spacecraft accelerate or decelerate when we believe it to be in danger:"

# ╔═╡ 473d77c0-f4da-11ea-0af5-7f7690f39566
struct AccelerateWhenBelievedDanger <: Policy end

# ╔═╡ df1a89c8-28fe-428f-b31b-6d2ea48616bc
# ATTENTION: only defining policy to ACCELERATE in simple model => will have to enhance by implementing a policy to DECELERATE
# struct DecelerateWhenBelievedDanger <: Policy end

# ╔═╡ 072fd490-f52d-11ea-390a-5f0c9d8be485
md"""
### 2.9 Belief

Our `Belief` type is a vector of probabilities representing our belief that the spacecraft is in danger:

$$\begin{align}
\mathbf{b} = \biggl[p(\text{danger}), \;\; p(\text{safe})\biggr]
\end{align}$$

The belief vector must be non-negative and sum to 1 to make it a valid probability distribution.
"""

# ╔═╡ b9439a52-f522-11ea-3caf-2b4bf635b887
const Belief = Vector{Real};

# ╔═╡ 39d07e40-f52d-11ea-33a6-7b31da19d683
md"""
##### Policies
"""

# ╔═╡ 1a991add-b9f9-4140-a2cf-194963a8b22c
md"""**Observation Policies**"""

# ╔═╡ 9a438026-93f9-4ca4-9cba-89d8c4c23cdd
"""Simple policy with takes in the previous observation in place of the belief."""
function POMDPs.action(::AccelerateWhenCDM, o::Observation)
	return o == CDMₒ ? ACCELERATEₐ : CLEARofCONFLICTₐ
end;

# ╔═╡ 1c59e5f5-e644-4ade-9ade-b55203e2d80b
md"""**Belief Policies**"""

# ╔═╡ fb777410-f52b-11ea-294b-77b36ef4f6b3
"""Policy that make the spacecraft accelerate when our belief is stronger towards being in danger"""
function POMDPs.action(::AccelerateWhenBelievedDanger, b::Belief)
	return b[1] > b[2] ? ACCELERATEₐ : CLEARofCONFLICTₐ
end;

# ╔═╡ 2a144c90-f4db-11ea-3a54-bdb5002577f1
md"""
### 2.10 Belief Updater
Belief updaters are provided by the [BeliefUpdaters.jl](https://github.com/JuliaPOMDP/BeliefUpdaters.jl) package.
"""

# ╔═╡ 1687b4e0-f52c-11ea-04f0-b36f816b46c1
md"""
##### Discrete Belief Update
Let's run through an example decision process, first defining a discrete belief updater for our problem. This "belief updater" is a Bayesian filter that will update the current belief of the spacecraft & debris _actual state_ (which we cannot observe directly), using observation that we _can_ get (i.e. `CDM` or `NoCDM`).
"""

# ╔═╡ 25079370-f525-11ea-1c0a-ad5e0b53744a
updater(pomdp::QuickPOMDP{SpacecraftCollisionAvoidance}) = DiscreteUpdater(pomdp);

# ╔═╡ 78cfbd1c-690a-42a2-8ebe-f50df761a03f
md"""
We start out with a uniform belief over where $p(\texttt{danger}) = 0.5$ and $p(\texttt{safe})=0.5$.
"""

# ╔═╡ 13ec6d00-f4de-11ea-3cad-057e7556d7a0
b0 = uniform_belief(pomdp); b0.b

# ╔═╡ c5ae823d-2489-4ceb-a42b-7d4956998bf0
md"""
Then we can "update" our current belief based on our selected action and subsequent observation. The `update` function has the following signature:
```julia
update(::Updater, belief_old, action, observation)
```
"""

# ╔═╡ 8c7b3020-f525-11ea-0518-232981a16f99
begin
	a1 = CLEARofCONFLICTₐ
	o1 = CDMₒ
	b1 = update(updater(pomdp), b0, a1, o1)
	b1.b
end

# ╔═╡ 84573fe4-d895-4076-94ca-adfeba1b2641
md"""
Then we choose to do nothing with the spacecraft ($a_1=\texttt{clear of conflict}$) and observe the spacecraft receiving a CDM ($o_1=\texttt{CDM}$). This belief vector of [$(round.(b1.b,digits=5))] says there is a $(round(b1.b[1], digits=5)) probability that the spacecraft is _actually_ in danger (meaning it's true state is `danger`), and a $(round(b1.b[2], digits=5)) probability that the spacecraft is _actually_ safe.
"""

# ╔═╡ c86f8021-c6b8-44dd-868a-f97270a147c3
md"""
Next, we make the spacecraft accelerate ($a=\texttt{accelerate}$) and observe that it is safe (which we defined to be deterministic, meaning that the spacecraft will _always_ become safe when we accelerate, thus we would expect the observation of `NoCDM`).
"""

# ╔═╡ bae44aa0-f525-11ea-1450-837eb93e42a5
begin
	a2 = ACCELERATEₐ
	o2 = NoCDMₒ
	b2 = update(updater(pomdp), b1, a2, o2)
	b2.b
end

# ╔═╡ 88c62826-fa38-4014-a814-47b4fe9489aa
md"""
Then we do nothing with the spacecraft and observe it receives (again) no CDM.
"""

# ╔═╡ ed4efac0-f526-11ea-283e-a5726ef507ae
begin
	a3 = CLEARofCONFLICTₐ
	o3 = NoCDMₒ
	b3 = update(updater(pomdp), b2, a3, o3)
	b3.b
end

# ╔═╡ 53c1defa-9495-43c0-afad-ddbb7716fc23
md"""
We continue to do nothing with the spacecraft, and continue to see it receives no CDM.
"""

# ╔═╡ fdb129b0-f526-11ea-0f61-b9c4b2356efa
begin
	a4 = CLEARofCONFLICTₐ
	o4 = NoCDMₒ
	b4 = update(updater(pomdp), b3, a4, o4)
	b4.b
end

# ╔═╡ fc03234f-1b89-4735-a447-7082998a6110
md"""
We do nothing with the spacecraft once more, but this time we observe that the spacecraft receives a CDM , thus our belief that the spacecraft is _actually_ in danger leans towards _danger_ (only slightly more than uniform).
"""

# ╔═╡ 0c9e92f0-f527-11ea-1fc2-71bb41a0405a
begin
	a5 = CLEARofCONFLICTₐ
	o5 = CDMₒ
	b5 = update(updater(pomdp), b4, a5, o5)
	b5.b
end

# ╔═╡ 0fea57a0-f4dc-11ea-3133-571b9a56d25b
md"""
## 3. Solutions: _Offline_
As with POMDPs, we can solve for a policy either _offline_ (to generate a full mapping from _beliefs_ to _actions_ for all _states_) or _online_ to only generate a mapping from the current belief state to the next action.

Solution methods typically follow the defined `POMDPs.jl` interface syntax:

```julia
solver = FancyAlgorithmSolver() # inputs are the parameters of said algorithm
policy = solve(solver, pomdp)   # solves the POMDP and returns a policy
```
"""

# ╔═╡ 0aa2497d-f979-46ee-8e93-98cb35706963
md"""
### 3.1 Policy Representation: Alpha Vectors
Since we do not know the current state exactly, we can compute the *utility* of our belief *b*

$$U(b) = \sum_s b(s)U(s) = \mathbf{α}^\top \mathbf{b}$$

where $\mathbf{α}$ is called an _alpha vector_ that contains the expected utility for each _belief state_ under a policy.
"""

# ╔═╡ cfef767b-211f-40d6-af02-3ad0635ffa85
md"""
### 3.2 QMDP
To solve the POMDP, we first need a *solver*. We'll use the QMDP solver$^3$ from `QMDP.jl`. QMDP will treat each belief state as the true state (thus turning it into an MDP), and then use **value iteration** to solve that MDP.

$$\alpha_a^{(k+1)}(s) = R(s,a) + \gamma\sum_{s'}T(s'\mid s, a)\max_{a'}\alpha_{a'}^{(k)}(s')$$
"""

# ╔═╡ 1e14b800-f529-11ea-320b-59280510d94c
md"*Now we solve the POMDP to create the policy. Note the policy type of `AlphaVectorPolicy`.*"

# ╔═╡ 34b98892-1167-41bc-8907-06d5b63da213
# Given a belief vector...
𝐛 = [0.8, 0.2]

# ╔═╡ 70c99bb2-f524-11ea-1509-79b6ce54df1f
md"""
### 3.3 Fast Informed Bound (FIB)
Another _offline_ POMDP solver is the _fast informed bound_ (FIB)$^2$. FIB actually uses information from the observation model $O$ (i.e. "informed").

$$\alpha_a^{(k+1)}(s) = R(s,a) + \gamma\sum_o\max_{a'}\sum_{s'}O(o \mid a,s')T(s'\mid s, a)\alpha_{a'}^{(k)}(s')$$

See the usage here: [https://github.com/JuliaPOMDP/FIB.jl](https://github.com/JuliaPOMDP/FIB.jl)
"""

# ╔═╡ b2c4ef60-f524-11ea-02eb-434f1eed5a99
fib_solver = FIBSolver()

# ╔═╡ 383c5b40-f4e1-11ea-3546-d7d143ce24d8
fib_policy = solve(fib_solver, pomdp)

# ╔═╡ 37d81c83-51a6-49a6-800d-1d2d241f5e29
md"""
### 3.4 Point-Based Value Iteration (PBVI)
_Point-based value iteration_ provides a lower bound and operates on a finite set of $m$ beliefs $B=\{\mathbf{b}_1, \ldots, \mathbf{b}_m\}$, each with an associated alpha vector $\Gamma = \{\boldsymbol{\alpha}_1, \ldots, \boldsymbol{\alpha}_m\}$. These alpha vector define an _approximately optimal value function_:

$$U^\Gamma(\mathbf{b}) = \max_{\boldsymbol\alpha \in \Gamma}\boldsymbol\alpha^\top\mathbf{b}$$

with a lower bound on the optimal value function, $U^\Gamma(\mathbf{b}) \le U^*(\mathbf{b})$ for all $\mathbf{b}$.

PBVI iterates through every possible action $a$ and observation $o$ to extract the alpha vector from the set $\Gamma$ that is maximal at the _resulting_ (i.e., updated) belief $\mathbf{b}'$:

$$\begin{align*}
	\boldsymbol{\alpha}_{a,o} &= \operatorname*{arg\,max}_{\boldsymbol{\alpha} \in \Gamma}\boldsymbol{\alpha}^\top\operatorname{Update}(\mathbf{b}, a, o)\\
                             &= \operatorname*{arg\,max}_{\boldsymbol{\alpha} \in \Gamma}\boldsymbol{\alpha}^\top\mathbf{b}'
\end{align*}$$

Then we construct a new alpha vector for each action $a$ based on these $\boldsymbol{\alpha}_{a,o}$ vectors:

$$\alpha_a(s) = R(s,a) + \gamma\sum_{s',o}O(o \mid a,s')T(s'\mid s, a)\alpha_{a,o}(s')$$

With the final alpha vector produced by the backup operator being:

$$𝛂 = \operatorname*{arg\,max}_{𝛂_a} 𝛂_a^\top \mathbf{b}$$
"""

# ╔═╡ 4a98bbf6-6ad2-43ae-95b5-a7e8fa53ec0e
pbvi_solver = PBVISolver()

# ╔═╡ 27b31ffc-0c9e-4bff-99ad-2a5ff0511101
pbvi_policy = solve(pbvi_solver, pomdp)

# ╔═╡ 6ea123be-f4df-11ea-21d2-71b166bb066a
md"""
## 4. Visualizing Alpha Vectors
**_Recall_**: Since we do not know the current state exactly, we can compute the *utility* of our belief *b*

$$U(b) = \sum_s b(s)U(s) = \mathbf{α}^\top \mathbf{b}$$

where $\mathbf{α}$ is called an _alpha vector_ that contains the expected utility for each _belief state_ under a policy.
"""

# ╔═╡ 293183b4-36a9-462a-a88e-1d125baac781
function plot_alpha_vectors(policy, p_hungry, label="QMDP")
	# calculate the maximum utility, which determines the action to take
	current_belief = [p_hungry, 1-p_hungry]
	feed_idx = Int(policy.action_map[1])+1
	ignore_idx = Int(policy.action_map[2])+1
	utility_feed = policy.alphas[feed_idx]' * current_belief # dot product
	utility_ignore = policy.alphas[ignore_idx]' * current_belief # dot product
	lw_feed, lw_ignore = 1, 1
	check_feed, check_ignore = "", ""
	if utility_feed >= utility_ignore
		current_utility = utility_feed
		lw_feed = 2
		check_feed = "✓"
	else
		current_utility = utility_ignore
		lw_ignore = 2
		check_ignore = "✓"
	end
	
	# plot the alpha vector hyperplanes
	plot(size=(600,340))
	plot!(Int.([FULLₛ, HUNGRYₛ]), policy.alphas[ignore_idx],
		  label="ignore ($label) $(check_ignore)", c=:red, lw=lw_ignore)
	plot!(Int.([FULLₛ, HUNGRYₛ]), policy.alphas[feed_idx],
		  label="feed ($label) $(check_feed)", c=:blue, lw=lw_feed)
	
	# plot utility of selected action
	rnd(x) = round(x,digits=3)
	scatter!([p_hungry], [current_utility], 
		     c=:black, ms=5, label="($(rnd(p_hungry)), $(rnd(current_utility)))")

	title!("Alpha Vectors")
	xlabel!("𝑝(hungry)")
	ylabel!("utility 𝑈(𝐛)")
	xlims!(0, 1)
	ylims!(-40, 5)
end

# ╔═╡ 53b584db-e716-4090-a19e-4530d8694c65
[p_to_collide, 1-p_to_collide]

# ╔═╡ ebe99578-a11c-4c30-a33f-10e78614a70e
@bind p_hungry Slider(0:0.01:1, default=0.5, show_value=true)

# ╔═╡ d4f99682-76ce-4ebc-828b-cff812d4ff56
@bind qmdp_iters Slider(0:60, default=60, show_value=true)

# ╔═╡ 1ae7c200-f4dc-11ea-29c1-b3710f89f475
qmdp_solver = QMDPSolver(max_iterations=qmdp_iters);

# ╔═╡ 3fea65d0-f4dc-11ea-3531-6de282399dce
qmdp_policy = solve(qmdp_solver, pomdp)

# ╔═╡ 30f07d08-229c-4c73-a7d3-51c4c301dc1c
#Query policy for an action
a = action(qmdp_policy, 𝐛)

# ╔═╡ a8460253-884f-474c-9d21-a7d3ee261120
plot_alpha_vectors(qmdp_policy, p_to_collide)

# ╔═╡ a4883a50-39df-4875-8554-30c97240b53d
action(qmdp_policy, [p_hungry, 1-p_hungry])

# ╔═╡ c322f12c-eb6e-4ec0-b5d5-1f1ba00cc216
md"""
Just like MDPs, we can query the policy for an action—but when dealing with POMDPs we input the _belief_ $b$ instead of the _state_ $s$. Reminder that the belief is a probability distribution over the true states:

$$\begin{align}
\mathbf{b} = \biggl[p(\text{hungry}), \;\; p(\text{full})\biggr]
\end{align}$$

Because this vector has to sum to one to be a valid probability distribution, we can represent it as:

$$\begin{align}
\mathbf{b} = \biggl[p(\text{hungry}), \;\; 1 - p(\text{hungry})\biggr]
\end{align}$$

"""

# ╔═╡ da57eb54-db90-42c2-ad30-8479ea2ff857
md"""
### 4.1 Dominanting alpha vectors
To show the piecewise combination of the dominant alpha vectors, here we plot the combination and color the portion of the vector that corresponds to the two actions: $\texttt{feed}$ and $\texttt{ignore}$.
"""

# ╔═╡ f85e829f-ebb4-4eba-a2e9-85661b338339
@bind show_thresholds CheckBox(true)

# ╔═╡ 09a4bc95-f566-4248-bae0-ee142b1c9b4f
@bind show_fib CheckBox(true)

# ╔═╡ d95811f2-98c9-417d-9f72-2ed161e5419c
@bind show_pbvi CheckBox(true)

# ╔═╡ c8cdfb8a-8666-443b-bd42-782585c95948
begin
	p_range = 0:0.001:1

	dominating_action_idx(policy, 𝐛) = Int(action(policy, 𝐛))+1

	dominant_actions(policy) = map(p->
		dominating_action_idx(policy,[p,1-p]), p_range)

	dominant_line(policy) = map(p->
		policy.alphas[dominating_action_idx(policy,[p,1-p])]'*[p,1-p], p_range)

	dominant_line_multiple_α(policy) = map(p->
		argmax(αₐ->αₐ'*[p,1-p], policy.alphas)'*[p,1-p], p_range)

	dominant_color(policy, c1=:blue, c2=:red) = map(p->
		dominating_action_idx(policy,[p,1-p]) == 1 ? c1 : c2, p_range)

	qmdp_solver2 = QMDPSolver()
	qmdp_policy2 = solve(qmdp_solver2, pomdp)

	fib_solver2 = FIBSolver()
	fib_policy2 = solve(fib_solver2, pomdp)

	pbvi_solver2 = PBVISolver()
	pbvi_policy2 = solve(pbvi_solver2, pomdp)

	dominant_line_qmdp = dominant_line(qmdp_policy2)
	dominant_color_qmdp = dominant_color(qmdp_policy2)

	dominant_line_fib = dominant_line(fib_policy2)
	dominant_color_fib = dominant_color(fib_policy2, :cyan, :magenta)

	dominant_line_pbvi = dominant_line_multiple_α(pbvi_policy2)
	dominant_color_pbvi = dominant_color(pbvi_policy2, :green, :black)
	
	# plot the dominant alpha vector hyperplanes
	plot(size=(600,340))
	plot!(p_range, dominant_line_qmdp, label="QMDP",
		  c=dominant_color_qmdp, lw=2)
	
	if show_fib
		plot!(p_range, dominant_line_fib, label="FIB",
			  c=dominant_color_fib, lw=2)
	end
	
	if show_pbvi
		plot!(p_range, dominant_line_pbvi, label="PBVI",
			  c=dominant_color_pbvi, lw=2)
	end

	thresh_qmdp = p_range[findfirst(dominant_actions(qmdp_policy2) .== 1)]
	thresh_fib = p_range[findfirst(dominant_actions(fib_policy2) .== 1)]
	thresh_pbvi = p_range[findfirst(dominant_actions(pbvi_policy2) .== 1)]
	
	if show_thresholds
		plot!([thresh_qmdp, thresh_qmdp], [-40, 5], color=:gray, style=:dash,
			  label="p ≈ $thresh_qmdp (QMDP)")

		if show_fib
			plot!([thresh_fib, thresh_fib], [-40, 5], color=:gray, style=:dash,
				  label="p ≈ $thresh_fib (FIB)")
		end

		if show_pbvi
			plot!([thresh_pbvi, thresh_pbvi], [-40, 5], color=:gray, style=:dash,
				  label="p ≈ $thresh_pbvi (PBVI)")
		end
	end

	title!("Dominant Alpha Vectors")
	xlabel!("𝑝(hungry)")
	ylabel!("utility 𝑈(𝐛)")
	xlims!(0, 1)
	ylims!(-40, 5)
end

# ╔═╡ fb06f470-cba7-4a46-b997-bc3f8b7da7e1
md"""
### 4.2 PBVI alpha vector
We now show how the PBVI algorithm selects the dominant alpha vector.
"""

# ╔═╡ f485cd7a-fe3c-4383-a446-8ce92d53384a
@bind p Slider(0:0.01:1, default=0.5, show_value=true)

# ╔═╡ 347ed884-bf27-4364-9ab9-f51795a38852
plot_alpha_vectors(pbvi_policy, p, "PBVI")

# ╔═╡ 01f371b8-7717-4f31-849e-9062ed79953c
action(pbvi_policy, 𝐛)

# ╔═╡ 9ffe1fcd-fab0-42e0-ab3a-4b847b2986d7
md"""
$$𝛂 = \operatorname*{arg\,max}_{𝛂_a} 𝛂_a^\top \mathbf{b}$$
"""

# ╔═╡ 478ead5f-7a08-4a35-bf1a-34efdd876ec6
𝛂 = argmax(αₐ->αₐ'*𝐛, pbvi_policy.alphas)

# ╔═╡ 78166467-9d55-4756-aabc-93d6f6f71e85
pbvi_policy.alphas # PBVI: m alpha vectors

# ╔═╡ abd5ae8d-aa46-4a6e-af7e-b43fb3cbd0a1
qmdp_policy.alphas # QMDP: 1 alpha vector per action

# ╔═╡ d251bb16-985f-480c-ae2a-51d04412d975
begin
	dominant_color_pbvi2 = dominant_color(pbvi_policy2)
	
	# plot the dominant alpha vector hyperplanes
	plot(size=(600,340))
	plot!(p_range, dominant_line_pbvi, label="PBVI",
		  c=dominant_color_pbvi2, lw=2)

	title!("Dominant Alpha Vectors")
	xlabel!("𝑝(hungry)")
	ylabel!("utility 𝑈(𝐛)")
	xlims!(0, 1)
	ylims!(-40, 5)
end

# ╔═╡ 2262f5e0-f60d-11ea-3744-c569380f8d28
md"""
## 5. Solutions: _Online_
We can solve POMDPs online to produce a `planner` which we then query for an action *online*.
"""

# ╔═╡ e489c1b0-f619-11ea-1cbd-e18cd6b598cb
md"""
#### Partially Observable Monte Carlo Planning (POMCP)
The `BasicPOMCP` package implements the partially observable upper confidence tree (PO-UCT) online tree search algorithm, a subset of the full POMCP algorithm.$^4$
"""

# ╔═╡ f9757160-f60e-11ea-3ca8-358b725c5239
pomcp_solver = POMCPSolver()

# ╔═╡ 252add90-f60f-11ea-0fd2-ad1e951b318a
pomcp_planner = solve(pomcp_solver, pomdp);

# ╔═╡ ba54d8e8-f88b-4b93-a0b7-76347bd87ff0
initialstate(pomdp)

# ╔═╡ 3f7adba0-f60f-11ea-3713-b7c1a3f2c285
aₚ, info = action_info(pomcp_planner, initialstate(pomdp), tree_in_info=true); aₚ

# ╔═╡ ee558380-f611-11ea-2e46-77211ed54f6b
tree = D3Tree(info[:tree], init_expand=3)

# ╔═╡ 71406b44-9eed-4e18-b0e8-d1b723d943aa
md"""
## 6. Concise POMDP definition

```julia
using POMDPs, POMDPModelTools, QuickPOMDPs

@enum State danger safe
@enum Action clearofconflict accelerate decelerate
@enum Observation CDM NoCDM

pomdp = QuickPOMDP(
    states       = [danger, safe],  # 𝒮
    actions      = [clearofconflict, accelerate, decelerate],  # 𝒜
    observations = [CDM, NoCDM], # 𝒪
    initialstate = [safe],          # Deterministic initial state
    discount     = 0.9,             # γ

    transition = function T(s, a)
        if a == accelerate
            return SparseCat([danger, safe], [0, 1])

		elseif a == decelerate
            return SparseCat([danger, safe], [0, 1])

		elseif s == danger && a == clearofconflict
            return SparseCat([danger, safe], [1, 0])

		elseif s == safe && a == clearofconflict
            return SparseCat([danger, safe], [0.1, 0.9])
        end
    end,

    observation = function O(s, a, s′)
        if s′ == danger
            return SparseCat([CDM, NoCDM], [0.9, 0.1])
		elseif s′ == safe
            return SparseCat([CDM, NoCDM], [0.1, 0.9])
        end
    end,

    reward = (s,a)->(s == danger ? -10 : 0) + (a == accelerate ? -5 : 0) + (a == 		decelerate ? -5 : 0)
)

# Solve POMDP
using QMDP
solver = QMDPSolver()
policy = solve(solver, pomdp)

# Query policy for an action, given a belief vector
𝐛 = [0.2, 0.8]
a = action(policy, 𝐛)
```
"""

# ╔═╡ 827bd43e-f4b6-11ea-04be-5b49c1b1a30f
md"""
## References
1. Maxim Egorov, Zachary N. Sunberg, Edward Balaban, Tim A. Wheeler, Jayesh K. Gupta, and Mykel J. Kochenderfer, "POMDPs.jl: A Framework for Sequential Decision Making under Uncertainty", *Journal of Machine Learning Research*, vol. 18, no. 26, pp. 1–5, 2017. [http://jmlr.org/papers/v18/16-300.html](http://jmlr.org/papers/v18/16-300.html)

2. Mykel J. Kochenderfer, Tim A. Wheeler, and Kyle H. Wray, "Algorithms for Decision Making", *MIT Press*, 2022. [https://algorithmsbook.com](https://algorithmsbook.com)

3. Michael Littman, Anthony Cassandra, and Leslie Kaelbling, "Learning Policies for Partially Observable Environments: Scaling Up", *International Conference on Machine Learning (ICML)*, pg. 362--370, 1995.

4. David Silver and Joel Veness, "Monte-Carlo Planning in Large POMDPs", *Advances in Neural Information Processing Systems (NeurIPS)*, pg. 2164--2172, 2010.
"""

# ╔═╡ 7022711e-f522-11ea-30d7-b9f15b2d5f14
begin
	hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]));
	almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]));
	keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]));
	correct(text=md"Great! You got the right answer! Let's move on to the next section.") = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]));
	md"> Academic markdown helper functions located here."
end

# ╔═╡ ce584a40-f521-11ea-2119-01ed93a3f7cc
if POMDPs.action(AccelerateWhenBelievedDanger(), Real[0,1]) == CLEARofCONFLICTₐ && POMDPs.action(AccelerateWhenBelievedDanger(), Real[1,0]) == ACCELERATEₐ
	if POMDPs.action(AccelerateWhenBelievedDanger(), Real[0.5, 0.5]) == ACCELERATEₐ
		almost(md"Err on the side of ignoring when there are uniform beliefs.")
	else
		correct(md"That's right! A simple policy to make the spacecraft accelerate when CDM is received.")
	end
else
	keep_working(md"We want to `accelerate` the spacecraft when we believe it's in danger, and `do nothing` otherwise.")
end

# ╔═╡ dae97d90-f52d-11ea-08c5-bd13a9acbb8a
hint(md"Maybe the `stateindex` function defined above could help—knowing that `b[1]` = _p(hungry)_ and `b[2]` = _p(full)_.")

# ╔═╡ 50b377bc-5246-4eaa-9f83-d9e1592d4447
TableOfContents(title="Partially Observable MDPs", depth=4)

# ╔═╡ a8b53304-c500-48e8-90ef-40ed362b9a6a
md"""
---
"""

# ╔═╡ a35bff87-612c-47a5-b03e-a85f3183cecc
html"""
<script>
var section = 0;
var subsection = 0;
var headers = document.querySelectorAll('h2, h3');
for (var i=0; i < headers.length; i++) {
    var header = headers[i];
    var text = header.innerText;
    var original = header.getAttribute("text-original");
    if (original === null) {
        // Save original header text
        header.setAttribute("text-original", text);
    } else {
        // Replace with original text before adding section number
        text = header.getAttribute("text-original");
    }
    var numbering = "";
    switch (header.tagName) {
        case 'H2':
            section += 1;
            numbering = section + ".";
            subsection = 0;
            break;
        case 'H3':
            subsection += 1;
            numbering = section + "." + subsection;
            break;
    }
    header.innerText = numbering + " " + text;
};
</script>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BasicPOMCP = "d721219e-3fc6-5570-a8ef-e5402f47c49e"
BeliefUpdaters = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
D3Trees = "e3df1716-f71e-5df9-9e2d-98e193103c45"
FIB = "13b007ba-0ca8-5af2-9adf-bc6a6301e25a"
POMDPModelTools = "08074719-1b2a-587c-a292-00f91cc44415"
POMDPPolicies = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PointBasedValueIteration = "835c131e-675f-4498-8e2c-c054c75556e1"
QMDP = "3aa3ecc9-5a5d-57c8-8188-3e47bd8068d2"
QuickPOMDPs = "8af83fb2-a731-493c-9049-9e19dbce6165"

[compat]
BasicPOMCP = "~0.3.6"
BeliefUpdaters = "~0.2.2"
D3Trees = "~0.3.1"
FIB = "~0.4.3"
POMDPModelTools = "~0.3.7"
POMDPPolicies = "~0.4.1"
POMDPs = "~0.9.3"
Parameters = "~0.12.2"
Plots = "~1.22.0"
PlutoUI = "~0.7.9"
PointBasedValueIteration = "~0.2.1"
QMDP = "~0.1.6"
QuickPOMDPs = "~0.2.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BasicPOMCP]]
deps = ["BeliefUpdaters", "CPUTime", "Colors", "D3Trees", "MCTS", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPSimulators", "POMDPs", "Parameters", "ParticleFilters", "Printf", "Random"]
git-tree-sha1 = "a3bab2fd7787f135591ab9220a1deecef2dc56b0"
uuid = "d721219e-3fc6-5570-a8ef-e5402f47c49e"
version = "0.3.6"

[[BeliefUpdaters]]
deps = ["POMDPModelTools", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "7d4f9d57116796ae3fc768d195386b0a42b4a58d"
uuid = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
version = "0.2.2"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CPUTime]]
git-tree-sha1 = "2dcc50ea6a0a1ef6440d6eecd0fe3813e5671f45"
uuid = "a9c8d775-2e2e-55fc-8582-045d282d599e"
version = "1.0.0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4ce9393e871aca86cc457d9f66976c3da6902ea7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[D3Trees]]
deps = ["AbstractTrees", "Base64", "JSON", "Random", "Test"]
git-tree-sha1 = "311af855efa91a595940cd5c0cdb0ff9e8d6b948"
uuid = "e3df1716-f71e-5df9-9e2d-98e193103c45"
version = "0.3.1"

[[DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiscreteValueIteration]]
deps = ["POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "Printf", "SparseArrays"]
git-tree-sha1 = "7ac002779617a7e1693ccdcc3a534f555b3ea61e"
uuid = "4b033969-44f6-5439-a48b-c11fa3648068"
version = "0.4.5"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a837fdf80f333415b69684ba8e8ae6ba76de6aaa"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.24.18"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FIB]]
deps = ["POMDPModelTools", "POMDPPolicies", "POMDPs", "Printf"]
git-tree-sha1 = "d4e5f77b947cb89510feb18ab58336066c908f3f"
uuid = "13b007ba-0ca8-5af2-9adf-bc6a6301e25a"
version = "0.4.3"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FiniteHorizonPOMDPs]]
deps = ["POMDPLinter", "POMDPModelTools", "POMDPs", "Random"]
git-tree-sha1 = "b2f2db6402cf6682ce7b1f1b5ccee84de3b5d19e"
uuid = "8a13bbfe-798e-11e9-2f1c-eba9ee5ef093"
version = "0.3.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MCTS]]
deps = ["CPUTime", "Colors", "D3Trees", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPSimulators", "POMDPs", "Printf", "ProgressMeter", "Random"]
git-tree-sha1 = "864a39c4136998e421c7be0743b9bcfc770037e5"
uuid = "e12ccd36-dcad-5f33-8774-9175229e7b33"
version = "0.4.7"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[POMDPModelTools]]
deps = ["CommonRLInterface", "Distributions", "LinearAlgebra", "POMDPLinter", "POMDPs", "Random", "SparseArrays", "Statistics", "Tricks", "UnicodePlots"]
git-tree-sha1 = "be6e420779e4a076acac228aa68440ae7ce73331"
uuid = "08074719-1b2a-587c-a292-00f91cc44415"
version = "0.3.7"

[[POMDPPolicies]]
deps = ["BeliefUpdaters", "Distributions", "LinearAlgebra", "POMDPModelTools", "POMDPs", "Parameters", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "2920bc20706b82cf6c5058da51b1bb5d3c391a27"
uuid = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
version = "0.4.1"

[[POMDPSimulators]]
deps = ["BeliefUpdaters", "DataFrames", "Distributed", "NamedTupleTools", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "ProgressMeter", "Random"]
git-tree-sha1 = "1c8a996d3b03023bdeb7589ad87231e73ba93e19"
uuid = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
version = "0.3.12"

[[POMDPTesting]]
deps = ["POMDPs", "Random"]
git-tree-sha1 = "6186037fc901d91703c0aa7ab10c145eeb6d0796"
uuid = "92e6a534-49c2-5324-9027-86e3c861ab81"
version = "0.2.5"

[[POMDPs]]
deps = ["Distributions", "LightGraphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "3a8f6cf6a3b7b499ec4294f2eb2b16b9dc8a7513"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.3"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[ParticleFilters]]
deps = ["POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "9cdc1db2a4992d1ba19bf896372b4eaaac78fa98"
uuid = "c8b314e2-9260-5cf8-ae76-3be7461ca6d0"
version = "0.5.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "b1a708d607125196ea1acf7264ee1118ce66931b"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.0"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PointBasedValueIteration]]
deps = ["BeliefUpdaters", "Distributions", "FiniteHorizonPOMDPs", "LinearAlgebra", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs"]
git-tree-sha1 = "c34ec7660c76eb694a90fdedd8859acb20d3fdde"
uuid = "835c131e-675f-4498-8e2c-c054c75556e1"
version = "0.2.1"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QMDP]]
deps = ["DiscreteValueIteration", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "Random"]
git-tree-sha1 = "4f5b20454c103900dbd6aa74184c16d311a5063c"
uuid = "3aa3ecc9-5a5d-57c8-8188-3e47bd8068d2"
version = "0.1.6"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[QuickPOMDPs]]
deps = ["BeliefUpdaters", "NamedTupleTools", "POMDPModelTools", "POMDPTesting", "POMDPs", "Random", "Tricks", "UUIDs"]
git-tree-sha1 = "53b35c8174e56a24d350c66e10ec3ce141530e0c"
uuid = "8af83fb2-a731-493c-9049-9e19dbce6165"
version = "0.2.12"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "f41020e84127781af49fc12b7e92becd7f5dd0ba"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.2"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[Tricks]]
git-tree-sha1 = "ae44af2ce751434f5fa52e23f46533b45f0cfd81"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.5"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodePlots]]
deps = ["Crayons", "Dates", "SparseArrays", "StatsBase"]
git-tree-sha1 = "dc9c7086d41783f14d215ea0ddcca8037a8691e9"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "1.4.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─746961b0-f4b6-11ea-3289-03b36dffbea7
# ╟─a88c0bf0-f4c0-11ea-0e61-853ac9a0c0cb
# ╟─32c56c10-f4d2-11ea-3c79-3dc8b852c182
# ╟─a13e36e0-f4d2-11ea-28cf-d18a43e34c3e
# ╟─dbc77e50-f529-11ea-0d79-71196165ac17
# ╟─222c7568-d9b3-4148-9800-c42372969c14
# ╟─33b27b26-2b32-4620-9212-262fb30fcbbd
# ╟─67a7ce56-5cff-432c-96a2-08098bb8af44
# ╠═bd6724d1-3067-4285-9ee0-c0b1363e8243
# ╠═647c5372-a632-42ae-ab32-ea6ad491b0b6
# ╟─b2a53c8e-f4db-11ea-08ba-67b9158f39b3
# ╠═d0a58780-f4d2-11ea-155d-f55c848f91a8
# ╟─be1258b0-f4db-11ea-390e-2bcc849111d0
# ╠═f49ffe90-f4dc-11ea-1ecb-9d6e6e66d3d4
# ╟─9df137d0-f61c-11ea-0dd6-67535f3b0d52
# ╟─c720f8a0-f61e-11ea-155d-c13361437a85
# ╠═b03708f0-f61e-11ea-38c8-5945da744bff
# ╟─ce359010-f61e-11ea-2f71-a1fc0b6d5300
# ╠═e97a7a20-f4d9-11ea-0aca-659f1ede1fd9
# ╟─d1b6ee9e-f61e-11ea-0619-d13585355550
# ╠═f4427ca2-f4d9-11ea-2822-e16314168c58
# ╟─2e6aff30-f61d-11ea-1f71-bb0a7c3aad2e
# ╠═02937ed0-f4da-11ea-1f82-cb56e99e5e20
# ╟─eb932850-f4d6-11ea-3102-cbbf0e9d8189
# ╠═3d57d840-f4d5-11ea-2744-c3e456949d67
# ╟─d00d9b00-f4d7-11ea-3a5c-fdad48fabf71
# ╟─61655130-f4d6-11ea-3aaf-53233c68b6a5
# ╟─a78db25b-c324-4ffb-b39b-383768b0919c
# ╟─648d16b0-f4d9-11ea-0a53-39c0bfe2b4e1
# ╠═153496b0-f4d9-11ea-1cde-bbf92733afe3
# ╟─b664c3b0-f52a-11ea-1e44-71034541ace4
# ╠═aa58d350-f4d9-11ea-3532-8be52f0d6f2b
# ╟─b35776ca-6f61-47ee-ab37-48da09bbfb2b
# ╠═0aa6d08a-8d41-44d5-a1e5-85a6bcb92e81
# ╠═a858eddc-716b-49ac-864f-04c46b816ab6
# ╟─704ea980-f4db-11ea-01db-233562722c4d
# ╠═fd7f872d-7ef2-4987-af96-4ca4573f29fc
# ╟─d2a3f220-f52b-11ea-2360-bf797a6f9374
# ╠═f3b9f270-f52b-11ea-2f2e-ef56d5522ffb
# ╠═9373ebc5-efde-4a92-ad2c-a92ad58d7a80
# ╟─ea5c5ff0-f52c-11ea-2d8f-73cdc0137343
# ╠═473d77c0-f4da-11ea-0af5-7f7690f39566
# ╠═df1a89c8-28fe-428f-b31b-6d2ea48616bc
# ╟─072fd490-f52d-11ea-390a-5f0c9d8be485
# ╠═b9439a52-f522-11ea-3caf-2b4bf635b887
# ╟─39d07e40-f52d-11ea-33a6-7b31da19d683
# ╟─1a991add-b9f9-4140-a2cf-194963a8b22c
# ╠═9a438026-93f9-4ca4-9cba-89d8c4c23cdd
# ╟─1c59e5f5-e644-4ade-9ade-b55203e2d80b
# ╠═fb777410-f52b-11ea-294b-77b36ef4f6b3
# ╟─ce584a40-f521-11ea-2119-01ed93a3f7cc
# ╟─dae97d90-f52d-11ea-08c5-bd13a9acbb8a
# ╟─2a144c90-f4db-11ea-3a54-bdb5002577f1
# ╟─1687b4e0-f52c-11ea-04f0-b36f816b46c1
# ╠═25079370-f525-11ea-1c0a-ad5e0b53744a
# ╟─78cfbd1c-690a-42a2-8ebe-f50df761a03f
# ╠═13ec6d00-f4de-11ea-3cad-057e7556d7a0
# ╟─c5ae823d-2489-4ceb-a42b-7d4956998bf0
# ╟─84573fe4-d895-4076-94ca-adfeba1b2641
# ╠═8c7b3020-f525-11ea-0518-232981a16f99
# ╟─c86f8021-c6b8-44dd-868a-f97270a147c3
# ╠═bae44aa0-f525-11ea-1450-837eb93e42a5
# ╟─88c62826-fa38-4014-a814-47b4fe9489aa
# ╠═ed4efac0-f526-11ea-283e-a5726ef507ae
# ╟─53c1defa-9495-43c0-afad-ddbb7716fc23
# ╠═fdb129b0-f526-11ea-0f61-b9c4b2356efa
# ╟─fc03234f-1b89-4735-a447-7082998a6110
# ╠═0c9e92f0-f527-11ea-1fc2-71bb41a0405a
# ╟─0fea57a0-f4dc-11ea-3133-571b9a56d25b
# ╟─0aa2497d-f979-46ee-8e93-98cb35706963
# ╟─cfef767b-211f-40d6-af02-3ad0635ffa85
# ╠═17bbb35d-b74d-47a7-8349-904338127977
# ╠═1ae7c200-f4dc-11ea-29c1-b3710f89f475
# ╟─1e14b800-f529-11ea-320b-59280510d94c
# ╠═3fea65d0-f4dc-11ea-3531-6de282399dce
# ╠═34b98892-1167-41bc-8907-06d5b63da213
# ╠═30f07d08-229c-4c73-a7d3-51c4c301dc1c
# ╟─70c99bb2-f524-11ea-1509-79b6ce54df1f
# ╠═9cdc9132-f524-11ea-2051-41beccfeb0e4
# ╠═b2c4ef60-f524-11ea-02eb-434f1eed5a99
# ╠═383c5b40-f4e1-11ea-3546-d7d143ce24d8
# ╟─37d81c83-51a6-49a6-800d-1d2d241f5e29
# ╠═b0f1a551-9eec-4b7d-8fa5-afcbc6a86cd9
# ╠═4a98bbf6-6ad2-43ae-95b5-a7e8fa53ec0e
# ╠═27b31ffc-0c9e-4bff-99ad-2a5ff0511101
# ╟─6ea123be-f4df-11ea-21d2-71b166bb066a
# ╠═df3fd98d-40df-4850-8410-992610bbad10
# ╟─293183b4-36a9-462a-a88e-1d125baac781
# ╠═a8460253-884f-474c-9d21-a7d3ee261120
# ╠═53b584db-e716-4090-a19e-4530d8694c65
# ╠═a4883a50-39df-4875-8554-30c97240b53d
# ╠═ebe99578-a11c-4c30-a33f-10e78614a70e
# ╠═d4f99682-76ce-4ebc-828b-cff812d4ff56
# ╟─c322f12c-eb6e-4ec0-b5d5-1f1ba00cc216
# ╟─da57eb54-db90-42c2-ad30-8479ea2ff857
# ╠═f85e829f-ebb4-4eba-a2e9-85661b338339
# ╠═09a4bc95-f566-4248-bae0-ee142b1c9b4f
# ╠═d95811f2-98c9-417d-9f72-2ed161e5419c
# ╟─c8cdfb8a-8666-443b-bd42-782585c95948
# ╟─fb06f470-cba7-4a46-b997-bc3f8b7da7e1
# ╠═347ed884-bf27-4364-9ab9-f51795a38852
# ╠═f485cd7a-fe3c-4383-a446-8ce92d53384a
# ╠═26869db9-4400-4d5a-ba3d-a6d2d7c79c4e
# ╠═01f371b8-7717-4f31-849e-9062ed79953c
# ╟─9ffe1fcd-fab0-42e0-ab3a-4b847b2986d7
# ╠═478ead5f-7a08-4a35-bf1a-34efdd876ec6
# ╠═78166467-9d55-4756-aabc-93d6f6f71e85
# ╠═abd5ae8d-aa46-4a6e-af7e-b43fb3cbd0a1
# ╟─d251bb16-985f-480c-ae2a-51d04412d975
# ╟─2262f5e0-f60d-11ea-3744-c569380f8d28
# ╟─e489c1b0-f619-11ea-1cbd-e18cd6b598cb
# ╠═315ede12-f60d-11ea-076e-e1b8b460aa9e
# ╠═f9757160-f60e-11ea-3ca8-358b725c5239
# ╠═252add90-f60f-11ea-0fd2-ad1e951b318a
# ╠═ba54d8e8-f88b-4b93-a0b7-76347bd87ff0
# ╠═3f7adba0-f60f-11ea-3713-b7c1a3f2c285
# ╠═6dc0ddc0-f60f-11ea-2a57-158b8a68be4e
# ╠═ee558380-f611-11ea-2e46-77211ed54f6b
# ╟─71406b44-9eed-4e18-b0e8-d1b723d943aa
# ╟─827bd43e-f4b6-11ea-04be-5b49c1b1a30f
# ╟─7022711e-f522-11ea-30d7-b9f15b2d5f14
# ╠═50b377bc-5246-4eaa-9f83-d9e1592d4447
# ╟─a8b53304-c500-48e8-90ef-40ed362b9a6a
# ╟─a35bff87-612c-47a5-b03e-a85f3183cecc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
