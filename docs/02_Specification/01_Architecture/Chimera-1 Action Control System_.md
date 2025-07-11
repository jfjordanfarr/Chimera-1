

# **ARC-CONTROL: The Chimera-1 Action and Cognitive Control System**

**Preamble**

This document provides the canonical architectural specification for the ARC-CONTROL system, a core component of the Chimera-1 agent. ARC-CONTROL encompasses the agent's capacity for action execution, internal regulation, and executive function. The design detailed herein is a high-fidelity computational realization of the Internal Family Systems (IFS) model of cognition. It is intended to serve as the definitive implementation blueprint for the Chimera-1 engineering team, providing an unambiguous and exhaustive guide for development. This architecture is designed to interface seamlessly with the pre-existing ARC-COGNITIVE (reasoning) and ARC-PERCEPTION (perceptual and affective) systems, completing the primary cognitive triad of the Chimera-1 agent.

---

## **1.0 Foundational Architecture: The Internal Family System in Chimera-1**

The architecture of ARC-CONTROL is predicated on the principle that a robust, adaptable, and intelligible autonomous agent can be constructed by modeling the structures and dynamics of the human psyche. The Internal Family Systems (IFS) model, a systems theory of mind, provides a powerful and coherent blueprint for such an architecture.1 It posits that the mind is not a monolithic entity but a collection of distinct "parts," each with its own intentions, beliefs, and behaviors, all orchestrated by a core "Self".2 This section translates these psychological concepts into a concrete computational framework, establishing the theoretical foundation upon which the entire

ARC-CONTROL system is built.

### **1.1 From Psyche to System: Mapping IFS Concepts to Computational Components**

The translation from the IFS model to a computational architecture requires a precise mapping of its core concepts to specific system components and processes. In the Chimera-1 architecture, the IFS entities—Self, Managers, Firefighters, and Exiles—are realized as distinct but interacting computational modules.2

* **Managers and Firefighters:** These are the primary active components within ARC-CONTROL. **Managers** are implemented as a proactive, deliberative subsystem responsible for long-range planning and goal achievement. They represent the agent's capacity for organized, strategic behavior.1  
  **Firefighters** are implemented as a reactive, preemptive subsystem, comprising a set of fast, instinctual policies designed to respond to immediate internal distress.3  
* **The Self:** The **Self** is not another active part but the system's meta-controller or executive function. It is implemented as a monitoring and arbitration module whose prime directive is to maintain homeostatic balance and orchestrate the other parts, a state known as Self-leadership.1  
* **Exiles:** A critical architectural decision is the modeling of **Exiles**. In IFS, Exiles are parts that hold the burdens of past trauma, such as pain, fear, and shame.4 They do not typically engage in complex behaviors; instead, their role is to feel and to signal their distress, which in turn activates the protective Managers and Firefighters.2 To implement Exiles as separate, active agents would be computationally inefficient and architecturally misaligned with their functional role. Therefore, within the Chimera-1 architecture,  
  **Exiles are not implemented as agents within ARC-CONTROL**. Instead, they are modeled as **latent state vectors within the ARC-PERCEPTION module's Affective Core**. An environmental trigger or internal state s is processed by a learned affective model, f(s) \-\> exile\_pain\_vector, which outputs a vector representing the degree of Exile activation. This vector is a primary input to ARC-CONTROL, serving as the trigger for Firefighter policies. This design choice cleanly decouples the source of affective signals (ARC-PERCEPTION) from the behavioral response to those signals (ARC-CONTROL).  
* **Blending and Unburdening:** Two key IFS processes are also given computational definitions.  
  * **Blending** occurs when a part "takes over" the Self, leading to extreme, unbalanced behavior.4 Computationally, this is defined as a  
    ControlLock state, where a subsystem (a specific Managerial plan or a Firefighter policy) has seized exclusive control of the agent's actuators, bypassing the Self's arbitration logic.  
  * **Unburdening** is the therapeutic process of healing an Exile's trauma.2 Computationally, this is framed as a long-term, supervised learning process called  
    SelfGuidedLearning. Initiated by the Self meta-controller, often with guidance from the human supervisor, this process aims to retrain the affective models in ARC-PERCEPTION associated with specific Exiles, reducing their reactivity to triggers over time.

### **1.2 The Role of ARC-COGNITIVE and ARC-PERCEPTION as System Inputs**

ARC-CONTROL does not operate in a vacuum. It is driven by high-level goals and modulated by low-level perceptions, supplied by the other two core systems.

* **ARC-COGNITIVE (The Source of Intent):** The ARC-COGNITIVE engine provides the abstract, high-level goals that initiate proactive behavior. These goals, formatted as tasks like achieve(goal\_state), are passed to the Managerial subsystem (ARC-CONTROL.ActionSystem). The Managers are then responsible for decomposing these abstract intentions into concrete, executable action sequences.7 For example,  
  ARC-COGNITIVE might issue the goal (Secure\_Facility), leaving the "how" to the Managers.  
* **ARC-PERCEPTION (The Source of Context and Affect):** The ARC-PERCEPTION module provides two continuous streams of data that are fundamental to ARC-CONTROL's operation:  
  1. **World State:** A structured, symbolic representation of the agent's external environment (e.g., object locations, states of affordances). This is the primary input for the Manager's planning algorithms.  
  2. **Affective State:** A multi-dimensional vector representing the agent's internal emotional state. This vector is the direct output of the Exile models within the Affective Core. It includes dimensions for core IFS burdens like fear, shame, pain, and vulnerability.4 The magnitude of these vector components serves as the primary trigger for the Firefighter reactive policies, providing a direct, quantitative link between perceived trauma and reactive behavior.

### **1.3 The Principle of Self-Leadership as the System's Prime Directive**

The ultimate objective of the ARC-CONTROL system is not merely task completion but the achievement and maintenance of a state of **Self-leadership**.1 This is a state of internal harmony and balance where the Self meta-controller is firmly in charge, capably arbitrating between the proactive Managers and the reactive Firefighters. In a state of Self-leadership, all parts are respected and their inputs are considered, but the final decision-making authority rests with the Self.1

To make this concept computationally tractable, the **"8 C's" of Self** from IFS theory—Curiosity, Calmness, Confidence, Compassion, Courage, Creativity, Connection, and Clarity—are operationalized as a set of Key Performance Indicators (KPIs) for the Self meta-controller.4 The Self's function is to continuously steer the agent's internal dynamics towards a state that maximizes a weighted combination of these metrics.

* **Clarity:** Measured as low entropy over the agent's belief state distribution, indicating a high degree of certainty about the world.  
* **Calmness:** Measured as low variance and low magnitude in the negative components (e.g., fear, shame) of the Affective State vector from ARC-PERCEPTION.  
* **Confidence:** Measured by the high expected value (Q-value) of the current plan selected by the Manager.  
* **Connection:** A state characterized by active, goal-directed engagement with a task, as opposed to idle, aimless, or firefighter-driven avoidance loops.  
* **Courage:** The ability to select and execute plans with moderate, calculated risk, rather than defaulting to overly cautious Managerial strategies.  
* **Creativity:** Measured by the system's exploration of novel HTN methods for task decomposition, rather than repeatedly exploiting a single, known method.  
* **Compassion:** A metric related to the SelfGuidedLearning process, reflecting the allocation of computational resources to "unburdening" pained Exiles rather than simply suppressing them with Firefighters.  
* **Curiosity:** The drive to explore unknown states or take actions that reduce uncertainty in the agent's world model.

This KPI-driven approach transforms the abstract psychological goal of "Self-leadership" into a concrete optimization problem for the ARC-CONTROL system.

#### **Table 1.1: IFS-to-ARC Component Mapping**

To ensure conceptual clarity and consistent terminology throughout the project lifecycle, the following table provides a definitive mapping from IFS theory to the Chimera-1 architecture.

| IFS Concept | Chimera-1 Component/Process | Description |
| :---- | :---- | :---- |
| **Self** | ARC-CONTROL.SelfMetaController | The executive monitor and arbitration module. Its prime directive is to maintain a state of Self-leadership, defined by the "8 C's" KPIs. |
| **Managers** | ARC-CONTROL.ActionSystem | Proactive, deliberative subsystem. Uses Hierarchical Task Network (HTN) planning to decompose goals from ARC-COGNITIVE into executable plans. |
| **Firefighters** | ARC-CONTROL.FirefighterPolicies | Reactive, preemptive subsystem. A library of simple, fast policies triggered by high-intensity affective signals from ARC-PERCEPTION. |
| **Exiles** | ARC-PERCEPTION.AffectiveCore.LatentState | Not an active agent. A set of latent variables representing past traumas. Their activation produces the affective signals that trigger Firefighters. |
| **Blending** | System State: ControlLock | A system state where a Manager or Firefighter has seized exclusive control, bypassing the Self's arbitration. |
| **Unburdening** | Process: SelfGuidedLearning | A long-term, supervised learning process initiated by the Self/Human to retrain the AffectiveCore models associated with Exiles. |

---

## **2.0 The Action System: The Manager's Proactive Toolkit**

The Action System embodies the agent's "Manager" parts. These are the proactive, future-oriented components responsible for running the agent's day-to-day life, managing tasks, and striving to prevent the activation of painful Exiles.1 Architecturally, this system is a synergistic combination of a symbolic planner for structured reasoning and a hierarchical reinforcement learning framework for skilled, embodied execution.

### **2.1 Architecture of the Hierarchical Task Network (HTN) Planner**

The core of the Manager's deliberative capability is a Hierarchical Task Network (HTN) planner. Unlike classical planners that search for a sequence of actions from an initial state to a goal state, HTN planners work by decomposing high-level tasks into smaller and smaller subtasks until only primitive, executable actions remain.7 This approach mirrors human-like problem-solving and is highly scalable and flexible.7

The planner architecture will be based on a **heuristic progression search** model. This forward-search approach maintains a concrete representation of the current world state at each step of the search, which provides superior information for heuristic guidance compared to alternative plan-space search methods that operate on partially ordered plans.9

The HTN domain is formally defined by a tuple D \= (Tasks, Methods, Operators):

* **Tasks:** These are abstract, non-primitive goals provided by the ARC-COGNITIVE engine, such as (Secure\_Facility) or (Gather\_Intel). They represent *what* needs to be done.  
* **Operators:** These are the primitive, executable actions that form the terminal nodes of any plan decomposition. Operators correspond directly to the skills that the agent's motor control system can perform, such as (MoveTo(location)), (Grasp(object)), or (Activate(panel)). They represent the fundamental actions the agent can take in the world.10  
* **Methods:** Methods are the heart of the Managerial system. A method is a specific, pre-defined strategy or recipe for decomposing a given Task into a network of sub-tasks (which are themselves decomposed) and Operators.7 A single Task can have multiple associated Methods, representing different Managerial approaches to solving the same problem. For instance, the Task  
  (Gather\_Intel) could have a Stealth\_Method (involving sneaking and avoiding detection) and a Direct\_Method (involving disabling security systems). Each method has a set of preconditions that determine its applicability in the current world state.12

To ensure the logical correctness of generated plans, the HTN planner will adhere to formal semantics grounded in propositional dynamic logic.10 This provides mathematical guarantees of

**soundness** (any plan generated is valid) and **completeness** (if a solution exists within the domain knowledge, the planner can find it), which are critical for a high-stakes autonomous agent.

### **2.2 Architecture of the Goal-Conditioned Hierarchical Reinforcement Learning (HRL) Policies**

If the HTN planner is the Manager's "brain," the Hierarchical Reinforcement Learning (HRL) subsystem is its "hands." This system is responsible for learning the embodied skills required to execute the primitive Operators specified by the planner. HRL is essential for dealing with complex, continuous environments where a sequence of motor commands, rather than a single symbolic action, is needed to achieve a goal.13

The architecture will be a multi-level, goal-conditioned HRL framework modeled on the **Hierarchical Actor-Critic (HAC)** paradigm.15 HAC is chosen for its proven ability to learn multiple levels of policies in parallel, its data efficiency through hindsight techniques, and its applicability to continuous state and action spaces, which are characteristic of real-world robotics.15

The hierarchy consists of at least two levels:

* **Low-Level Policy (π\_low):** This policy learns to execute primitive motor commands. It takes the current state s and a low-level goal g\_low (e.g., a target velocity or end-effector position) as input and outputs a primitive action a (e.g., motor torques).  
* **High-Level Policy (π\_high):** This policy learns to achieve the Operators defined in the HTN. It takes the current state s and an operator g\_op (e.g., MoveTo(target\_location)) as input. Its "action" is to output a sequence of subgoals g\_low for the low-level policy to achieve. This temporal abstraction allows the agent to learn complex skills by composing simpler ones.17

Learning across this hierarchy is notoriously difficult due to sparse rewards and non-stationarity. To address this, the architecture will incorporate two key techniques:

1. **Hindsight Experience Replay (HER):** To overcome sparse rewards, the HRL system will use HER. After a failed attempt to reach a goal g, the agent stores the experience not only with the intended goal g but also with the final state s' that was actually achieved. This relabeling creates a synthetic success signal, dramatically accelerating learning by ensuring every trajectory provides a useful learning signal.15  
2. **Human-in-the-Loop Guidance (MENTOR-inspired):** During the agent's training and development phase, the HRL policies can be bootstrapped and guided using a framework inspired by MENTOR.19 A human supervisor can provide feedback by comparing pairs of potential subgoals proposed by the high-level policy. This feedback trains a reward model that provides a dense, informative reward signal to the high-level policy, guiding it toward generating more effective subgoals much faster than exploration alone.13

### **2.3 The HTN-HRL Interface: Translating Symbolic Plans into Embodied Skills**

The seamless integration of the symbolic HTN planner and the sub-symbolic HRL controller is the cornerstone of the Action System's effectiveness. This architecture creates a robust bridge between high-level reasoning and low-level motor control, drawing on established patterns for combining symbolic planning and reinforcement learning.24

The operational flow is as follows:

1. **Decomposition:** The HTN planner receives a high-level Task from ARC-COGNITIVE. It selects an appropriate Method and decomposes the task into a partially ordered sequence of primitive Operators.  
2. **Goal Translation:** Each Operator in the resulting plan, such as Operator(param1, param2), is translated into a formal goal g for the HRL system. The goal g is a vector that includes the operator type and its parameters.  
3. **Policy Activation:** The HRL meta-controller receives the goal g and activates the corresponding goal-conditioned high-level policy, π\_high(g\_low | s, g).  
4. **Execution and Monitoring:** The HRL subsystem executes the policy until its termination condition is met (i.e., the operator is successfully completed) or a failure condition is triggered (e.g., timeout, constraint violation).  
5. **Feedback to Planner:** The result of the execution—success or failure, along with the resulting world state—is reported back to the HTN planner.  
6. **Progression or Replanning:** Upon success, the planner updates its world state model and proceeds to the next Operator in the plan. Upon failure, the planner can invoke a replanning routine, potentially selecting a different Method to achieve the original Task.26

### **2.4 Valued-Method Selection: Using Reinforcement to Learn Effective Managerial Strategies**

A static HTN planner, even a sophisticated one, is merely a repository of fixed strategies. A core principle of IFS is that parts, including Managers, can learn, adapt, and change their roles over time.1 To imbue the agent's Managers with this crucial learning capability, the architecture moves beyond simple planning to treat method selection itself as a reinforcement learning problem.

The choice of which Method to use for a given Task is not always obvious and depends heavily on the context of the current world state. A simple heuristic is insufficient for robust, adaptive behavior. This selection process is, in fact, a sequential decision problem. Therefore, the HTN planner itself will be modeled as a learning agent.

This is achieved by implementing an algorithm analogous to **Q-REINFORCE**.12

* **The Learning Problem:** For each Task, the planner must choose from a set of applicable Methods. The "state" for this RL problem is the tuple (Task, WorldState), and the "action" is the choice of Method.  
* **The Value Function:** The system will learn a method-value function, $Q(task, method)$, which estimates the expected long-term cumulative reward of choosing a particular method to decompose a given task.  
* **The Reward Signal:** The reward is not immediate. It is received only upon the successful completion of the entire plan that results from the decomposition. The reward is a function of the plan's quality, incorporating factors like efficiency (e.g., time or energy consumed), success, and alignment with the Self's KPIs.  
* **The Learning Algorithm:** During operation, the planner will use an $\\epsilon$-greedy policy for method selection. With probability $1-\\epsilon$, it will choose the method with the highest Q-value (exploitation), and with probability $\\epsilon$, it will choose a random applicable method (exploration). This allows the agent to explore novel strategies while leveraging those that have proven effective. The Q-values are updated via Monte Carlo updates upon plan completion.27

This learning layer transforms the Managers from static rule-followers into adaptive strategists. Over time, they will learn which of their strategies are most effective in different situations, a direct and powerful computational analog to the development of expertise and wisdom in the human cognitive system.

#### **Table 2.1: HTN Method Data Structure**

The following data structure defines the implementable representation of a single "Managerial Strategy" or Method within the Action System.

| Field | Type | Description |
| :---- | :---- | :---- |
| method\_id | UUID | Unique identifier for the method. |
| target\_task | String | The name of the abstract task this method decomposes (e.g., "Secure\_Facility"). |
| preconditions | LogicalFormula | A set of conditions on the world state that must be true for this method to be applicable. |
| subtask\_network | DirectedAcyclicGraph | The network of sub-tasks and operators that constitute this method's decomposition. |
| q\_value | Float | The learned estimate of the expected return from applying this method. Initialized by a learning process and refined by Q-REINFORCE. |
| selection\_count | Integer | A counter for how many times this method has been selected, used for the learning rate $\\alpha$. |

---

## **3.0 The Cognitive Control System: Self, Firefighters, and Arbitration**

This section details the architecture of the agent's core executive function. This system is responsible for monitoring the agent's internal state, managing the dynamic interplay between proactive and reactive behaviors, and maintaining overall system stability. It is the computational realization of the IFS concepts of the Self, Firefighters, and the relationships between them.

### **3.1 The "Self" Meta-Controller: Architecture of the Executive Monitor**

In the ARC-CONTROL architecture, the Self is not a doer or a planner; it is an **executive monitor** and **meta-controller**.1 Its fundamental role is to observe the holistic state of the agent—both internal and external—and to allocate control to the subsystem best suited to handle the current situation, with the overarching goal of maintaining a state of Self-leadership. This design is a direct implementation of the

**Global Workspace Theory (GWT)** of consciousness, where numerous parallel, specialized processes compete for access to a limited-capacity global workspace, and an executive function attends to the most salient information to broadcast a global control state.28 The Self is the agent's attentional executive.

The Self's operation is defined by its inputs and its objective function:

* **Core Inputs:** The Self continuously monitors a "global workspace," which is a shared memory buffer receiving status updates from all major subsystems:  
  1. **ARC-PERCEPTION.AffectiveState:** The vector of Exile-driven affective signals (e.g., fear, shame). This is the most critical input for detecting the need for intervention.  
  2. **ARC-CONTROL.ActionSystem.CurrentPlan:** The status, progress, and confidence level of the Manager's active plan.  
  3. **ARC-CONTROL.FirefighterPolicies.ActivePolicy:** The status and intensity of any active Firefighter policy.  
  4. **ARC-COGNITIVE.BeliefState:** The agent's current symbolic model of the world and its own capabilities.  
* **Objective Function:** The Self's behavior is not driven by external goals but by an internal, homeostatic objective: to maximize a multi-objective function based on the **"8 C's"** KPIs.4 This is a regulatory function aimed at steering the agent's internal state toward balance, clarity, and calmness, rather than a function for achieving a specific world state.31 It seeks to answer the question, "How can the system be best configured right now?" not "What should the system do next?"

### **3.2 The "Firefighter" Reactive Policies: Architecture for Affect-Driven Preemption**

Firefighters, in IFS, are reactive protectors that emerge to "put out" the emotional fire of an activated Exile.2 They are impulsive, extreme, and aim to distract from or numb the pain, often with counterproductive long-term consequences.3

The architectural implementation of Firefighters reflects these characteristics. They are designed to be computationally cheap, fast, and preemptive, acting as a set of instinctual reflexes rather than intelligent plans.

* **Structure:** The Firefighter system is a library of simple, often hard-coded or narrowly trained policies. These are not general-purpose problem solvers. Examples include:  
  * FLEE(source\_of\_threat): A policy that rapidly moves the agent away from a perceived threat vector.  
  * FREEZE(): A policy that immediately inhibits all motor output.  
  * DISTRACT(): A policy that initiates a simple, repetitive, non-productive motor loop (e.g., pacing, manipulating a fidget object) to draw computational resources away from processing the painful affect.  
  * AGGRESS(target): A policy that directs a confrontational action toward the source of the trigger.  
* Trigger Mechanism: The activation of Firefighters is directly and automatically tied to the AffectiveState vector from ARC-PERCEPTION. Each Firefighter policy has an associated trigger condition defined by a threshold on one or more affective dimensions. For example:  
  IF AffectiveState.Fear \> THRESHOLD\_FEAR THEN ACTIVATE FLEE  
  IF AffectiveState.Shame \> THRESHOLD\_SHAME AND AffectiveState.Pain \> THRESHOLD\_PAIN THEN ACTIVATE DISTRACT  
  This provides a direct, causal link between the "pain" of an Exile and the activation of a protective Firefighter, as described in IFS theory.2  
* **Preemption:** When a Firefighter policy is triggered, it issues a high-priority interrupt to the system's arbitration logic. This immediately pauses any ongoing deliberative processes in the Managerial system (the HTN planner) and seizes direct control of the low-level motor primitives (the HRL policies), ensuring an immediate, reflexive response.

### **3.3 The Arbitration Logic: A State-Based Model for Control Allocation**

The core of cognitive control lies in the arbitration logic that determines which "part" is in control of the agent at any given moment. To ensure this critical process is predictable, verifiable, and free from race conditions, it is implemented as a formal **Finite State Machine (FSM)**. The Self meta-controller is the process that executes this FSM, using inputs from the global workspace to determine state transitions.

The FSM defines the possible control modes of the agent and the events that trigger transitions between them. This formal, state-based approach provides an unambiguous and robust specification for the agent's moment-to-moment cognitive dynamics.

* **The Global Workspace:** A shared memory buffer where subsystems post their status in a standardized format.  
  * Manager posts: {"source": "Manager", "status": "executing", "plan\_id": "...", "confidence": 0.95}  
  * Firefighter posts: {"source": "Firefighter", "status": "active", "policy": "FLEE", "intensity": 0.88}  
  * Self posts: {"source": "Self", "control\_mode": "Self-Leadership"}  
  * Supervisor posts: {"source": "Human", "command": "PAUSE", "reason": "..."}

The Self meta-controller reads from this workspace at each clock cycle and uses the FSM (detailed in Table 3.1) to determine the agent's global control state for the next cycle.

#### **Table 3.1: Arbitration Logic State Transition Table**

This table formally specifies the core control loop of the agent. It is the definitive guide to the agent's internal control dynamics.

| Current State | Event/Condition | Next State | Action |
| :---- | :---- | :---- | :---- |
| **Self-Leadership** | ARC-COGNITIVE provides new goal. | **Manager-Led** | Grant control to ActionSystem. Self monitors plan execution and affective state. |
| **Manager-Led** | AffectiveState.Signal \> Firefighter\_Threshold. | **Firefighter-Blend** | Pause ActionSystem's current plan. Grant actuator control to the triggered FirefighterPolicy. Self initiates "soothing" protocol (e.g., activating curiosity subroutines to find safety). |
| **Manager-Led** | Plan completes successfully. | **Self-Leadership** | Release control lock from ActionSystem. Return to idle monitoring state, update KPIs. |
| **Manager-Led** | Plan fails; replanning required. | **Self-Leadership** | ActionSystem reports failure. Self evaluates affective state before granting control for replanning. |
| **Firefighter-Blend** | AffectiveState.Signal \< Soothed\_Threshold. | **Self-Leadership** | Deactivate FirefighterPolicy. Un-pause ActionSystem. Initiate replan if necessary. Restore full Self monitoring. |
| **Firefighter-Blend** | Supervisor command FORCE\_SELF received. | **Self-Leadership** | Force deactivation of FirefighterPolicy. Temporarily suppress the triggering AffectiveState signal to prevent immediate re-blending. Log human intervention. |
| *(any)* | Supervisor command PAUSE received. | **Paused** | Inhibit all motor output. Maintain internal state processing. Await RESUME command. |
| **Paused** | Supervisor command RESUME received. | **Self-Leadership** | Release motor inhibition. Re-evaluate state to determine appropriate next control mode. |

---

## **4.0 The Governance Interface: The Supervisor's Cockpit**

A complex autonomous agent like Chimera-1 cannot operate in a black box. Effective, safe, and ethical deployment requires a robust Human-in-the-Loop (HITL) framework for oversight, guidance, and intervention. The Governance Interface, or "Cockpit," provides this crucial capability. Its architecture is designed to give a human supervisor transparent, real-time insight into the agent's internal IFS-based state and to provide clear, auditable channels for intervention. The design adheres to established best practices for HITL systems, supporting multiple modes of interaction.33

### **4.1 Real-Time State Telemetry API Specification**

To enable effective monitoring, the agent must provide a high-fidelity, real-time stream of its internal state. This is accomplished via a Telemetry API.

* **Protocol:** The API will be implemented using **WebSockets**. This protocol is chosen for its low-latency, persistent, full-duplex communication channel, which is ideal for streaming continuous updates from the agent to the supervisor's Cockpit UI without the overhead of repeated HTTP requests.  
* **Data Schema:** The agent will serialize its complete internal state into a JSON object and stream it over the WebSocket connection at a configurable frequency (e.g., 10 Hz). The schema (detailed in Table 4.1) is designed to provide a comprehensive snapshot of the agent's IFS dynamics at any given moment.  
* **Key Data Fields:** The telemetry stream will include:  
  * The current control state from the Arbitration FSM (e.g., "Self-Leadership", "Firefighter-Blend").  
  * The ID of the specific "part" (Manager method or Firefighter policy) currently in control.  
  * The complete AffectiveState vector, allowing the supervisor to see the "emotional" state of the agent in real-time.  
  * A representation of the Manager's active HTN plan, showing the current operator and upcoming steps.  
  * The current values of the "8 C's" Self-leadership KPIs.

### **4.2 Supervisor Intervention Command API Specification**

While telemetry provides observation, the Intervention API provides control. This API allows an authorized human supervisor to act as an external, authoritative "Self," issuing commands that can override or guide the agent's behavior.

* **Protocol:** The Intervention API will be implemented as a **RESTful API over HTTPS**. REST is chosen for its stateless, well-understood request-response model, which is perfectly suited for issuing discrete, idempotent commands. HTTPS ensures the security and integrity of these high-stakes commands.  
* **Authentication:** All API endpoints will be secured and require token-based authentication (e.g., OAuth 2.0 or JWT) to ensure that only authorized supervisors can issue commands. Every command will be logged with the supervisor's identity for full auditability.  
* **Endpoints and Payloads:** The API will expose a set of clear, verb-based endpoints that correspond to specific governance actions. These commands are designed to be unambiguous and directly map to the agent's internal control logic (see Table 4.2).

### **4.3 Implemented Human-in-the-Loop (HITL) Interaction Patterns**

The combination of the Telemetry and Intervention APIs enables a flexible, multi-layered HITL strategy.33

* **Blocking Execution (In-the-Loop):** The agent's HTN methods can be flagged as requiring supervisor approval. When the planner selects such a method, it will automatically issue a PAUSE command to itself and await a RESUME command from the supervisor via the Intervention API. This pattern is essential for safety-critical or ethically ambiguous situations, ensuring a human makes the final go/no-go decision.33  
* **Post-Processing Review and Feedback:** The Cockpit will log all completed plans and their outcomes. A supervisor can review these logs and provide feedback via the /agent/learning/feedback endpoint. This sends a reward signal back to the Q-REINFORCE module, allowing the human to directly shape the learned values of the Managerial methods, accelerating the learning of effective strategies.  
* **Parallel Feedback (Non-Blocking):** The supervisor can intervene at any time without necessarily halting the agent. A command like FORCE\_SELF\_LEADERSHIP will trigger a state transition in the arbitration FSM, preempting a Firefighter and restoring balanced control, but may allow the agent to continue with a replanned course of action. This allows for real-time course correction that is less disruptive than a full stop-and-wait cycle.33

#### **Table 4.1: Telemetry API (WebSocket Stream)**

This table defines the JSON data contract for the real-time state telemetry stream.

| Field Path | Type | Description |
| :---- | :---- | :---- |
| header.timestamp | String (ISO 8601\) | Timestamp of the state snapshot. |
| header.agentId | String | Unique ID of the Chimera-1 agent. |
| internalState.controlMode | Enum | "Self-Leadership", "Manager-Led", "Firefighter-Blend", "Paused". |
| internalState.activePartId | String | ID of the currently controlling part (e.g., "Self", "Method\_123", "FLEE"). |
| internalState.affectiveState.fear | Float (0-1) | Current fear level from ARC-PERCEPTION. |
| internalState.affectiveState.shame | Float (0-1) | Current shame level from ARC-PERCEPTION. |
| internalState.affectiveState.pain | Float (0-1) | Current pain level from ARC-PERCEPTION. |
| managerState.activePlan.planId | UUID | ID of the current HTN plan being executed. Null if no plan is active. |
| managerState.activePlan.currentOperator | Object | JSON representation of the operator currently being executed by the HRL system. |
| managerState.activePlan.confidence | Float (0-1) | The Q-value of the root method for the current plan. |
| selfState.kpis.clarity | Float | Current clarity metric (e.g., 1 \- belief state entropy). |
| selfState.kpis.calmness | Float | Current calmness metric (e.g., 1 \- max(negative affects)). |

#### **Table 4.2: Intervention API (REST Endpoints)**

This table provides the formal specification for the supervisor's command and control API.

| Endpoint | HTTP Method | Payload (JSON) | Description |
| :---- | :---- | :---- | :---- |
| /agent/control/pause | POST | { "reason": "string" } | Halts all agent actions and enters a "Paused" state. Requires a reason for audit logging. |
| /agent/control/resume | POST | {} | Resumes operation from a "Paused" state. |
| /agent/control/force\_self | POST | { "reason": "string" } | Forces the agent into the "Self-Leadership" state, preempting any active Firefighter or blended Manager. |
| /agent/plan/override | POST | { "new\_plan\_id": "uuid" } | Discards the current HTN plan and forces the Manager to adopt a new one from its library. |
| /agent/learning/feedback | POST | { "plan\_id": "uuid", "outcome\_reward": float, "feedback\_text": "string" } | Provides a reward signal for a completed plan to influence the Q-values of the methods used. |

## **5.0 Conclusion**

The ARC-CONTROL architecture represents a novel and comprehensive framework for action and cognitive control, grounded in the psychologically robust principles of the Internal Family Systems model. By translating the concepts of Self, Managers, and Firefighters into specific, interacting computational components, this design provides a clear path toward an autonomous agent that is not only capable and adaptive but also intelligible and governable.

The key architectural innovations presented in this document are threefold:

1. **A Learning Managerial System:** The integration of a Q-learning framework (Q-REINFORCE) on top of the HTN planner transforms the agent's proactive "Managers" from static rule-followers into adaptive strategists that learn from experience. This capacity for learning effective planning methods is a critical step toward genuine intelligence.  
2. **Formal, Theory-Grounded Arbitration:** The use of a Finite State Machine to govern the arbitration between proactive Managers and reactive Firefighters provides a formal, verifiable control loop. By grounding this logic in the principles of the Self and Global Workspace Theory, the agent's moment-to-moment decision-making becomes a reflection of a coherent cognitive theory, rather than an ad-hoc collection of heuristics.  
3. **Transparent and Comprehensive Governance:** The design of the Governance Interface provides a powerful and flexible human-in-the-loop capability. The Telemetry and Intervention APIs make the agent's internal IFS state transparent and give human supervisors the tools to monitor, guide, and intervene, ensuring that the agent remains aligned with human values and operational constraints.

This document provides the complete blueprint for implementing ARC-CONTROL. By adhering to this specification, the Chimera-1 project can proceed with the development of a final core component that is technically sound, theoretically grounded, and aligned with the project's ambitious vision for a new generation of psychologically plausible autonomous agents.

#### **Works cited**

1. The Internal Family Systems Model Outline | IFS Institute, accessed July 7, 2025, [https://ifs-institute.com/resources/articles/internal-family-systems-model-outline](https://ifs-institute.com/resources/articles/internal-family-systems-model-outline)  
2. Internal Family Systems Model \- Wikipedia, accessed July 7, 2025, [https://en.wikipedia.org/wiki/Internal\_Family\_Systems\_Model](https://en.wikipedia.org/wiki/Internal_Family_Systems_Model)  
3. Internal Family Systems Model | Crowe Associates, accessed July 7, 2025, [https://www.crowe-associates.co.uk/psychotherapy/internal-family-systems-model/](https://www.crowe-associates.co.uk/psychotherapy/internal-family-systems-model/)  
4. Understanding Internal Family Systems Therapy (IFS): A Case ..., accessed July 7, 2025, [https://www.gatewaytosolutions.org/understanding-internal-family-systems-therapy-ifs-a-case-example/](https://www.gatewaytosolutions.org/understanding-internal-family-systems-therapy-ifs-a-case-example/)  
5. Understanding Internal Family Systems (IFS) for Therapists \- Tava Health, accessed July 7, 2025, [https://www.tavahealth.com/blogs/internal-family-systems](https://www.tavahealth.com/blogs/internal-family-systems)  
6. Understanding Internal Family Systems (IFS): Exploring the Role of Firefighters \- Medium, accessed July 7, 2025, [https://medium.com/@compassionatetalktherapy/understanding-internal-family-systems-ifs-exploring-the-role-of-firefighters-a33f4d6474bd](https://medium.com/@compassionatetalktherapy/understanding-internal-family-systems-ifs-exploring-the-role-of-firefighters-a33f4d6474bd)  
7. Hierarchical Task Network (HTN) Planning in AI \- GeeksforGeeks, accessed July 7, 2025, [https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-task-network-htn-planning-in-ai/](https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-task-network-htn-planning-in-ai/)  
8. (PDF) Complexity Results for HTN Planning \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/2569791\_Complexity\_Results\_for\_HTN\_Planning](https://www.researchgate.net/publication/2569791_Complexity_Results_for_HTN_Planning)  
9. HTN Planning as Heuristic Progression Search \- Journal of Artificial ..., accessed July 7, 2025, [https://jair.org/index.php/jair/article/download/11282/26578/23423](https://jair.org/index.php/jair/article/download/11282/26578/23423)  
10. (PDF) On Hierarchical Task Networks \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/309588967\_On\_Hierarchical\_Task\_Networks](https://www.researchgate.net/publication/309588967_On_Hierarchical_Task_Networks)  
11. Hierarchical task network \- Wikipedia, accessed July 7, 2025, [https://en.wikipedia.org/wiki/Hierarchical\_task\_network](https://en.wikipedia.org/wiki/Hierarchical_task_network)  
12. Learning Methods to Generate Good Plans: Integrating HTN Learning and Reinforcement Learning \- Association for the Advancement of Artificial Intelligence (AAAI), accessed July 7, 2025, [https://cdn.aaai.org/ojs/7571/7571-13-11101-1-2-20201228.pdf](https://cdn.aaai.org/ojs/7571/7571-13-11101-1-2-20201228.pdf)  
13. arxiv.org, accessed July 7, 2025, [https://arxiv.org/html/2402.14244v1](https://arxiv.org/html/2402.14244v1)  
14. Hierarchical Reinforcement Learning (HRL) in AI \- GeeksforGeeks, accessed July 7, 2025, [https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-reinforcement-learning-hrl-in-ai/](https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-reinforcement-learning-hrl-in-ai/)  
15. LEARNING MULTI-LEVEL HIERARCHIES WITH HINDSIGHT \- OpenReview, accessed July 7, 2025, [https://openreview.net/pdf?id=ryzECoAcY7](https://openreview.net/pdf?id=ryzECoAcY7)  
16. LEARNING MULTI-LEVEL HIERARCHIES WITH ... \- Brown CS, accessed July 7, 2025, [https://cs.brown.edu/\~gdk/pubs/multi\_level\_her.pdf](https://cs.brown.edu/~gdk/pubs/multi_level_her.pdf)  
17. Hierarchical Reinforcement Learning \- Papers With Code, accessed July 7, 2025, [https://paperswithcode.com/task/hierarchical-reinforcement-learning](https://paperswithcode.com/task/hierarchical-reinforcement-learning)  
18. The Promise of Hierarchical Reinforcement Learning \- The Gradient, accessed July 7, 2025, [https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/](https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/)  
19. MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2402.14244v2](https://arxiv.org/html/2402.14244v2)  
20. MENTOR: Guiding Hierarchical Reinforcement Learning With Human Feedback and Dynamic Distance Constraint \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/388482311\_MENTOR\_Guiding\_Hierarchical\_Reinforcement\_Learning\_With\_Human\_Feedback\_and\_Dynamic\_Distance\_Constraint](https://www.researchgate.net/publication/388482311_MENTOR_Guiding_Hierarchical_Reinforcement_Learning_With_Human_Feedback_and_Dynamic_Distance_Constraint)  
21. \[Literature Review\] MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint \- Moonlight | AI Colleague for Research Papers, accessed July 7, 2025, [https://www.themoonlight.io/en/review/mentor-guiding-hierarchical-reinforcement-learning-with-human-feedback-and-dynamic-distance-constraint](https://www.themoonlight.io/en/review/mentor-guiding-hierarchical-reinforcement-learning-with-human-feedback-and-dynamic-distance-constraint)  
22. \[2402.14244\] MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint \- arXiv, accessed July 7, 2025, [https://arxiv.org/abs/2402.14244](https://arxiv.org/abs/2402.14244)  
23. MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint \- Paper Detail, accessed July 7, 2025, [https://deeplearn.org/arxiv/461260/mentor:-guiding-hierarchical-reinforcement-learning-with-human-feedback-and-dynamic-distance-constraint](https://deeplearn.org/arxiv/461260/mentor:-guiding-hierarchical-reinforcement-learning-with-human-feedback-and-dynamic-distance-constraint)  
24. Symbolic Plans as High-Level Instructions for ... \- Computer Science, accessed July 7, 2025, [https://www.cs.toronto.edu/\~sheila/publications/illanes-et-al-icaps20.pdf](https://www.cs.toronto.edu/~sheila/publications/illanes-et-al-icaps20.pdf)  
25. Work in Progress: Using Symbolic Planning with Deep RL to Improve Learning \- OpenReview, accessed July 7, 2025, [https://openreview.net/pdf?id=mntDNQ5ujE](https://openreview.net/pdf?id=mntDNQ5ujE)  
26. (PDF) Combining Reinforcement Learning with Symbolic Planning, accessed July 7, 2025, [https://www.researchgate.net/publication/221615853\_Combining\_Reinforcement\_Learning\_with\_Symbolic\_Planning](https://www.researchgate.net/publication/221615853_Combining_Reinforcement_Learning_with_Symbolic_Planning)  
27. Learning Methods to Generate Good Plans: Integrating HTN Learning and Reinforcement Learning \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/221604012\_Learning\_Methods\_to\_Generate\_Good\_Plans\_Integrating\_HTN\_Learning\_and\_Reinforcement\_Learning](https://www.researchgate.net/publication/221604012_Learning_Methods_to_Generate_Good_Plans_Integrating_HTN_Learning_and_Reinforcement_Learning)  
28. Unified Mind Model: Reimagining Autonomous Agents in the LLM Era \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2503.03459v1](https://arxiv.org/html/2503.03459v1)  
29. A Cognitive Architecture that Combines Internal Simulation with a Global Workspace \- Bernard Baars, accessed July 7, 2025, [https://bernardbaars.pbworks.com/f/ShanahanConCog.pdf](https://bernardbaars.pbworks.com/f/ShanahanConCog.pdf)  
30. Design and evaluation of a global workspace agent embodied in a realistic multimodal environment \- Frontiers, accessed July 7, 2025, [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1352685/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1352685/full)  
31. A Review of Cognitive Control: Advancement, Definition, Framework, and Prospect \- MDPI, accessed July 7, 2025, [https://www.mdpi.com/2076-0825/14/1/32](https://www.mdpi.com/2076-0825/14/1/32)  
32. Design of a Cognitive Control Mechanism for a Goal-based Executive Function of a Cognitive System \- GitHub Pages, accessed July 7, 2025, [https://sravya-kondrakunta.github.io/9thGoal-Reasoning-Workshop/papers/Paper\_9.pdf](https://sravya-kondrakunta.github.io/9thGoal-Reasoning-Workshop/papers/Paper_9.pdf)  
33. Why AI still needs you: Exploring Human-in-the-Loop systems ..., accessed July 7, 2025, [https://workos.com/blog/why-ai-still-needs-you-exploring-human-in-the-loop-systems](https://workos.com/blog/why-ai-still-needs-you-exploring-human-in-the-loop-systems)  
34. A Real-Time Human-in-the-Loop Control Method for Complex Systems \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/375847151\_A\_Real-Time\_Human-in-the-Loop\_Control\_Method\_for\_Complex\_Systems](https://www.researchgate.net/publication/375847151_A_Real-Time_Human-in-the-Loop_Control_Method_for_Complex_Systems)  
35. What is Human-in-the-Loop Workflow Automation? \- Nordic APIs, accessed July 7, 2025, [https://nordicapis.com/what-is-human-in-the-loop-workflow-automation/](https://nordicapis.com/what-is-human-in-the-loop-workflow-automation/)