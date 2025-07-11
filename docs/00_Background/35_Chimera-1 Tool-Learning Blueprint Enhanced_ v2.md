

# **ARC-TOOL-LEARNING: A Blueprint for a Tool-Use Acquisition Curriculum**

Document ID: ARC-TOOL-LEARNING-V1.0  
Date: July 11, 2025

## **Preamble: From Naive Agent to Digital Artisan**

### **Introduction**

The objective of the Chimera-1 project is not merely to create an agent that can use digital tools, but to cultivate a *digital artisan*—an intelligent system that wields its tools with mastery, context-awareness, and adaptive finesse. Where a novice follows instructions, an artisan understands the underlying principles of their craft. This document, **ARC-TOOL-LEARNING**, presents the definitive architectural and training blueprint to guide the transformation of a naive Chimera-1 agent into such an artisan. This curriculum is designed to move beyond simple tool invocation and instill a deep, generative understanding of tool-use, enabling the agent to solve novel, complex problems with creativity and robustness.

### **The Three Pillars of Mastery**

The ARC-TOOL-LEARNING curriculum is founded upon three integrated pillars, each addressing a fundamental aspect of tool mastery. Together, they form a cohesive strategy for developing advanced agentic capabilities.

1. **Neuro-Computational Architecture (Gene Atlas):** At its core, the agent's ability to learn and generalize tool use is grounded in a biologically plausible architecture. We will mechanistically map tool representation and learning to the Chimera-1 Gene Atlas, our implementation of the cortical columnar hypothesis proposed by the Thousand Brains Theory. This modular, structured approach provides a robust foundation for continuous learning and generalization far beyond what is possible with monolithic models.  
2. **Psychologically-Grounded Policy (SDTM):** True mastery requires adapting one's approach to the situation at hand. We introduce the State-Dependent Tool Modulation (SDTM) framework, a novel cognitive architecture that endows the agent with internal states analogous to human psychological modes (e.g., focused manager, rapid-response firefighter). These states, driven by an appraisal of the environment, dynamically modulate the agent's tool-use policies, enabling it to respond with appropriate urgency, caution, or creativity.  
3. **State-of-the-Art Training Regimen:** The agent's skills are forged through a rigorous, multi-phase curriculum that synthesizes the most effective learning paradigms. This regimen progresses from foundational imitation to strategic exploration and finally to nuanced preference alignment, ensuring a comprehensive and deeply ingrained skill set. The entire process is standardized around the ActionTranscript-v1.0 data schema and leverages cutting-edge optimization algorithms like Odds Ratio Preference Optimization (ORPO) for maximum efficiency and performance.

### **Document Purpose and Structure**

This document serves as the formal design specification and implementation guide for the Chimera-1 tool-learning initiative. It is intended for the project's AI research and engineering teams. The following sections provide a comprehensive and exhaustive exploration of the theoretical underpinnings, architectural designs, and practical training protocols required to realize the vision of the digital artisan.

---

## **Section 1: A Synthesis of Modern Tool-Learning Paradigms**

To construct a state-of-the-art curriculum, it is imperative to first synthesize the principles and findings from the rapidly evolving field of tool-augmented Large Language Models (LLMs). Current research offers a fragmented landscape of techniques 1; this section provides a unified perspective, establishing the core learning strategies that form the foundation of the ARC-TOOL-LEARNING blueprint.

### **1.1. The Learning Triad: A Hybrid Approach to Skill Acquisition**

A consistent meta-pattern has emerged across a variety of successful agent training frameworks: a progression from imitation to exploration and, finally, to refinement. This three-phase approach effectively addresses the challenges of learning in a vast and complex action space, such as tool use, and forms the structural backbone of our curriculum.

#### **Imitation Learning (IL) as the Foundation**

The initial phase of learning must provide the agent with a foundational understanding of tool-use syntax and semantics. This is most effectively achieved through Imitation Learning (IL), also known as Supervised Fine-Tuning (SFT) in the context of LLMs. Frameworks like **ToolAlpaca** have demonstrated that even compact models can acquire generalized tool-use abilities by fine-tuning on a diverse corpus of simulated tool-use instances.3 This initial training phase teaches the agent the essential "grammar" of tool interaction: how to correctly format an API call, select the appropriate function name, and populate parameters with valid arguments.4 This principle is not confined to digital tools; it is a cornerstone of learning in domains from robotic manipulation to autonomous driving, where agents first learn by observing human demonstrations.6

However, relying solely on IL is insufficient for true mastery. IL suffers from a critical flaw known as causal confusion or the i.i.d. assumption violation: the agent learns to mimic expert actions in familiar states but struggles when its own actions lead it into novel, out-of-distribution states not present in the training data.6 This can cause compounding errors, where a single mistake leads to a trajectory from which the agent cannot recover. Therefore, IL serves as a necessary, but not sufficient, "cold-start" mechanism.10

#### **Reinforcement Learning (RL) for Strategic Optimization**

To overcome the brittleness of pure imitation, the curriculum's second phase introduces Reinforcement Learning (RL). RL enables the agent to move from simple mimicry to strategic reasoning by allowing it to learn from the consequences of its own actions through trial and error.12 Frameworks like

**ReTool** explicitly leverage RL to teach an agent not just *how* to call a tool, but *when* and *why*. By receiving outcome-based rewards (e.g., successful task completion), the agent can explore the vast space of possible action sequences to discover optimal strategies that may not have been present in the initial demonstration data.10 This process fosters the emergence of more sophisticated behaviors, such as the ability to recover from tool errors or to self-correct a flawed line of reasoning—hallmarks of true problem-solving ability.10 The power of RL to enhance complex reasoning has been demonstrated in numerous contexts, highlighting its essential role in pushing models beyond their initial supervised training.15

#### **Self-Improvement for Autonomous Evolution**

The pinnacle of mastery is the ability to improve autonomously, without constant dependence on externally curated data or human supervision. The final dimension of our learning triad is self-improvement. Frameworks like **ToolACE-DEV** and **SeaLong** provide a blueprint for this capability.18 ToolACE-DEV introduces a self-evolving paradigm where the agent uses its existing knowledge to generate new, high-quality training instances for itself.18 It decomposes the problem, learns to generate candidate tools for a query, and then generates invocations for those tools. Successful trajectories are then used to further fine-tune the model. This creates a virtuous cycle of self-improvement, reducing reliance on expensive and potentially incompatible data from more advanced "teacher" models.18 Similarly, SeaLong demonstrates that an agent can improve its long-context reasoning by sampling multiple outputs for a problem, scoring them, and then fine-tuning on the best ones.19 This capacity for self-driven evolution is what will ultimately enable Chimera-1 to adapt and grow its skills over time, truly embodying the concept of a learning artisan.

### **1.2. Comparative Analysis of State-of-the-Art Frameworks**

The design of our curriculum is informed by a detailed analysis of the leading tool-learning frameworks, adopting their strengths while mitigating their weaknesses.

* **Data Generation Strategies:** The quality and diversity of training data are paramount. **ToolAlpaca** utilizes a multi-agent simulation environment to generate a broad dataset of over 3,900 tool-use instances across more than 400 real-world APIs, prioritizing scale and diversity with minimal human intervention.3  
  **FireAct**, in contrast, employs a "distillation" approach, using a powerful teacher model (GPT-4) to generate high-quality, diverse task-solving trajectories.21 This focus on trajectory diversity—incorporating different reasoning methods like Chain of Thought (CoT) and ReAct—is a key contributor to the resulting agent's robustness.22  
  **GenTool** takes a more targeted approach, synthetically generating data specifically designed to train for two critical types of generalization: "Zero-to-One" (learning to use a tool that was previously unavailable) and "Weak-to-Strong" (learning to prefer a more powerful version of an existing tool).24 The ARC-TOOL-LEARNING curriculum will employ a hybrid data generation strategy, combining the breadth of simulation, the quality of distillation, and the targeted challenges of synthetic generalization scenarios.  
* **Training Methodologies:** The structure of the training process itself is also critical. **FireAct** demonstrates the profound benefits of fine-tuning on a diverse corpus, showing that smaller models can outperform larger, prompted-only models when trained on varied trajectories.21  
  **ReTool** formalizes a two-phase pipeline: an initial SFT "cold-start" to teach foundational competency, followed by an RL phase to discover optimal strategies for using a code interpreter tool.10  
  **ToolACE-DEV** proposes a three-stage curriculum: (1) adapting to tool documentation, (2) learning query-aware tool generation and invocation, and (3) entering a continuous self-evolution loop.18 Our curriculum synthesizes these approaches, adopting the two-phase SFT-then-RL structure of ReTool, enriching it with the data diversity principles of FireAct, and culminating in the self-evolution loop of ToolACE-DEV.  
* **Generalization Approaches:** A key failure mode for tool-using agents is the inability to generalize to unseen tools or situations. **GenTool** directly addresses this by building its training data and fine-tuning process around explicit generalization tasks.24 Its two-stage fine-tuning process, which first teaches the model to rank tools by capability before selecting one, enables a more nuanced understanding of functional differences between similar tools.24 This explicit focus on generalization is a crucial insight that will be integrated into our evaluation suite (Section 6), ensuring we are not just measuring performance on known tools but actively testing for the ability to adapt to novel ones.

### **1.3. Core Challenges in Tool Acquisition**

A comprehensive survey of the field reveals a consistent set of challenges that any robust tool-learning curriculum must address.26 The ARC-TOOL-LEARNING blueprint is explicitly designed to overcome these hurdles.

* **Tool Invocation Timing & Selection:** Knowing *when* to use a tool is as important as knowing *how*. Unnecessary tool calls increase latency and cost, and can introduce errors.26 The RL phase of our curriculum (Phase 2\) directly addresses this by rewarding the agent not just for success, but for efficiency. Furthermore, the SDTM framework (Section 3\) provides a powerful contextual bias; for example, an agent in "Firefighter" mode will learn to prioritize fast, reliable tools over more complex, exploratory ones.  
* **Robustness to Noisy Environments:** Real-world tools are not always reliable; APIs can fail, return malformed data, or provide irrelevant information. The **FireAct** framework showed that training on diverse trajectories, including those with noisy tool outputs, significantly improves agent robustness.21 Our data generation process will therefore explicitly include simulated API failures, random responses, and other forms of noise to inoculate the agent against such real-world imperfections.  
* **Generalization to Unseen Tools:** The ability to use a tool never seen during training is the hallmark of true intelligence. This is a primary focus of our architecture. The Gene Atlas (Section 2\) is designed to learn a structured, compositional model of tools rather than memorizing specific API calls. By learning the "concept" of a REST API or a SQL database, it can generalize to new instances. This is complemented by incorporating **GenTool's** explicit generalization training scenarios into our evaluation suite.24  
* **Catastrophic Forgetting:** Fine-tuning a model on a new tool can cause it to "forget" how to use older ones.26 The modular design of the  
  Gene Atlas is our primary defense against this. When a new tool is introduced, a new set of "Tool Genes" (cortical columns) is allocated and trained, leaving the weights associated with existing tools largely untouched.28 This is augmented by experience replay techniques during the continuous learning phase.  
* **Efficiency and Cost:** Training and deploying large agentic models can be prohibitively expensive. Our curriculum addresses this in two ways. First, by adopting more efficient preference optimization algorithms like **ORPO** (Phase 3), we reduce the computational overhead compared to traditional RLHF methods.30 Second, by embracing the self-evolution paradigm of  
  **ToolACE-DEV**, we reduce the long-term cost and dependency on expensive, proprietary teacher models for data generation.18

---

## **Section 2: The Gene Atlas: A Neuro-Computational Architecture for Tool Representation**

The ability of an agent to generalize its skills to novel tools and situations is not merely a function of its training data, but is fundamentally constrained by its underlying architecture. To build a true digital artisan, we must move beyond monolithic neural networks and adopt an architecture that reflects the principles of biological intelligence. The Gene Atlas is Chimera-1's implementation of such an architecture, grounded in the computational principles of the human neocortex as described by the Thousand Brains Theory (TBT).

### **2.1. The Thousand Brains Theory as a Blueprint**

The foundational premise of our architecture is drawn directly from modern neuroscience. The Thousand Brains Theory posits that the neocortex, the seat of human intelligence, is not a single, hierarchical processor. Instead, it is a vast, uniform sheet composed of approximately 150,000 repeating microcircuits known as cortical columns.32 Each of these columns functions as a complete sensorimotor modeling system, learning models of objects and concepts through interaction with the world.35 Perception is achieved through a consensus-building "vote" among these thousands of parallel models.36

This distributed, modular design offers immense advantages in terms of robustness, scalability, and continuous learning—qualities that are notoriously difficult to achieve in standard deep learning paradigms.29 In the Chimera-1 architecture, the

Gene Atlas serves as this cortical sheet. Each "gene" within the atlas is a functional unit analogous to a cortical column. This modularity is the key to overcoming challenges like catastrophic forgetting; when the agent needs to learn a new tool, a new "Tool Gene" (a dedicated set of columns) can be allocated and trained, preserving the knowledge stored in existing genes.28

### **2.2. The "Tool Gene": Anatomy of a Cortical Column**

Within this framework, a tool is not merely a function to be called but an "object" to be understood. Each Tool Gene is responsible for learning a complete, structured model of a single tool or a cohesive family of related tools (e.g., a "REST API Gene" for handling HTTP requests, or a "Database Query Gene" for SQL interactions). This aligns directly with the TBT's proposal that individual columns learn comprehensive models of objects in the world.36

The internal mechanics of each Tool Gene are inspired by the Hierarchical Temporal Memory (HTM) algorithm, which is the primary algorithmic implementation of TBT.42 We will implement two core HTM components within each gene:

* **Spatial Pooler (SP):** The SP is responsible for learning the static structure and features of a tool's interface. It takes unstructured or semi-structured input, such as an OpenAPI specification or other API documentation, and converts it into a Sparse Distributed Representation (SDR).45 SDRs are high-dimensional, binary representations where only a small percentage of bits are active at any time. This format is inherently robust to noise and allows for semantic similarity to be measured by the overlap of active bits.43 The SP will learn to represent the essential features of a tool—its parameters, their data types, the structure of its return values—in this resilient format.46  
* **Temporal Memory (TM):** The TM is responsible for learning sequences. While the SP learns *what* a tool is, the TM learns *how* it is used over time.47 It learns to recognize and predict sequences of SDRs generated by the SP. This allows the agent to understand complex, multi-step workflows, such as a sequence of API calls where the output of one call becomes the input for the next. The TM is the mechanism by which the agent learns the temporal patterns inherent in sophisticated tool use.48

### **2.3. Reference Frames and Efference Copies: The Mechanics of Tool Interaction**

To move from passive pattern recognition to active, intelligent interaction, the Gene Atlas incorporates two further principles from TBT: reference frames and efference copies.

* **Reference Frames for Tool Structure:** TBT posits that the brain understands the world by mapping sensory inputs onto reference frames, which are like coordinate systems anchored to objects.32 This allows for the representation of objects' 3D structure and the location of features relative to the object itself, enabling recognition from any viewpoint.54 We apply this concept directly to the structure of a tool's API. For each  
  Tool Gene, a reference frame is established where:  
  * The **tool itself** (e.g., WeatherAPI) is the anchor of the frame.  
  * **Endpoints or functions** (e.g., /getCurrentWeather, /getForecast) are specific locations within this frame.  
  * **Parameters** (e.g., location, units, days) are features stored at these locations.

This structured, object-centric representation is profoundly powerful. It allows the agent to learn a compositional model of the tool, enabling it to reason about the relationships between its components. This is the foundation for generalization: an agent that has learned the reference frame for one REST API can much more quickly learn another, because it has learned the *concept* of a REST API, not just a specific instance.41

* **Efference Copies for Tool Action:** An efference copy is a duplicate of a motor command that the brain uses to predict the sensory consequences of its own actions.56 This predictive mechanism is what allows us to distinguish self-generated sensations from external ones and is crucial for rapid error correction and learning.58 In the ARC-TOOL-LEARNING framework, this process is modeled as follows:  
  1. **Motor Command:** The agent's policy module decides to execute an action, such as calling a tool with specific parameters.  
  2. **Efference Copy:** An internal copy of this API call (the tool name, endpoint, and instantiated parameters) is generated.  
  3. **Sensory Prediction:** This efference copy is fed into the corresponding Tool Gene. The gene's learned model uses this input to generate a *prediction* of the expected API response (the "predicted sensory feedback").  
  4. **Sensory Input:** The tool is actually executed in the external environment, and the result is returned (the "real sensory input").  
  5. **Learning from Mismatch:** The agent's learning mechanism compares the predicted response with the actual response. A match confirms the model's accuracy. A mismatch—a "surprise"—generates a powerful, targeted learning signal that updates the synaptic permanences within the Tool Gene, refining its model of the tool's behavior.58 This allows the agent to learn from its mistakes, adapt to changes in API behavior, and build a robust, predictive model of its toolset.

#### **Table 2.1: Mapping Tool Primitives to Gene Atlas Components**

To ensure a clear and unambiguous implementation path, the following table formally maps the abstract concepts of tool use to their concrete architectural components within the Gene Atlas. This specification serves as a critical bridge from neuro-cognitive theory to engineering practice.

| Tool Primitive | Gene Atlas Component | Description | Relevant Snippets |
| :---- | :---- | :---- | :---- |
| Tool Documentation (OpenAPI Spec) | Input to Spatial Pooler | The raw text/JSON data used to learn the tool's features and static structure, forming a Sparse Distributed Representation (SDR). | 3 |
| Tool Name / Endpoint | Reference Frame Anchor | The root of the object-centric model for the tool, providing a stable coordinate system for its features. | 51 |
| API Parameters (name, type, required) | Features at Locations | Specific attributes (e.g., city: string) learned and stored at discrete locations within the tool's reference frame. | 41 |
| A specific API call (action) | Motor Command | The efferent signal generated by the agent's policy module, representing the intent to act upon the environment. | 56 |
| Predicted API Response | Sensory Prediction | The output generated by the Tool Gene from the efference copy of the motor command, representing the agent's expectation. | 57 |
| Actual API Response | Sensory Input | The ground-truth result received from the external tool environment after the action is executed. | 57 |
| Mismatch (Prediction Error) | Learning Signal | The difference between predicted and actual response, which drives Hebbian-style learning to update synaptic permanences in the Temporal Memory. | 58 |
| Sequence of API Calls | Temporal Sequence | A chain of SDRs representing a multi-step workflow, learned and recognized by the Temporal Memory algorithm. | 47 |

---

## **Section 3: The Agent's Psyche: The State-Dependent Tool Modulation (SDTM) Framework**

A digital artisan does not apply the same technique to every problem. Their approach is modulated by the context: the urgency of the task, the novelty of the material, and the risk of failure. To endow Chimera-1 with this adaptive capability, we introduce the State-Dependent Tool Modulation (SDTM) framework. This framework defines a set of internal cognitive states that dynamically shape the agent's decision-making, tool selection, and learning strategies, creating a more flexible and psychologically plausible policy. This moves beyond simple goal-driven behavior and introduces a layer of cognitive control that is sensitive to the nuances of the situation.61

### **3.1. Defining the Cognitive States of Chimera-1**

The SDTM framework is built around a finite set of discrete, high-level operational modes. These modes are not merely descriptive labels; they are distinct functional states that actively reconfigure the agent's processing priorities and behavioral policies, drawing inspiration from cognitive architectures that model human memory and attention.61

* **The *Manager* State:** This is the agent's default, goal-oriented operational mode. In this state, the agent's behavior is optimized for efficiency and planning. It prioritizes achieving the primary task objective while minimizing resource consumption (e.g., API calls, computational cycles). The agent engages in careful, deliberative reasoning, akin to a project manager balancing scope, time, and budget.66  
* **The *Firefighter* State:** This state is triggered by conditions of high urgency, critical errors, or immediate threats to task success. The *Firefighter* state is characterized by a shift to rapid, decisive action. It prioritizes speed and certainty over long-term optimality. In this mode, the agent will favor tools with low latency and high historical success rates, potentially deferring complex reasoning to resolve the immediate crisis.67 This mirrors the human response to acute stress, where cognitive control narrows focus to the most pressing issue.  
* **The *Explorer* State:** This mode is activated in situations of high uncertainty, such as encountering a novel problem type or experiencing repeated failures with known strategies. The *Explorer* state is defined by a programmatic increase in exploration. The agent will actively try novel tool combinations, experiment with different parameterizations, and prioritize actions that yield the most information about the environment, even if they do not directly advance the primary task goal. This state is a direct implementation of a strategy to manage the exploration-exploitation tradeoff inherent in reinforcement learning.68  
* **The *Artisan* State:** This is a specialized state for tasks that demand high creativity, precision, or the composition of complex, multi-step tool workflows. When in *Artisan* mode, the agent may deliberately slow its execution to perform more detailed reasoning, such as engaging in explicit self-reflection loops or "plan-and-solve" strategies.19 This state values the quality and elegance of the solution over raw speed, analogous to a craftsman taking the time to perfect their work.

### **3.2. The Appraisal Mechanism: From Environment to Emotion**

The transition between these cognitive states is not random; it is governed by an **Appraisal Mechanism**. This module acts as the agent's emotional core, continuously evaluating the environment and the agent's internal status to determine the most appropriate cognitive mode. This is a computational implementation of cognitive appraisal theories from psychology, which hold that emotions arise from an individual's evaluation of a situation's relevance to their goals.71

* **Inputs to Appraisal:** The mechanism monitors a vector of key variables that provide a rich, multi-dimensional view of the current context:  
  * *Task Urgency:* Deadlines, time constraints, or explicit user prompts indicating high priority.  
  * *Error Rate:* The frequency of recent tool failures, negative feedback from the environment, or API error codes (e.g., 4xx, 5xx).  
  * *Uncertainty/Novelty:* A measure of how dissimilar the current task is from previously seen examples, or the number of times the agent has failed to make progress.  
  * *Resource Cost:* The cumulative cost of API calls or computational resources used in the current trajectory.  
  * *Goal Proximity:* An estimate of the number of steps remaining to complete the task.  
* **Output State:** The appraisal mechanism uses a learned function (e.g., a small neural network or a set of rules) to map this input vector to a probability distribution over the cognitive states. For example, a combination of (high urgency, high error\_rate) would produce a high probability of transitioning to the *Firefighter* state. This process is analogous to how cognitive architectures like ACT-R and Soar use internal state representations to guide the selection of cognitive operations and behavioral responses.64

### **3.3. Emotion-Driven Reinforcement Learning for State Modulation**

The agent's cognitive state is a functional component that actively modulates the reinforcement learning process. This represents a practical application of affective computing, where simulated emotions serve a functional purpose in decision-making rather than a purely aesthetic one.72 This modulation occurs in two primary ways:

* **State-Dependent Reward Function:** The cognitive state introduces an *intrinsic motivation* term to the overall reward function used in the RL training phase (Section 5.2). This provides a dense and informative learning signal that goes beyond sparse, binary task success. A simple (success \= \+1, failure \= \-1) reward signal makes the credit assignment problem exceptionally difficult for long-horizon tasks. If a 20-step plan fails, it is nearly impossible to determine which of the 20 actions was the cause. Intrinsic motivation, shaped by the agent's cognitive state, provides immediate, step-by-step feedback. For instance, an action that leads to a high error rate, triggering the "Firefighter" mode, can be associated with an immediate negative intrinsic reward (the "stress" of being in a crisis state). Conversely, an action that resolves an error and allows a transition back to the "Manager" state receives a positive intrinsic reward (the "relief"). This transforms the learning problem into one with a much richer reward landscape, making RL more tractable and effective.  
  * In *Explorer* mode, the agent receives an intrinsic reward for trying new tools or novel parameter combinations, creating a "curiosity drive" that encourages systematic exploration of the state space.78  
  * In *Firefighter* mode, the reward function is heavily weighted towards speed and success, with large penalties for actions that increase latency.  
  * In *Manager* mode, the reward function balances task success with resource efficiency, penalizing profligate or unnecessary tool calls.  
* **State-Dependent Action Selection:** The cognitive state directly influences the agent's action-selection policy (π(action|state)). This can be implemented in several ways:  
  * **Policy Head Switching:** The architecture can employ a mixture-of-experts (MoE) model, where each cognitive state corresponds to a distinct policy head (a specialized sub-network). The output of the appraisal mechanism determines the weighting of these experts for any given decision.  
  * **Modulating Exploration Parameters:** The state can dynamically adjust key RL parameters. The *Explorer* state, for example, would increase the epsilon value in an epsilon-greedy strategy or the temperature for softmax sampling, forcing the agent to behave more stochastically and try new actions when it is stuck.69 The  
    *Firefighter* state would do the opposite, setting epsilon to near-zero to favor exploitation of known, reliable strategies.

This state-driven modulation introduces a form of "machine psychology".80 An agent might develop an "aversion" to a class of tools that frequently cause errors and trigger the negative intrinsic reward of the

*Firefighter* state. While this could lead to suboptimal behavior in some cases, it also makes the agent's failure modes more predictable and interpretable. Instead of inexplicable catastrophic failures, we might observe understandable patterns of "hesitation" or "anxiety," which can be diagnosed and addressed.

#### **Table 3.1: State-Dependent Tool-Use Policies**

This table provides a concrete specification for implementing the SDTM framework, translating the abstract cognitive states into specific, measurable modulations of the agent's policy and reward function.

| Cognitive State | Triggering Conditions | Policy Modulation | Intrinsic Reward Signal |
| :---- | :---- | :---- | :---- |
| **Manager** | Default state; low error, low urgency, clear goal path. | Balanced exploration/exploitation. Prioritize tools based on a learned cost/benefit analysis. | Positive for task progress, negative for resource cost and time. |
| **Firefighter** | High task urgency; critical system errors; high rate of negative feedback. | Drastically reduce exploration (low epsilon/temperature). Prioritize high-success-rate, low-latency tools. Defer complex, multi-step reasoning. | Large positive reward for rapid task completion; large negative penalty for time delay and continued errors. |
| **Explorer** | High uncertainty; novel task type; repeated failures with known toolsets. | Drastically increase exploration (high epsilon/temperature). Prioritize novel tool combinations. Reward actions that reduce state uncertainty. | Positive reward for visiting novel state-action pairs or using new tools (information gain). |
| **Artisan** | Task explicitly flagged for high quality; tasks requiring complex, creative composition of tools. | Allow for longer planning phases. May trigger explicit self-reflection loops (e.g., "plan-and-solve" prompting). | Positive reward for solution quality/elegance (e.g., as judged by a preference model), even at higher computational cost. |

---

## **Section 4: The Training Corpus: Data and Schema**

The foundation of any successful machine learning system is the data upon which it is trained. For the ARC-TOOL-LEARNING curriculum, we require a corpus that is not only large-scale but also diverse, realistic, and structured. This section details our hybrid strategy for generating this corpus and specifies the ActionTranscript-v1.0 schema, a standardized format designed to support all phases of our training regimen.

### **4.1. Corpus Generation Strategy: A Hybrid Approach**

No single data source can provide the necessary breadth, realism, and diversity required to train a digital artisan. Therefore, we will employ a hybrid generation strategy that combines the strengths of simulation, real-world code mining, and infrastructure-as-code analysis.

* **Simulated Data via Multi-Agent Environment:** To achieve the necessary breadth and scale, we will adopt the methodology pioneered by the **ToolAlpaca** framework.3 We will construct a multi-agent simulation environment consisting of:  
  1. A **User Agent** (powered by a large LLM) responsible for generating a wide variety of natural language instructions and user queries.  
  2. An **Assistant Agent** (the Chimera-1 model being trained) that attempts to solve the instructions by generating tool-use trajectories.  
  3. A Tool Executor Agent that simulates the behavior of hundreds of real-world APIs, providing realistic responses, including successes, failures, and noisy data.  
     This automated pipeline allows for the rapid generation of thousands of tool-use instances across a vast landscape of APIs with minimal human intervention, providing the foundational dataset for initial SFT.4  
* **Real-World Data via Code and Configuration Mining:** To ensure our agent is grounded in the practical realities of how tools are used by human developers and operators, we will mine large-scale public data sources for authentic usage patterns.  
  * **GitHub Mining for API Usage:** We will mine public Java and Python repositories on GitHub to extract examples of real-world API and library usage. A naive approach of simply extracting code snippets is insufficient, as they are often cluttered with irrelevant logic. Instead, we will employ **program slicing**, a static analysis technique that identifies and extracts only the subset of program statements relevant to a specific point of interest (e.g., an API call).81 As demonstrated by frameworks like ExampleCheck, program slicing significantly improves the accuracy of mined usage patterns by filtering out unrelated code.82 For Python, we will leverage the built-in  
    ast (Abstract Syntax Tree) module to parse the source code, traverse the tree, and identify Call nodes, thereby extracting function and method call sequences.83 A similar approach using static analysis libraries will be applied to Java code. This provides context-rich, realistic examples of how developers chain tool calls and handle their outputs.82  
  * **Ansible Playbook Mining:** Infrastructure-as-Code (IaC) repositories are a treasure trove of structured tool-use data. Ansible playbooks, written in YAML, explicitly declare sequences of tasks, where each task is a call to a specific module (a tool) with defined parameters.87 We will develop parsers using a Python library like  
    PyYAML to ingest these playbooks and extract the sequences of module calls, their parameters, and their control flow dependencies (e.g., loops, conditionals).89 This provides a unique dataset focused on system orchestration and configuration management tasks.

### **4.2. The ActionTranscript-v1.0 Schema**

To ensure consistency across our diverse data sources and to support the distinct requirements of SFT, RL, and ORPO, all training examples will be standardized into the ActionTranscript-v1.0 schema. This machine-readable JSON format captures the complete context of a tool-use episode, from initial instruction to final outcome. The design is informed by the structures observed in leading tool-use datasets like ToolBench and API-Bank.1

#### **Schema Definition (JSON)**

JSON

{  
  "transcript\_id": "string",  
  "timestamp": "datetime",  
  "source": "string",  
  "agent\_id": "string",  
  "user\_instruction": {  
    "text": "string",  
    "metadata": {}  
  },  
  "agent\_cognitive\_state": "string",  
  "trajectory": \[  
    {  
      "step": "integer",  
      "thought": "string",  
      "action": {  
        "tool\_name": "string",  
        "parameters": "object",  
        "raw\_call": "string"  
      },  
      "observation": {  
        "source": "string",  
        "raw\_output": "string",  
        "parsed\_output": "string",  
        "is\_error": "boolean"  
      }  
    }  
  \],  
  "final\_answer": "string",  
  "outcome\_evaluation": {  
    "success": "boolean",  
    "preference\_score": "float",  
    "efficiency\_metrics": {  
      "tool\_calls": "integer",  
      "latency\_seconds": "float",  
      "cost\_units": "float"  
    }  
  }  
}

#### **Key Fields and Data Types Explained**

* **transcript\_id (string):** A unique UUID for the transcript to ensure traceability.  
* **timestamp (datetime):** An ISO 8601 formatted timestamp indicating when the transcript was generated or logged.  
* **source (string):** An enumerated string indicating the origin of the data ('simulation', 'github\_mining', 'ansible\_mining', 'live\_interaction').  
* **agent\_id (string):** The version identifier of the Chimera-1 model that generated the trajectory, if applicable.  
* **user\_instruction (object):** Contains the initial prompt.  
  * text (string): The natural language instruction from the user.  
  * metadata (object): Optional key-value pairs for additional context.  
* **agent\_cognitive\_state (string):** The cognitive state from the SDTM framework ('Manager', 'Firefighter', 'Explorer', 'Artisan') that is associated with this transcript. For simulated data, this can be a predefined condition. For mined data, it can be inferred from context (e.g., a commit message like "Hotfix for critical bug" would imply a 'Firefighter' state). This field is crucial for training the state-dependent policies.  
* **trajectory (array of objects):** The core of the transcript, representing the sequential thought-action-observation loop inspired by the ReAct framework.21 This structure is essential for all learning phases.  
  * step (integer): The sequence number of the step in the trajectory.  
  * thought (string): The agent's internal monologue or chain-of-thought reasoning that justifies the subsequent action.  
  * action (object): The tool call executed by the agent.  
    * tool\_name (string): The fully qualified name of the tool, module, or API being called.  
    * parameters (object): A JSON object representing the parameters and their values passed to the tool.  
    * raw\_call (string): The exact code or command executed (e.g., requests.get(...), ansible.builtin.apt(...)).  
  * observation (object): The feedback received after the action was executed.  
    * source (string): Where the observation came from ('tool\_executor', 'user\_feedback', 'internal\_error').  
    * raw\_output (string): The raw, unparsed output from the tool (e.g., JSON response, shell output).  
    * parsed\_output (string): A natural language summary of the raw output, generated by the agent.  
    * is\_error (boolean): A flag indicating if the observation represents an error.  
* **final\_answer (string):** The final, user-facing response generated by the agent after completing the trajectory.  
* **outcome\_evaluation (object):** Contains the labels necessary for training and evaluation.  
  * success (boolean): A binary label indicating whether the trajectory successfully fulfilled the user\_instruction. This serves as the primary reward signal for Phase 2 RL.  
  * preference\_score (float): A continuous or discrete score indicating the quality of the trajectory. This is used in Phase 3 to generate (chosen, rejected) pairs for ORPO.  
  * efficiency\_metrics (object): A collection of metrics used for secondary rewards and analysis. Includes tool\_calls (integer), latency\_seconds (float), and cost\_units (float) for API costs.

---

## **Section 5: The Training Curriculum: From Novice to Master**

The ARC-TOOL-LEARNING curriculum is an integrated, three-phase training plan designed to progressively build Chimera-1's tool-use capabilities. This structured progression is essential, as each phase builds upon the capabilities developed in the previous one, systematically transforming the agent from a novice imitator into a strategic, preference-aligned master.

### **Phase 1: Foundational Competency (Supervised Fine-Tuning)**

* **Objective:** The primary goal of this initial phase is to instill foundational competency by teaching the agent the fundamental "grammar" of tool use. This involves learning the correct syntax for API calls, how to populate parameters from a natural language instruction, and the basic reasoning paths for single-tool tasks. The agent learns *what* a valid tool call looks like.  
* **Methodology:** We will employ standard Supervised Fine-Tuning (SFT), a form of imitation learning. The model will be trained on the ActionTranscript-v1.0 corpus generated in Section 4\. The learning task is framed as a next-token prediction problem, where the model is given the user\_instruction as input and is trained to generate the corresponding trajectory and final\_answer as the target output. The loss is computed only on the agent's generated tokens (i.e., the thought, action, and final\_answer fields), not on the environmental observations.  
* **Rationale:** This SFT phase is a critical prerequisite for the more advanced learning stages that follow. It serves to "warm up" the model, moving its weights into a region of the parameter space where it can generate syntactically valid and semantically plausible tool-use trajectories. Attempting to start with reinforcement learning from a randomly initialized or base pre-trained model would be computationally intractable, as the agent would rarely, if ever, stumble upon a valid action sequence by chance. This SFT-first approach is a standard and necessary step in modern LLM alignment pipelines, establishing a robust initial policy before more complex optimization techniques are applied.94

### **Phase 2: Strategic Mastery (Reinforcement Learning)**

* **Objective:** To transition the agent from imitation to strategic decision-making. In this phase, the agent learns a robust and adaptive policy for solving complex, multi-step problems where the optimal path is not known beforehand. The focus shifts from syntactic correctness to strategic effectiveness, including error recovery, efficient resource usage, and context-aware planning.  
* **Environment:** The agent will be trained in a dynamic simulation environment. This environment will present the agent with tasks from our generated corpus and allow it to interact with the Tool Executor Agent. Crucially, to build resilience, the environment will be designed to introduce stochasticity and noise, a key lesson from the **FireAct** framework.21 This includes simulating API latency, intermittent failures (e.g., HTTP 503 errors), and noisy or malformed data returns.  
* **Reward Function Design:** A carefully designed, composite reward function is essential to guide the agent toward the desired behaviors while avoiding "reward hacking," a common pitfall where an agent exploits loopholes in the reward function to achieve high scores without fulfilling the task's true intent.96 Our total reward,  
  Rtotal​, will be a weighted sum of three components:  
  Rtotal​=wtask​⋅Rtask​+wefficiency​⋅Refficiency​+wintrinsic​⋅Rintrinsic​  
  1. **Rtask​ (Task Success Reward):** A sparse, binary reward based on the success field in the ActionTranscript-v1.0 schema. A successful completion of the user's instruction yields a large positive reward (e.g., \+1), while failure results in a negative reward (e.g., \-1).98 This provides the primary signal for goal achievement.  
  2. **Refficiency​ (Efficiency Penalty):** A dense, negative reward applied at each step to encourage parsimonious tool use. This penalty will be proportional to the efficiency\_metrics in the schema, such as the number of tool calls and their associated computational or monetary cost.99 This teaches the agent to solve problems without unnecessary actions.  
  3. **Rintrinsic​ (Intrinsic Motivation Reward):** This novel component, derived from our SDTM framework (Section 3), provides a dense, state-dependent reward signal that shapes the agent's "psychology." The agent's current cognitive state modulates this reward:  
     * In *Explorer* mode, Rintrinsic​ is positive for trying novel tools or parameter combinations, explicitly rewarding curiosity and information-gathering.78  
     * In *Firefighter* mode, Rintrinsic​ provides a large negative penalty for actions that increase time-to-resolution, reinforcing urgency.  
     * In Manager mode, Rintrinsic​ might penalize actions that increase model uncertainty, promoting stable and predictable behavior.  
       This intrinsic reward provides a rich, moment-to-moment feedback signal that is critical for solving the credit assignment problem in long-horizon tasks and for guiding the agent toward developing nuanced, state-appropriate behaviors.71  
* **Exploration-Exploitation Strategy:** The agent's cognitive state, as defined by the SDTM, will directly manage the exploration-exploitation tradeoff.68 When the agent enters the  
  *Explorer* state due to high uncertainty or repeated failure, the RL algorithm will increase its exploration rate (e.g., by increasing the temperature of its sampling policy or the epsilon in an epsilon-greedy approach). This forces the agent to break out of suboptimal loops and try new strategies. Conversely, the *Firefighter* and *Manager* states will favor exploitation of known, reliable action sequences.70

### **Phase 3: Preference Alignment (Odds Ratio Preference Optimization)**

* **Objective:** The final phase of training is to refine the RL-trained policy to align with nuanced, often subjective, human preferences. This goes beyond simple task success to encompass qualities like the elegance, interpretability, safety, and helpfulness of the agent's behavior.  
* **Methodology: Odds Ratio Preference Optimization (ORPO):** While DPO and RLHF are common choices, we will leverage the more advanced and efficient **Odds Ratio Preference Optimization (ORPO)** algorithm.30  
  * **Rationale for ORPO:** ORPO presents several key advantages over its predecessors. First, it is a "monolithic" algorithm that combines SFT and preference alignment into a single, unified training step. Unlike DPO, it does **not** require maintaining a separate, frozen reference model during training. This simplification halves the memory footprint and computational cost (requiring one forward pass per model instead of two), making the training process significantly more efficient.30 Second, its loss function is based on the  
    *odds ratio* rather than the *probability ratio* used by DPO. The odds ratio provides a milder, more stable penalty for disfavored responses, which prevents the model from overly suppressing logits for potentially useful tokens that happen to appear in a rejected response. This makes it particularly well-suited for a monolithic training process where the model is learning domain knowledge and preferences simultaneously.102  
* **Preference Data Generation:** The dataset for this phase will consist of triplets: (prompt, chosen\_trajectory, rejected\_trajectory). This data will be generated by sampling multiple solution trajectories from the RL-trained policy developed in Phase 2\. These trajectories will then be ranked based on a preference model or human labels.  
  * A **chosen** trajectory is one that is successful, efficient, and aligns with the desired cognitive state.  
  * A **rejected** trajectory could be one that:  
    1. Failed to complete the task.  
    2. Succeeded but was highly inefficient (e.g., used an excessive number of tool calls).  
    3. Succeeded but violated the principles of the intended cognitive state (e.g., used a risky, exploratory tool while in *Firefighter* mode).  
    4. Produced a correct but unhelpful or poorly formatted final answer.  
* **ORPO Implementation:** We will implement the ORPO loss function as specified in the original research.30 The total loss,  
  LORPO​, is a combination of the standard SFT negative log-likelihood loss on the chosen response and a unique odds ratio loss term:  
  LORPO​=E(x,yw​,yl​)∼D​\[−logPθ​(yw​∣x)+λ⋅logσ(log1−Pθ​(yw​∣x)Pθ​(yw​∣x)​−log1−Pθ​(yl​∣x)Pθ​(yl​∣x)​)\]  
  Here, yw​ is the chosen response, yl​ is the rejected response, and λ is a hyperparameter that balances the strength of the SFT objective against the preference alignment objective. This monolithic loss function allows the agent to continue refining its core capabilities while simultaneously learning to express them in a way that aligns with human values and expectations.

---

## **Section 6: Evaluation and Deployment Recommendations**

A rigorous and multi-faceted evaluation suite is critical to validate the effectiveness of the ARC-TOOL-LEARNING curriculum and to ensure that the Chimera-1 agent is not only capable but also robust, generalizable, and aligned with its design principles. Following initial deployment, a clear roadmap for continual learning will ensure the agent's skills evolve and improve over time.

### **6.1. A Multi-Faceted Evaluation Suite**

To capture the full spectrum of the agent's abilities, our evaluation will go beyond standard accuracy metrics and incorporate specialized tests for generalization, cognitive state modulation, and robustness.

* **Standard Benchmarks:** To establish a baseline and compare Chimera-1 against the broader state-of-the-art, we will evaluate its performance on established tool-use benchmarks. These include **ToolBench**, which provides a wide range of real-world APIs and complex instructions 91, and  
  **APIBank**, which focuses on API detection, slot filling, and sequencing tasks.92 Performance will be measured using standard metrics like pass rate and win rate against reference models.  
* **Generalization Testing:** A key claim of our architecture is its ability to generalize to unseen tools. To validate this, we will design a specific evaluation suite inspired by the **GenTool** framework.24 This suite will include:  
  * **Zero-to-One Generalization Test:** The agent will be presented with a task for which no suitable tool is initially available. After a few turns, a new, highly relevant tool will be introduced into its toolset. We will measure whether the agent can recognize the utility of the new tool and successfully adapt its strategy to use it.  
  * **Weak-to-Strong Generalization Test:** The agent will be given a task and a toolset containing a "weak" tool (e.g., a tool with limited functionality or higher latency). After establishing a baseline, an enhanced "strong" version of the tool will be added. We will measure whether the agent correctly identifies and switches to the superior tool to solve the task more effectively.  
* **State-Dependent Tool Modulation (SDTM) Evaluation:** To verify that the SDTM framework is functioning as intended, we will create scenarios specifically designed to trigger each of the defined cognitive states.  
  * **Firefighter Test:** We will simulate a high-stakes, time-sensitive task with a high rate of simulated API errors. We will measure whether the agent correctly transitions to the *Firefighter* state and verify that its behavior changes accordingly (e.g., a measurable decrease in tool selection latency, a preference for simpler, more reliable tools).  
  * **Explorer Test:** We will present the agent with a novel problem for which its standard strategies fail repeatedly. We will measure whether it transitions to the *Explorer* state and verify an increase in exploratory behaviors (e.g., trying a wider variety of tools, using tools with novel parameter combinations).  
* **Robustness Testing:** Real-world environments are inherently noisy. Following the methodology from the **FireAct** paper 21, we will systematically test the agent's performance when its tools provide faulty outputs. This will involve injecting noise into the API responses from the  
  Tool Executor, including random data, null responses, and incorrect data types, and measuring the degradation in task success rate. A robust agent should demonstrate graceful degradation and an ability to recover from such errors.

### **6.2. Roadmap for Continual Learning and Self-Evolution**

Deployment is not the end of the learning process; it is the beginning of a continuous feedback loop.

* **In-the-Loop Data Collection:** Post-deployment, all agent interactions with users and tools will be logged using the standardized ActionTranscript-v1.0 schema. This creates a rich, continuous stream of real-world data. Human feedback (e.g., user ratings, corrections) and automated outcome evaluations will be used to automatically label these transcripts, continually enriching our core training dataset.  
* **Autonomous Self-Improvement Loop:** To enable Chimera-1 to improve autonomously, we will implement a self-evolution loop inspired by the **ToolACE-DEV** framework.18 This process will run periodically (e.g., weekly):  
  1. **Challenge Generation:** A set of challenging problems is selected from our evaluation suite or generated dynamically.  
  2. **Self-Correction:** The agent uses its current policy to generate multiple, diverse solution trajectories for each challenge problem.  
  3. **Self-Evaluation:** These generated trajectories are evaluated using a combination of a trained reward model and self-consistency checks (e.g., majority voting on the final answer).  
  4. **Data Augmentation:** The highest-quality, successfully-evaluated trajectories are added to the fine-tuning dataset.  
  5. Retraining: The model is then retrained on this augmented dataset using the ORPO objective.  
     This creates a virtuous cycle where the agent continuously refines its own capabilities, discovering new and more effective strategies over time and reducing the long-term need for manual data curation and supervision.

### **6.3. Concluding Architectural Recommendations**

The ARC-TOOL-LEARNING blueprint presents a novel and comprehensive strategy for developing a truly capable tool-using agent. The success of this initiative hinges on the integrated adoption of its three core pillars. We provide the following final recommendations to the Chimera-1 engineering and research teams:

1. **Embrace the Hybrid Learning Triad:** The sequential "Imitate, Explore, Refine" training curriculum should be adopted as the standard pipeline. Each phase is critical: SFT provides the foundation, RL builds strategic depth, and ORPO ensures nuanced alignment. Bypassing any stage will result in a less capable and less robust agent.  
2. **Commit to the Neuro-Computational Architecture:** The Gene Atlas is more than an academic curiosity; it is a strategic architectural choice designed to solve the fundamental problems of generalization and catastrophic forgetting. Its modular, brain-inspired design, leveraging reference frames and efference copies, is the key to creating an agent that can learn continuously and adapt to a constantly expanding universe of tools.  
3. **Invest in the Agent's Psychology:** The State-Dependent Tool Modulation (SDTM) framework is the mechanism that elevates Chimera-1 from a static executor to a dynamic, adaptive agent. By giving the agent internal states that modulate its behavior, we provide it with the cognitive flexibility to respond appropriately to the full spectrum of real-world challenges. This framework is not only crucial for performance but also provides a powerful tool for making the agent's behavior more interpretable and predictable.

By implementing this blueprint, we will not only teach Chimera-1 to use tools but will cultivate a digital artisan capable of extending human ingenuity in the digital realm.

#### **Works cited**

1. quchangle1/LLM-Tool-Survey: This is the repository for the ... \- GitHub, accessed July 11, 2025, [https://github.com/quchangle1/LLM-Tool-Survey](https://github.com/quchangle1/LLM-Tool-Survey)  
2. Tool Learning with Large Language Models: A Survey arXiv:2405.17935v3 \[cs.CL\] 4 Nov 2024, accessed July 11, 2025, [https://arxiv.org/pdf/2405.17935?](https://arxiv.org/pdf/2405.17935)  
3. AI-Powered Paper Summarization about the arXiv paper 2306.05301v1, accessed July 11, 2025, [https://www.summarizepaper.com/en/arxiv-id/2306.05301v1/](https://www.summarizepaper.com/en/arxiv-id/2306.05301v1/)  
4. (PDF) ToolAlpaca: Generalized Tool Learning for Language Models ..., accessed July 11, 2025, [https://www.researchgate.net/publication/371414066\_ToolAlpaca\_Generalized\_Tool\_Learning\_for\_Language\_Models\_with\_3000\_Simulated\_Cases](https://www.researchgate.net/publication/371414066_ToolAlpaca_Generalized_Tool_Learning_for_Language_Models_with_3000_Simulated_Cases)  
5. tangqiaoyu/ToolAlpaca: the official code for "ToolAlpaca ... \- GitHub, accessed July 11, 2025, [https://github.com/tangqiaoyu/ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca)  
6. \[2504.21769\] LLM-based Interactive Imitation Learning for Robotic Manipulation \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2504.21769](https://arxiv.org/abs/2504.21769)  
7. \[2404.04869\] Prompting Multi-Modal Tokens to Enhance End-to-End Autonomous Driving Imitation Learning with LLMs \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2404.04869](https://arxiv.org/abs/2404.04869)  
8. \[2309.11359\] Prompt, Plan, Perform: LLM-based Humanoid Control via Quantized Imitation Learning \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2309.11359](https://arxiv.org/abs/2309.11359)  
9. Making LLMs Write Better and Better Code for Self-Driving Using LangProp \- Medium, accessed July 11, 2025, [https://medium.com/data-science/making-llms-write-better-and-better-code-for-self-driving-using-langprop-99c6c3dc9508](https://medium.com/data-science/making-llms-write-better-and-better-code-for-self-driving-using-langprop-99c6c3dc9508)  
10. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs \- arXiv, accessed July 11, 2025, [https://arxiv.org/html/2504.11536v1](https://arxiv.org/html/2504.11536v1)  
11. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs \- arXiv, accessed July 11, 2025, [http://arxiv.org/pdf/2504.11536](http://arxiv.org/pdf/2504.11536)  
12. Train a software agent to behave rationally with reinforcement learning \- IBM Developer, accessed July 11, 2025, [https://developer.ibm.com/articles/cc-reinforcement-learning-train-software-agent/](https://developer.ibm.com/articles/cc-reinforcement-learning-train-software-agent/)  
13. Reinforcement Learning: Optimize Decision-Making with Reward-Based Models \- Lyzr AI, accessed July 11, 2025, [https://www.lyzr.ai/glossaries/reinforcement-learning/](https://www.lyzr.ai/glossaries/reinforcement-learning/)  
14. \[2504.11536\] ReTool: Reinforcement Learning for Strategic Tool Use in LLMs \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2504.11536](https://arxiv.org/abs/2504.11536)  
15. \[2501.11651\] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2501.11651](https://arxiv.org/abs/2501.11651)  
16. \[2504.20571\] Reinforcement Learning for Reasoning in Large Language Models with One Training Example \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2504.20571](https://arxiv.org/abs/2504.20571)  
17. Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2503.16219](https://arxiv.org/abs/2503.16219)  
18. ToolACE-DEV: Self-Improving Tool Learning via ... \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2505.07512](https://arxiv.org/abs/2505.07512)  
19. Large Language Models Can Self-Improve in Long-context Reasoning \- arXiv, accessed July 11, 2025, [https://arxiv.org/html/2411.08147v1](https://arxiv.org/html/2411.08147v1)  
20. ToolACE-DEV: Self-Improving Tool Learning via Decomposition and EVolution \- arXiv, accessed July 11, 2025, [https://arxiv.org/html/2505.07512v1](https://arxiv.org/html/2505.07512v1)  
21. FireAct: Toward Language Agent Fine-tuning, accessed July 11, 2025, [https://fireact-agent.github.io/](https://fireact-agent.github.io/)  
22. Fine-Tuning Language Models for AI Agents: FireAct and Beyond \- Ubiai, accessed July 11, 2025, [https://ubiai.tools/fine-tuning-language-models-for-ai-agents-using-ubiai-a-comprehensive-guide-and-walkthrough-to-fireact-and-beyond/](https://ubiai.tools/fine-tuning-language-models-for-ai-agents-using-ubiai-a-comprehensive-guide-and-walkthrough-to-fireact-and-beyond/)  
23. FireAct: Toward Language Agent Fine-tuning | Princeton NLP Group, accessed July 11, 2025, [https://princeton-nlp.github.io/fireact/](https://princeton-nlp.github.io/fireact/)  
24. (PDF) GenTool: Enhancing Tool Generalization in Language Models through Zero-to-One and Weak-to-Strong Simulation \- ResearchGate, accessed July 11, 2025, [https://www.researchgate.net/publication/389392171\_GenTool\_Enhancing\_Tool\_Generalization\_in\_Language\_Models\_through\_Zero-to-One\_and\_Weak-to-Strong\_Simulation](https://www.researchgate.net/publication/389392171_GenTool_Enhancing_Tool_Generalization_in_Language_Models_through_Zero-to-One_and_Weak-to-Strong_Simulation)  
25. GenTool: Enhancing Tool Generalization in Language Models ..., accessed July 11, 2025, [https://arxiv.org/abs/2502.18990](https://arxiv.org/abs/2502.18990)  
26. LLM With Tools: A Survey \- arXiv, accessed July 11, 2025, [https://arxiv.org/pdf/2409.18807](https://arxiv.org/pdf/2409.18807)  
27. \[2409.18807\] LLM With Tools: A Survey \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2409.18807](https://arxiv.org/abs/2409.18807)  
28. LLMs can now self-improve by updating their own weights : r/OpenAI \- Reddit, accessed July 11, 2025, [https://www.reddit.com/r/OpenAI/comments/1lbadjv/llms\_can\_now\_selfimprove\_by\_updating\_their\_own/](https://www.reddit.com/r/OpenAI/comments/1lbadjv/llms_can_now_selfimprove_by_updating_their_own/)  
29. Thousand Brains Principles, accessed July 11, 2025, [https://thousandbrains.org/learn/thousand-brains-principles/](https://thousandbrains.org/learn/thousand-brains-principles/)  
30. Reference-free Monolithic Preference Optimization with Odds Ratio \- arXiv, accessed July 11, 2025, [https://arxiv.org/html/2403.07691v1](https://arxiv.org/html/2403.07691v1)  
31. Using ORPO to Improve LLM Fine-tuning with MonsterAPI, accessed July 11, 2025, [https://blog.monsterapi.ai/using-orpo-to-improve-llm-fine-tuning/](https://blog.monsterapi.ai/using-orpo-to-improve-llm-fine-tuning/)  
32. A Thousand Brains Theory: A Review | by Christophe Pere | TDS Archive \- Medium, accessed July 11, 2025, [https://medium.com/data-science/a-thousand-brains-theory-a-review-3ea6bbeeced0](https://medium.com/data-science/a-thousand-brains-theory-a-review-3ea6bbeeced0)  
33. Does Jeff Hawkins' 1000 brain hypothesis solve a mystery in language? \- Reddit, accessed July 11, 2025, [https://www.reddit.com/r/linguistics/comments/owjsgh/does\_jeff\_hawkins\_1000\_brain\_hypothesis\_solve\_a/](https://www.reddit.com/r/linguistics/comments/owjsgh/does_jeff_hawkins_1000_brain_hypothesis_solve_a/)  
34. Cortical column \- Wikipedia, accessed July 11, 2025, [https://en.wikipedia.org/wiki/Cortical\_column](https://en.wikipedia.org/wiki/Cortical_column)  
35. Cortical Column Networks \- The Smart Robot, accessed July 11, 2025, [https://thesmartrobot.github.io/2021/08/26/thousand-brains.html](https://thesmartrobot.github.io/2021/08/26/thousand-brains.html)  
36. The Thousand Brains Theory of Intelligence \- Numenta, accessed July 11, 2025, [https://www.numenta.com/blog/2019/01/16/the-thousand-brains-theory-of-intelligence/](https://www.numenta.com/blog/2019/01/16/the-thousand-brains-theory-of-intelligence/)  
37. The Thousand Brains Theory \- YouTube, accessed July 11, 2025, [https://www.youtube.com/watch?v=5LFo36g4Lug](https://www.youtube.com/watch?v=5LFo36g4Lug)  
38. The Thousand Brains Theory: A Revolution in Understanding Intelligence \- Tanka, accessed July 11, 2025, [https://www.tanka.ai/blog/posts/the-thousand-brains-theory](https://www.tanka.ai/blog/posts/the-thousand-brains-theory)  
39. (PDF) A thousand brains: toward biologically constrained AI \- ResearchGate, accessed July 11, 2025, [https://www.researchgate.net/publication/353356132\_A\_thousand\_brains\_toward\_biologically\_constrained\_AI](https://www.researchgate.net/publication/353356132_A_thousand_brains_toward_biologically_constrained_AI)  
40. Research Papers \- Numenta, accessed July 11, 2025, [https://www.numenta.com/resources/research-publications/papers/](https://www.numenta.com/resources/research-publications/papers/)  
41. (PDF) The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence, accessed July 11, 2025, [https://www.researchgate.net/publication/387382892\_The\_Thousand\_Brains\_Project\_A\_New\_Paradigm\_for\_Sensorimotor\_Intelligence](https://www.researchgate.net/publication/387382892_The_Thousand_Brains_Project_A_New_Paradigm_for_Sensorimotor_Intelligence)  
42. HTM | Numenta, accessed July 11, 2025, [https://www.numenta.com/resources/htm/](https://www.numenta.com/resources/htm/)  
43. Cortical Learning Algorithm Implemented as HTM Deep Learning Model \- MUK Publications, accessed July 11, 2025, [https://www.mukpublications.com/resources/84.%20Hamid%20Masood%20Khan\_pagenumber.pdf](https://www.mukpublications.com/resources/84.%20Hamid%20Masood%20Khan_pagenumber.pdf)  
44. A Machine Learning Guide to HTM (Hierarchical Temporal Memory) \- Numenta, accessed July 11, 2025, [https://www.numenta.com/blog/2019/10/24/machine-learning-guide-to-htm/](https://www.numenta.com/blog/2019/10/24/machine-learning-guide-to-htm/)  
45. The HTM Spatial Pooler—A Neocortical Algorithm for Online Sparse Distributed Coding, accessed July 11, 2025, [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00111/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00111/full)  
46. nupic-legacy/src/nupic/algorithms/spatial\_pooler.py at master \- GitHub, accessed July 11, 2025, [https://github.com/numenta/nupic/blob/master/src/nupic/algorithms/spatial\_pooler.py](https://github.com/numenta/nupic/blob/master/src/nupic/algorithms/spatial_pooler.py)  
47. Hierarchical Temporal Memory · HierarchicalTemporalMemory.jl, accessed July 11, 2025, [https://oblynx.github.io/HierarchicalTemporalMemory.jl/](https://oblynx.github.io/HierarchicalTemporalMemory.jl/)  
48. nupic-legacy/src/nupic/algorithms/temporal\_memory.py at master \- GitHub, accessed July 11, 2025, [https://github.com/numenta/nupic/blob/master/src/nupic/algorithms/temporal\_memory.py](https://github.com/numenta/nupic/blob/master/src/nupic/algorithms/temporal_memory.py)  
49. Temporal Memory Part 1 (Episode 11\) \- YouTube, accessed July 11, 2025, [https://www.youtube.com/watch?v=UBzemKcUoOk](https://www.youtube.com/watch?v=UBzemKcUoOk)  
50. SP / TM connectivity questions \- Engineering \- HTM Forum, accessed July 11, 2025, [https://discourse.numenta.org/t/sp-tm-connectivity-questions/6801](https://discourse.numenta.org/t/sp-tm-connectivity-questions/6801)  
51. A Thousand Brains \- Wiki \- Sam Breed, accessed July 11, 2025, [https://sambreed.dev/wiki/library/non-fiction/a-thousand-brains](https://sambreed.dev/wiki/library/non-fiction/a-thousand-brains)  
52. The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence \- arXiv, accessed July 11, 2025, [https://arxiv.org/html/2412.18354v1](https://arxiv.org/html/2412.18354v1)  
53. “A Framework for Intelligence and Cortical Function ... \- Numenta, accessed July 11, 2025, [https://www.numenta.com/assets/pdf/research-publications/papers/Companion-paper-to-Thousand-Brains-Theory-of-Intelligence.pdf](https://www.numenta.com/assets/pdf/research-publications/papers/Companion-paper-to-Thousand-Brains-Theory-of-Intelligence.pdf)  
54. The Thousand Brains Theory of Intelligence | by RAM ESHWAR KAUNDINYA | Inside Out: Through the Looking Glass | Medium, accessed July 11, 2025, [https://medium.com/inside-out-through-the-looking-glass/the-thousand-brains-theory-of-intelligence-d5161e8a6007](https://medium.com/inside-out-through-the-looking-glass/the-thousand-brains-theory-of-intelligence-d5161e8a6007)  
55. The Thousand Brains Project \- Numenta, accessed July 11, 2025, [https://www.numenta.com/wp-content/uploads/2024/06/Short\_TBP\_Overview.pdf](https://www.numenta.com/wp-content/uploads/2024/06/Short_TBP_Overview.pdf)  
56. From Neural to Artificial: How Efference Copies Bridge Human and Robot Intelligence, accessed July 11, 2025, [https://gregrobison.medium.com/from-neural-to-artificial-how-efference-copies-bridge-human-and-robot-intelligence-9e6e7cbecfca](https://gregrobison.medium.com/from-neural-to-artificial-how-efference-copies-bridge-human-and-robot-intelligence-9e6e7cbecfca)  
57. Efference copy \- Wikipedia, accessed July 11, 2025, [https://en.wikipedia.org/wiki/Efference\_copy](https://en.wikipedia.org/wiki/Efference_copy)  
58. The role of efference copy in striatal learning \- PMC \- PubMed Central, accessed July 11, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4153469/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4153469/)  
59. Neurophysiological evidence of efference copies to inner speech \- PMC \- PubMed Central, accessed July 11, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5714499/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5714499/)  
60. \[2506.21554\] Finding Similar Objects and Active Inference for Surprise in Numenta Neocortex Model \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2506.21554](https://arxiv.org/abs/2506.21554)  
61. Glossary | What is AI Agent Cognitive Architecture \- Frontline, accessed July 11, 2025, [https://www.getfrontline.ai/glossary/what-is-ai-agent-cognitive-architecture](https://www.getfrontline.ai/glossary/what-is-ai-agent-cognitive-architecture)  
62. Unified Mind Model: Reimagining Autonomous Agents in the LLM Era \- arXiv, accessed July 11, 2025, [https://arxiv.org/html/2503.03459v2](https://arxiv.org/html/2503.03459v2)  
63. Cognitive Architecture, accessed July 11, 2025, [https://cogarch.ict.usc.edu/](https://cogarch.ict.usc.edu/)  
64. Cognitive architectures: Research issues and challenges \- Electrical Engineering and Computer Science, accessed July 11, 2025, [https://web.eecs.umich.edu/\~soar/sitemaker/docs/pubs/cogarch.cogsys08.pdf](https://web.eecs.umich.edu/~soar/sitemaker/docs/pubs/cogarch.cogsys08.pdf)  
65. Cognitive Architectures 10.1 Introduction \- David Vernon, accessed July 11, 2025, [http://www.vernon.eu/publications/20\_Vernon\_CR.pdf](http://www.vernon.eu/publications/20_Vernon_CR.pdf)  
66. (PDF) Goal-Driven Cognition in the Brain: A Computational Framework \- ResearchGate, accessed July 11, 2025, [https://www.researchgate.net/publication/262029408\_Goal-Driven\_Cognition\_in\_the\_Brain\_A\_Computational\_Framework](https://www.researchgate.net/publication/262029408_Goal-Driven_Cognition_in_the_Brain_A_Computational_Framework)  
67. Goal-Driven Cognition and Functional Behavior: The Fundamental-Motives Framework, accessed July 11, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3161125/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3161125/)  
68. What is the exploration vs exploitation trade off in reinforcement learning? \- Scribbr, accessed July 11, 2025, [https://www.scribbr.com/frequently-asked-questions/what-is-the-exploration-vs-exploitation-trade-off-in-reinforcement-learning/](https://www.scribbr.com/frequently-asked-questions/what-is-the-exploration-vs-exploitation-trade-off-in-reinforcement-learning/)  
69. What is the exploration-exploitation tradeoff in reinforcement learning? \- Milvus, accessed July 11, 2025, [https://milvus.io/ai-quick-reference/what-is-the-explorationexploitation-tradeoff-in-reinforcement-learning](https://milvus.io/ai-quick-reference/what-is-the-explorationexploitation-tradeoff-in-reinforcement-learning)  
70. What is the role of exploration in the early stages of reinforcement learning? \- Milvus, accessed July 11, 2025, [https://milvus.io/ai-quick-reference/what-is-the-role-of-exploration-in-the-early-stages-of-reinforcement-learning](https://milvus.io/ai-quick-reference/what-is-the-role-of-exploration-in-the-early-stages-of-reinforcement-learning)  
71. Emotion-Driven Reinforcement Learning \- eScholarship.org, accessed July 11, 2025, [https://escholarship.org/content/qt9jk839mw/qt9jk839mw\_noSplash\_a73d653eda1b7633bca010ebe6a06cfb.pdf](https://escholarship.org/content/qt9jk839mw/qt9jk839mw_noSplash_a73d653eda1b7633bca010ebe6a06cfb.pdf)  
72. (PDF) Emotion in Reinforcement Learning Agents and Robots: A ..., accessed July 11, 2025, [https://www.researchgate.net/publication/316955361\_Emotion\_in\_Reinforcement\_Learning\_Agents\_and\_Robots\_A\_Survey](https://www.researchgate.net/publication/316955361_Emotion_in_Reinforcement_Learning_Agents_and_Robots_A_Survey)  
73. Soar (cognitive architecture) \- Wikipedia, accessed July 11, 2025, [https://en.wikipedia.org/wiki/Soar\_(cognitive\_architecture)](https://en.wikipedia.org/wiki/Soar_\(cognitive_architecture\))  
74. COGNITIVE ARCHITECTURES FOR INTROSPECTING DEEP REINFORCEMENT LEARNING AGENTS, accessed July 11, 2025, [https://baicsworkshop.github.io/pdf/BAICS\_36.pdf](https://baicsworkshop.github.io/pdf/BAICS_36.pdf)  
75. (PDF) Emotion-Driven Reinforcement Learning \- ResearchGate, accessed July 11, 2025, [https://www.researchgate.net/publication/253912208\_Emotion-Driven\_Reinforcement\_Learning](https://www.researchgate.net/publication/253912208_Emotion-Driven_Reinforcement_Learning)  
76. Affective Computing: In-Depth Guide to Emotion AI in 2025 \- Research AIMultiple, accessed July 11, 2025, [https://research.aimultiple.com/affective-computing/](https://research.aimultiple.com/affective-computing/)  
77. Affective computing \- Wikipedia, accessed July 11, 2025, [https://en.wikipedia.org/wiki/Affective\_computing](https://en.wikipedia.org/wiki/Affective_computing)  
78. Play with Emotion: Affect-Driven Reinforcement Learning \- Antonios Liapis, accessed July 11, 2025, [https://antoniosliapis.com/papers/play\_with\_emotion\_affect-driven\_reinforcement\_learning.pdf](https://antoniosliapis.com/papers/play_with_emotion_affect-driven_reinforcement_learning.pdf)  
79. (PDF) A computational model of achievement motivation for artificial ..., accessed July 11, 2025, [https://www.researchgate.net/publication/221456876\_A\_computational\_model\_of\_achievement\_motivation\_for\_artificial\_agents](https://www.researchgate.net/publication/221456876_A_computational_model_of_achievement_motivation_for_artificial_agents)  
80. The bitter lesson for Reinforcement Learning and Emergence of AI Psychology \- Reddit, accessed July 11, 2025, [https://www.reddit.com/r/agi/comments/1izc8tw/the\_bitter\_lesson\_for\_reinforcement\_learning\_and/](https://www.reddit.com/r/agi/comments/1izc8tw/the_bitter_lesson_for_reinforcement_learning_and/)  
81. Program slicing \- Wikipedia, accessed July 11, 2025, [https://en.wikipedia.org/wiki/Program\_slicing](https://en.wikipedia.org/wiki/Program_slicing)  
82. Are Code Examples on an Online Q\&A Forum ... \- Tianyi Zhang, accessed July 11, 2025, [https://tianyi-zhang.github.io/files/icse2018-examplecheck.pdf](https://tianyi-zhang.github.io/files/icse2018-examplecheck.pdf)  
83. How to extract all functions and API calls used in a Python source code? \- Stack Overflow, accessed July 11, 2025, [https://stackoverflow.com/questions/51449914/how-to-extract-all-functions-and-api-calls-used-in-a-python-source-code](https://stackoverflow.com/questions/51449914/how-to-extract-all-functions-and-api-calls-used-in-a-python-source-code)  
84. Extracting the Module and Function Names from Python ASTs \- Arumoy Shome, accessed July 11, 2025, [https://arumoy.me/blogs/python-ast-extract-module-method-names/](https://arumoy.me/blogs/python-ast-extract-module-method-names/)  
85. Deciphering Python: How to use Abstract Syntax Trees (AST) to understand code, accessed July 11, 2025, [https://www.mattlayman.com/blog/2018/decipher-python-ast/](https://www.mattlayman.com/blog/2018/decipher-python-ast/)  
86. Clone-based code method usage pattern mining \- arXiv, accessed July 11, 2025, [https://arxiv.org/pdf/2109.13099](https://arxiv.org/pdf/2109.13099)  
87. Ansible playbooks — Ansible Community Documentation, accessed July 11, 2025, [https://docs.ansible.com/ansible/latest/playbook\_guide/playbooks\_intro.html](https://docs.ansible.com/ansible/latest/playbook_guide/playbooks_intro.html)  
88. Ansible Modules \- How To Use Them Efficiently (Examples) \- Spacelift, accessed July 11, 2025, [https://spacelift.io/blog/ansible-modules](https://spacelift.io/blog/ansible-modules)  
89. python \- How to parse & detect Ansible playbook tasks? \- Stack ..., accessed July 11, 2025, [https://stackoverflow.com/questions/46213890/how-to-parse-detect-ansible-playbook-tasks](https://stackoverflow.com/questions/46213890/how-to-parse-detect-ansible-playbook-tasks)  
90. ansible-content-parser \- GitHub, accessed July 11, 2025, [https://github.com/ansible/ansible-content-parser](https://github.com/ansible/ansible-content-parser)  
91. luban-agi/tool\_use\_benchmark: This is a tool-learning datasets for tool using\! \- GitHub, accessed July 11, 2025, [https://github.com/luban-agi/tool\_use\_benchmark](https://github.com/luban-agi/tool_use_benchmark)  
92. OpenBMB/ToolBench: \[ICLR'24 spotlight\] An open platform for training, serving, and evaluating large language model for tool learning. \- GitHub, accessed July 11, 2025, [https://github.com/OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)  
93. liminghao1630/API-Bank · Datasets at Hugging Face, accessed July 11, 2025, [https://huggingface.co/datasets/liminghao1630/API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank)  
94. Fine-Tuning Techniques \- Choosing Between SFT, DPO, and RFT (With a Guide to DPO), accessed July 11, 2025, [https://cookbook.openai.com/examples/fine\_tuning\_direct\_preference\_optimization\_guide](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide)  
95. DPO Trainer \- Hugging Face, accessed July 11, 2025, [https://huggingface.co/docs/trl/dpo\_trainer](https://huggingface.co/docs/trl/dpo_trainer)  
96. Lil'Log, accessed July 11, 2025, [https://lilianweng.github.io/](https://lilianweng.github.io/)  
97. Reward Hacking in Reinforcement Learning | Lil'Log, accessed July 11, 2025, [https://lilianweng.github.io/posts/2024-11-28-reward-hacking/](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)  
98. DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL, accessed July 11, 2025, [https://www.together.ai/blog/deepswe](https://www.together.ai/blog/deepswe)  
99. Toward a Theory of Agents as Tool-Use Decision-Makers \- arXiv, accessed July 11, 2025, [https://arxiv.org/html/2506.00886v1](https://arxiv.org/html/2506.00886v1)  
100. ORPO: Monolithic Odds Ratio Preference Optimization without Reference Model, accessed July 11, 2025, [https://openreview.net/forum?id=XNzfEFbEJB3](https://openreview.net/forum?id=XNzfEFbEJB3)  
101. \[2403.07691\] ORPO: Monolithic Preference Optimization without Reference Model \- arXiv, accessed July 11, 2025, [https://arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691)  
102. AI Paper Review: ORPO \- Monolithic Preference Optimization without Reference Model, accessed July 11, 2025, [https://dev.to/bullmouse/ai-paper-review-24l4](https://dev.to/bullmouse/ai-paper-review-24l4)  
103. API-Blend: A Comprehensive Corpora for Training and Benchmarking API LLMs \- arXiv, accessed July 11, 2025, [https://arxiv.org/html/2402.15491v2](https://arxiv.org/html/2402.15491v2)