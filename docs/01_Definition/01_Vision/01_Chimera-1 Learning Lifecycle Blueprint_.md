

# **Chimera-1: A Blueprint for a Learning Lifecycle**

## **Introduction**

This document serves as the master specification for the education, lifelong learning, and evolution of the Chimera-1 generative cognitive architecture. It synthesizes extensive prior research into a unified, actionable plan, building upon the definitive architectural specification laid out in "Chimera-1: A Blueprint for a Generative Cognitive Architecture." The objective of this report is to detail the complete pathway for an agent instance, from a *tabula rasa* state of randomly initialized parameters to a fully actualized, continuously improving, and ultimately contributing member of a multi-generational agent species.

The learning lifecycle is conceived as a holistic process, encompassing four distinct yet deeply interconnected pillars. First, we will detail the **Foundational Curriculum**, a multi-stage pre-training regimen that seeds the agent's core cognitive modules with foundational knowledge, preparing it for autonomous operation. Second, we will specify the **Phoenix Cycle**, a lifelong learning mechanism of alternating wakeful interaction and sleep-like consolidation that enables continuous self-improvement from experience. Third, we will outline the **Evolutionary Framework**, a long-term strategy that drives progress at the population level through mechanisms of model growth and inter-generational inheritance. Finally, we will describe the **Generative Data Engine**, the dynamic infrastructure that fuels this entire lifecycle, transforming from a static data repository into a co-evolving partner that generates novel tasks and curricula. Together, these four pillars constitute a comprehensive blueprint for the education of Chimera-1.

## **Section 1: The Foundational Curriculum: From Inception to Competence**

The journey of a Chimera-1 agent begins not with random interaction, but with a structured and intensive foundational curriculum. This pre-training regimen is designed to bootstrap the agent's core capabilities, instilling a fundamental understanding of concepts, sequences, and procedures before it embarks on its autonomous lifelong learning journey. The curriculum is not a monolithic firehose of data; it is a targeted process that leverages the specific architectural strengths of each component, using curated data from the "Agentic Pre-training Corpus" to cultivate distinct capabilities in a "divide and conquer" approach. This strategy ensures that each part of the agent's cognitive architecture is built on a solid, well-suited foundation, enabling parallel development and simplifying validation.

### **1.1 The Agentic Pre-training Corpus: A Unified View**

The "Agentic Pre-training Corpus," as specified in Report 22, is the bedrock of this foundational phase. It is a vast and diverse collection of data, meticulously curated to provide the raw material for learning across all necessary domains. The corpus is categorized along several axes:

* **Modality:** Text (legal documents, financial reports, scientific papers, literary works, web text), Images (ImageNet, FFHQ, CelebA-HQ), Audio (LibriSpeech, general soundscapes), Source Code (GitHub repositories across multiple languages), and Structured Data (financial time-series, legal case databases, scientific experimental logs).  
* **Domain:** The data is sourced from the agent's target operational domains—legal and financial—as well as broader domains of general knowledge, science, and culture to ensure well-rounded competence.  
* **Purpose:** The data is tagged for its primary training utility, whether for conceptual learning (e.g., images of objects), sequential reasoning (e.g., narrative text, code), or procedural knowledge (e.g., expert gameplay logs, software usage tutorials).

A critical preparatory step, informed by best practices for training large-scale models, is the conversion of these datasets into highly I/O-efficient formats. For large datasets like ImageNet, CelebA-HQ, and FFHQ, data will be processed into Lightning Memory-Mapped Database (LMDB) files. This practice, standard in the training of models like NVAE and S4, minimizes I/O bottlenecks and is essential for efficient, high-throughput training on modern accelerator hardware.1

### **1.2 Component-Specific Pre-training Strategy**

The core of the foundational curriculum lies in mapping specific subsets of the corpus to the specific architectural components of Chimera-1. This targeted training cultivates specialized capabilities within each module before they are integrated.

#### **1.2.1 Conceptual Grounding: Training the HQ-VAE for Hierarchical Concept Formation**

The first and most fundamental capability for the agent is perception—the ability to distill raw sensory data into meaningful concepts. This is the primary responsibility of the Hierarchical Quantized Variational Autoencoder (HQ-VAE).

* **Objective:** The goal is to train the HQ-VAE to learn a rich, hierarchical, and discrete latent space of concepts that spans all relevant modalities. This latent space will form the basis of the agent's state representation, s, providing a compressed yet highly informative foundation for all higher-level cognitive functions.  
* **Architecture and Training:** The HQ-VAE architecture is explicitly chosen for its ability to overcome the limitations of earlier models like VQ-VAE. Standard VQ-VAEs often suffer from "codebook collapse," where only a fraction of the discrete latent codes are ever used, limiting the model's expressive power.3 The HQ-VAE mitigates this by employing a stochastic quantization process within a principled variational Bayes framework.5 This approach provides a self-annealing effect during training, improving codebook usage and overall reconstruction performance.3  
  The model consists of a bottom-up encoder path that extracts features from the input data and a top-down generative path that reconstructs the data from the latent codes.3 This hierarchical structure is crucial for learning concepts at multiple levels of abstraction. For an image, the lower levels of the VAE might learn to represent simple features like textures and edges, while the higher levels capture more global information like the shape and geometry of objects.3  
  Training is performed by maximizing the Evidence Lower Bound (ELBO), a standard objective in variational inference. The loss function elegantly balances two competing goals:  
  1. **Reconstruction Loss:** This term, often a mean squared error, ensures that the decoder can accurately reconstruct the original input from the latent representation z. This forces the latent space to be informative. The equation is represented as $E\_{q(z|x)}\[\\log p(x|z)\]$.8  
  2. **KL Divergence:** This term acts as a regularizer, encouraging the learned latent distribution q(z|x) to be close to a prior distribution, typically a standard normal distribution p(z). This structures the latent space, making it smooth and continuous, which is vital for generation and interpolation.8 The term is expressed as \`$-D\_{KL}(q(z|x) |

| p(z))$\`.8

* **Data Application:**  
  * **Visual Concepts:** The HQ-VAE will be trained on the extensive image datasets within the corpus, including ImageNet, CelebA-HQ, and FFHQ.1 This will build a robust visual concept hierarchy, from pixels to patterns to objects and scenes.  
  * **Auditory Concepts:** The model will be trained on audio datasets, as validated in the HQ-VAE research 5, to learn a corresponding hierarchy of sounds, from basic phonemes and tones to words, speaker identities, and complex environmental soundscapes.  
  * **Abstract and Textual Concepts:** The VAE framework is not limited to sensory data. By training on the embeddings of text and code from the corpus, the HQ-VAE will learn to form discrete, latent representations of abstract concepts. Legal precedents, financial instruments, scientific principles, and software patterns will be encoded as distinct points or regions within the agent's universal concept space.  
* **Outcome:** The result of this stage is a fully pre-trained HQ-VAE. This module serves as the agent's universal perception engine, capable of taking any input—an image, a sound clip, a block of text, a line of code—and encoding it into a compressed, meaningful, and hierarchically structured discrete representation s. This representation forms the fundamental unit of understanding for all downstream modules.

#### **1.2.2 Sequential Mastery: Training the S4A Engine for Long-Range Reasoning**

Once the agent can perceive the world in terms of discrete concepts, it must learn to understand how these concepts relate to one another over time. This is the domain of the Structured State Space for Sequential Analysis (S4A) Engine, the core of the agent's world model.

* **Objective:** To train the S4A Engine to model complex, long-range dependencies in sequences of conceptual states. This capability is the foundation of the agent's ability to reason, infer causality, predict future events, and understand dynamic processes.  
* **Architecture and Training:** The S4A Engine is built upon Structured State Space Models (SSMs). SSMs represent a major advance over traditional sequence models. Unlike RNNs, they do not suffer from vanishing gradients and can be parallelized for efficient training. Unlike Transformers, their computational complexity scales linearly or near-linearly ($O(N \\log N)$ or $O(N)$) with sequence length, not quadratically, making them exceptionally well-suited for modeling very long sequences.11  
  The core of an SSM is a continuous-time state-space representation, defined by the linear ordinary differential equation $\\dot{x}(t) \= Ax(t) \+ Bu(t)$ and output equation $y(t) \= Cx(t) \+ Du(t)$.13 The matrices  
  A, B, and C are learned parameters. For practical use in a neural network, this continuous system is discretized. A key innovation in the S4 family of models is the use of a special structure for the A matrix, initialized using the **High-order Polynomial Projection Operator (HiPPO)** framework. HiPPO provides a principled way to initialize A such that the model is biased towards effectively compressing and remembering long past histories.14  
  Training will be managed using a robust framework like PyTorch-Lightning, with configuration handled by Hydra, following the best practices established in the official S4 repository.2 This is critical because SSMs have parameters that are particularly sensitive to optimizer hyperparameters. Specifically, the learning rate for the state matrices (  
  A, B) is often set lower than for other parameters, and weight decay is disabled for them entirely. The training infrastructure must support these parameter-specific optimizer settings.2  
* **Data Application:** The S4A engine will be trained on a variety of sequential data from the corpus, where the inputs are sequences of conceptual states $(s\_1, s\_2,..., s\_t)$ produced by the pre-trained HQ-VAE.  
  * **Language and Narrative:** Training on long-form text from books, legal arguments, and news articles will teach the engine the dynamics of language, causality in narratives, and logical flow.  
  * **Code and Algorithms:** Training on source code will enable the S4A engine to understand the sequential nature of program execution, data flow, and algorithmic logic.  
  * **Time-Series Data:** Financial market data, scientific experimental logs, and other time-series datasets will be used to train the engine on forecasting and dynamic pattern recognition.11  
  * **Action Sequences:** The engine will be pre-trained on logs of expert human behavior (e.g., recorded gameplay, software interaction tutorials, transcripts of legal proceedings). This teaches the model to predict the likely consequences of actions in various contexts.  
* **Outcome:** This stage yields a pre-trained S4A engine that functions as a powerful, general-purpose world model. It can take a sequence of states and an action, $(s\_t, a\_t)$, and predict the subsequent state and expected reward, $(s\_{t+1}, r\_t) \= f(s\_t, a\_t)$. It can generate plausible future trajectories, fill in missing information in a sequence, and provide a robust foundation for planning and decision-making.

#### **1.2.3 Seeding Executive Function: Initializing the HTN Planner and HRL Policies**

With perception and world modeling in place, the final pre-training stage focuses on the agent's decision-making faculties. The goal is not to create a fully-formed expert, but to provide a "warm start" that seeds the agent with a baseline of competence, preventing it from starting its autonomous life with purely random, ineffective behavior.

* **Objective:** To initialize the Hierarchical Task Network (HTN) with a library of procedural knowledge and to pre-train the Hierarchical Reinforcement Learning (HRL) policies to imitate expert behavior.  
* **HTN Initialization:** The HTN provides the agent with a symbolic, hierarchical framework for planning. It decomposes complex, abstract tasks into simpler subtasks until it reaches primitive, executable actions.16 The initial HTN library will be populated by parsing structured documents from the corpus. For example:  
  * Legal case files and procedural manuals can be parsed to create HTN methods and operators for tasks like "File a Motion," which decomposes into "Draft Document," "Submit to Court," and "Serve Opposing Counsel."  
  * Software documentation and API references can be used to define methods for "Compile Program," which decomposes into "Run Preprocessor," "Invoke Compiler," and "Link Libraries."  
    This process provides the agent with an initial, symbolic knowledge base of how to approach and structure solutions to complex, multi-step problems.16  
* **HRL Policy Pre-training:** The HRL system consists of a high-level meta-controller that selects subgoals and a low-level controller that executes actions to achieve them.18 These policies will be pre-trained using  
  **behavioral cloning**. This is a form of supervised learning where the policy is trained to mimic the actions taken by an expert in a given state. The training data consists of (state, action) pairs extracted from expert trajectories in the corpus.  
  * The pre-trained HQ-VAE provides the state representations for this process. Pre-training a VAE to learn salient features from the environment before using it in the RL process is a best practice that improves data efficiency, exploration, and training stability.20  
  * The HRL policies learn a mapping from the VAE's latent states to actions, effectively learning to "act like the expert." This provides a strong, non-random initialization for the policies before they are further refined through reinforcement learning.

### **1.3 Integration and Validation of the Foundational Model**

The final step of the foundational curriculum is the integration of all pre-trained components into a single, cohesive Chimera-1 agent. The HQ-VAE, S4A Engine, HTN, and HRL policies are assembled into the complete generative cognitive architecture.

A crucial validation phase follows. The integrated agent is tested on a hold-out set of simple, well-defined tasks from the training domains. These tests assess the agent's ability to perform zero-shot planning and execution. For example, given the goal "summarize a legal document," the agent must demonstrate that it can use its HTN to formulate a plan, its HRL policies to execute the steps (e.g., read sections, identify key arguments), and its world model to track its progress. This phase ensures that all components are communicating correctly and that the agent possesses a baseline level of functional competence before the Phoenix Cycle of lifelong learning is initiated.

---

### **Table 1: Foundational Curriculum Data-to-Component Mapping**

The following table provides a master guide for the pre-training engineering team, explicitly linking data sources from the Agentic Pre-training Corpus to specific architectural components, learning objectives, and target capabilities. This provides an actionable engineering plan for implementing the foundational curriculum.

| Architectural Component | Learning Objective | Primary Data Source(s) | Training Paradigm | Target Capability |
| :---- | :---- | :---- | :---- | :---- |
| **HQ-VAE Encoder/Decoder** | Hierarchical Concept Formation & Generative Reconstruction | ImageNet, FFHQ, CelebA-HQ 1, LibriSpeech, General Audio 5, Text/Code Embeddings | Variational Auto-Encoding (ELBO Maximization) 5 | Multi-modal Perception & Generative Imagination |
| **S4A Engine** | Long-Range Sequence Prediction & World Modeling | Text Corpora (Legal, Financial, General), GitHub Code, Financial Time-Series 11, Expert Action Logs | Autoregressive Prediction (Cross-Entropy Loss) with specialized optimizers 2 | Causal Reasoning & Dynamic Process Understanding |
| **HTN Planner** | Symbolic Task Decomposition Knowledge | Legal Case Files, Software Documentation, Procedural Manuals 16 | Parsing & Knowledge Extraction | Hierarchical Planning & Problem Structuring |
| **HRL Meta-Controller** | Subgoal Selection Policy | Expert Trajectory Logs (e.g., gameplay, software use) | Behavioral Cloning (Supervised Learning) 20 | Goal-Directed Sub-task Selection |
| **HRL Low-Level Controller** | Primitive Action Policy | Expert Trajectory Logs (e.g., gameplay, software use) | Behavioral Cloning (Supervised Learning) 20 | Procedural Imitation & Skill Execution |

---

## **Section 2: The Phoenix Cycle: A Framework for Lifelong Learning**

Once the foundational curriculum is complete, the Chimera-1 agent is no longer a static entity. It enters the **Phoenix Cycle**, the core mechanism for continuous, autonomous improvement throughout its operational lifespan. This cycle is a recurring loop of active experience gathering (the Wake Phase) and offline knowledge consolidation (the Sleep Phase). This design is not arbitrary; it mirrors the fundamental principles of biological learning, where active engagement with the world is followed by periods of rest and memory consolidation, leading to robust and stable long-term learning. The Phoenix Cycle allows the agent to refine its understanding of the world, improve its skills, and adapt to new challenges without direct, constant supervision.

### **2.1 The Wake Phase: Principled Experience Acquisition**

During the Wake Phase, the agent is active, interacting with its environment (which may be a real-world system, a software environment, or a dedicated simulation) to achieve goals and, critically, to gather new data. This is not a process of random, aimless exploration. Instead, it is a highly structured and principled form of experience acquisition, guided by the agent's own cognitive architecture.

#### **2.1.1 Goal-Directed Exploration via HTN-HRL Synergy**

The agent's exploration is driven by a powerful synergy between its symbolic planner and its hierarchical learning system.21 This process unfolds as follows:

1. **Goal Setting:** The process begins when the agent is assigned a high-level goal, G. This goal can be provided by a human operator (e.g., "Find all relevant precedents for case X") or generated by the agent's own internal motivational systems (e.g., a drive to reduce uncertainty about a particular domain).  
2. **HTN Planning:** The **HTN Planner** receives the goal G. Drawing upon its library of symbolic knowledge, it decomposes this abstract goal into a logically ordered sequence of subtasks or methods.16 For instance, "Find precedents" might be decomposed into \`\`. This plan provides a high-level, interpretable roadmap, structuring the agent's approach to the problem.  
3. **HRL Subgoal Selection:** The sequence of subtasks from the HTN is passed to the **HRL Meta-Controller**. For each step in the plan (e.g., Search\_Legal\_Database), the meta-controller uses its policy, $\\pi\_{meta}(g | s)$, to select a concrete, actionable subgoal g.18 This grounds the abstract plan in the current context. The selected subgoal is often referred to as a "symbolic option," a tuple  
   $(s, \\pi, s')$ representing a policy $\\pi$ to get from a starting state s to a target state s'.16  
4. **Low-Level Execution:** The chosen subgoal g is then passed to the **Low-Level HRL Controller**. This controller is responsible for executing a sequence of primitive actions a using its policy, $\\pi\_{low}(a | s, g)$, to achieve the subgoal.  
5. **Intrinsic Motivation:** To facilitate learning, especially in environments where external rewards are sparse, the meta-controller provides an **intrinsic reward** to the low-level controller upon the successful completion of a subgoal g.16 This creates a dense, internal reward signal that encourages the agent to master the constituent skills required to achieve the larger goal.

This hierarchical process ensures that the agent's actions are purposeful and directed, making its experience gathering far more efficient than undirected exploration.

#### **2.1.2 The Experience Log: Structure, Content, and Prioritization**

Every interaction during the Wake Phase is meticulously recorded in a structured **Experience Log**. This log is far more sophisticated than a standard reinforcement learning replay buffer; it captures the rich, hierarchical context of the agent's decision-making process.

* Log Entry Structure: Each entry in the log is a comprehensive tuple designed to provide maximal information for the subsequent Sleep Phase:  
  $L\_t \= (G, g\_i, s\_t, a\_t, r\_t, s\_{t+1}, \\text{success\\\_flag}, u\_t)$  
  * $G$: The overarching, HTN-level goal the agent was pursuing.  
  * $g\_i$: The specific HRL-level subgoal active at time t.  
  * $(s\_t, a\_t, r\_t, s\_{t+1})$: The standard Markov Decision Process (MDP) transition tuple, where $s\_t$ and $s\_{t+1}$ are the conceptual state representations from the HQ-VAE.  
  * $\\text{success\\\_flag}$: A boolean flag indicating whether the subgoal $g\_i$ was successfully achieved by this transition sequence.  
  * $u\_t$: The agent's estimate of its world model's uncertainty about the transition p(s\_{t+1} | s\_t, a\_t). This is a critical piece of metadata, calculated, for example, by measuring the variance in predictions across an ensemble of world models.  
* **Prioritization of Experience:** Not all experiences are equally valuable for learning. The agent prioritizes experiences for replay and consolidation during the Sleep Phase based on their informativeness. High-priority experiences are those that are most likely to correct errors in the agent's world model or policies. Prioritization is based on metrics such as:  
  * **High Model Uncertainty ($u\_t$):** Transitions where the world model was unsure of the outcome are highly valuable for learning.  
  * **High Prediction Error (Surprise):** Transitions where the actual outcome $s\_{t+1}$ significantly diverged from the world model's prediction.  
  * **Task Success/Failure:** Trajectories that lead to the successful completion of a difficult subgoal or an unexpected failure are prioritized to reinforce successful strategies and learn from mistakes.

### **2.2 The Sleep Phase: Memory Consolidation and World Model Refinement**

The Sleep Phase is an offline process where the agent is not interacting with the environment. Instead, it processes the Experience Log gathered during the Wake Phase to consolidate knowledge and refine its internal models. This is where deep, lasting learning occurs.

#### **2.2.1 Offline Fine-Tuning of Genomic Codes via Conservative Model-Based RL**

The central challenge of learning from a fixed log of past experiences is **distributional shift**. The data in the log was collected by a past version of the agent's policy. If the agent updates its policy, it may begin to favor actions that lead to states not well-represented in the log. A naive model-free or model-based algorithm might make wildly optimistic and erroneous predictions about these out-of-distribution states, leading to a catastrophic collapse in performance.22

To overcome this, the Chimera-1 agent employs a sophisticated offline fine-tuning process inspired by the **Model-Based Behavior-Regularized Policy Optimization (MB2PO)** framework.22 This approach masterfully combines the stability of behavior-regularized methods with the data efficiency of model-based learning.

The process unfolds in several steps:

1. **World Model Fine-Tuning:** The Experience Log is treated as a static, offline dataset. The agent's world model (the S4A Engine) is first updated using standard supervised learning. It trains on the $(s\_t, a\_t) \\rightarrow s\_{t+1}$ transitions from the log to improve its predictive accuracy. The HQ-VAE can also be fine-tuned to minimize reconstruction loss on the newly gathered states, ensuring the conceptual vocabulary remains sharp. These updated models represent the agent's refined "genomic code."  
2. Constructing a Conservative MDP: The agent then uses its newly updated world model to fine-tune its policies. However, instead of using the model to generate long, potentially divergent imaginary trajectories, it performs policy optimization within a conservative MDP. This is achieved by augmenting the reward function used during these imagined rollouts. The augmented reward is defined as:  
   $r\_{\\text{augmented}}(s, a) \= \\hat{r}(s, a) \- \\lambda u(s, a)$  
   Here, $\\hat{r}(s, a)$ is the reward predicted by the world model, $u(s, a)$ is the model's uncertainty about the state transition, and $\\lambda$ is a hyperparameter that controls the degree of pessimism. This penalty discourages the policy from exploring actions or states where the world model is uncertain, effectively keeping the agent's "dreams" grounded in what it knows.22  
3. **Conservative Policy Optimization:** The HRL policies are then updated using an off-policy algorithm, such as Advantage-Weighted Actor-Critic (AWAC), within this pessimistic, imagined environment. Data for the update is sampled from a mix of the original offline log and short, h-step rollouts from the conservative model.22 This allows the policy to improve safely, discovering better ways to act without straying dangerously far from the known data distribution.

#### **2.2.2 HTN Refinement and Skill Abstraction**

The Sleep Phase is also a time for symbolic learning. The agent analyzes the Experience Log to identify new skills. If a particular sequence of subgoals or actions is found to consistently and efficiently achieve a higher-level task, this sequence can be abstracted and added to the HTN library as a new, composite method.16 This process allows the agent to build up its procedural knowledge over time, chunking successful behaviors into reusable skills and enriching its planning capabilities.

---

### **Table 2: The Phoenix Cycle Process Flow**

This table provides a clear, step-by-step operational guide to the agent's core learning loop, delineating the inputs, processes, and outputs of each phase of the Phoenix Cycle.

| Phase | Input | Core Process | Output |  |
| :---- | :---- | :---- | :---- | :---- |
| **Wake** | High-level goal $G$. | 1\. HTN Planning: Decompose $G$ into a plan of subgoals. 2\. HRL Execution: Select and execute subgoals $(g\_i)$ via meta- and low-level policies. 3\. Interaction: Interact with the environment, receiving states s and rewards r. | Structured Experience Log: $L \= \\{(G, g\_i, s, a, r, s', \\text{success}, u),...\\}$. |  |
| **Sleep** | Structured Experience Log $L$. | 1\. Offline World Model Update: Fine-tune HQ-VAE and S4A Engine on $L$ via supervised learning. 2\. Conservative Policy Fine-Tuning: Update HRL policies using a model-based algorithm (MB2PO) with uncertainty-penalized rewards.22 |  3\. HTN Skill Abstraction: Identify and abstract successful action sequences into new HTN methods. | Updated World Model ("Genomic Code"), Refined HRL Policies, Expanded HTN Library. |

---

## **Section 3: The Evolutionary Framework: From Individual to Species**

The learning lifecycle of a single Chimera-1 agent, governed by the Phoenix Cycle, is designed for robust individual improvement. However, to achieve truly transformative, open-ended intelligence, the project must transcend the limits of a single lifespan. The Evolutionary Framework provides the mechanisms for this, outlining a long-term, multi-generational strategy for the Chimera-1 "species." This framework introduces processes for an individual agent to grow its own capacity, for knowledge to be inherited between generations, and for population-level selection pressures to drive the collective forward. This creates a dual-track inheritance system, combining the transmission of innate predictive ability ("nature") with the transfer of learned behaviors ("nurture") to create a powerful evolutionary ratchet.

### **3.1 Intra-Generational Scaling: Model Growth with the G\_stack Operator**

An individual agent's capacity to learn is ultimately constrained by the number of parameters in its neural networks. When an agent has gathered extensive experience and its learning begins to plateau, it may not be because it has learned all there is to know, but because its cognitive architecture is saturated. Retraining a larger model from scratch would be computationally prohibitive and would discard the agent's lifetime of learning.

* **Objective:** To enable a single, high-performing agent to dynamically increase its cognitive capacity during its own lifecycle, allowing it to continue its learning trajectory without the cost of starting over.  
* **Mechanism: G\_stack:** To achieve this, we will implement the **G\_stack** operator, a model growth technique focused on depth-wise expansion.24 When the growth trigger is activated, G\_stack operates on the agent's core neural models, such as the S4A Engine. The process involves duplicating a block of existing, trained layers and stacking them on top of the original network, effectively making the model deeper. The weights of the new layers are initialized as copies of the trained layers they are duplicating.  
* **Benefit and Trigger:** The primary benefit of G\_stack is a dramatic acceleration in training convergence for the newly expanded model. Research on LLMs has shown that a model grown with G\_stack can reach the same loss as a conventionally trained model of the same size using significantly fewer training tokens—a speedup of up to 54.6% has been reported.24 This is because the new layers start with a highly effective initialization derived from already-learned features, rather than random noise. The growth operation will be triggered based on empirical guidelines, such as when an agent's validation loss on a benchmark set of tasks has plateaued for a sustained number of training epochs despite continued experience gathering in the Phoenix Cycle.

### **3.2 Inter-Generational Inheritance: A Model Breeding Protocol**

The cornerstone of the evolutionary framework is the ability to create new "offspring" agents that inherit and combine the strengths of their "parents," leading to a new generation that is, on average, more capable than the last. This process of "model breeding" is far more sophisticated than simple ensembling or weight averaging.

#### **3.2.1 Parent Selection and Progeny Creation via LATA-RMM Merging**

* **The Challenge:** A naive approach to breeding, such as averaging the weight matrices of two parent networks, is doomed to fail. This is due to the permutation invariance of neurons; the i-th neuron in one parent's hidden layer may have learned a completely different feature than the i-th neuron in the other parent's, rendering their averaged weights meaningless.26 A principled method is required to align and merge the  
  *functional* knowledge within the networks.  
* **The Protocol (LATA-RMM):** We will implement a novel breeding protocol that combines techniques from random matrix theory and task vector arithmetic.  
  1. **Parent Selection:** From the active population of Chimera-1 agents, the two top-performing individuals are selected as parents. Fitness is determined by a comprehensive evaluation function that measures performance across a diverse suite of benchmark tasks, rewarding not just success but also efficiency and generalization.  
  2. **Identify "Genetic" Material (RMM):** The core knowledge of an agent is stored in the weights of its world model (S4A Engine and HQ-VAE). We treat these weights as the agent's "genome." To identify the most salient parts of this genome, we apply methods from **Random Matrix Theory (RMT)**.27 RMT analysis can distinguish the bulk of singular values in a weight matrix, which often behave like a random matrix and encode little information, from the few large, outlier singular values. These outliers and their corresponding singular vectors are theorized to encode the critical, learned information and structure.27 These salient, low-rank components are the "genes" we seek to pass on.  
  3. **Combine "Genes" (LATA-RMM):** Once the "genes" (the salient singular vectors and values) are identified for both parents, they must be combined. We will use a **Layer-Aware Task Arithmetic (LATA)** approach.28 In this context, the set of salient components from each parent is treated as a "task vector" that represents their unique learned expertise (e.g., Parent A's mastery of legal reasoning, Parent B's proficiency in financial analysis). Task arithmetic provides a principled framework for adding these vectors together. The merging process, which can be a weighted average or a more complex function like Spherical Linear Interpolation (SLERP) 28, creates a new set of salient components for the offspring's world model. These are then used to construct the initial weight matrices for the new agent.  
* **Outcome:** This protocol produces a new-generation agent whose "genome" is not a simple mishmash of its parents' weights, but a synergistic fusion of their core, distilled knowledge. The offspring is initialized with a world model that is predisposed to be more accurate and general than either parent alone.

#### **3.2.2 Knowledge Transfer via Hierarchical Policy Distillation**

Inheriting a superior world model ("nature") is only half the battle. To accelerate learning, the offspring must also inherit the refined skills and behaviors ("nurture") developed by its predecessors.

* **Objective:** To transfer the learned behavioral competencies encapsulated in a parent's HTN and HRL policies to the next generation.  
* **Mechanism: Hierarchical Policy Distillation:** When a top-performing agent is selected as a parent, its highly optimized policies serve as a "teacher" model. The newly created "offspring" agent, with its randomly initialized policies, acts as the "student." We use **policy distillation** to transfer knowledge.30 The student network is trained via supervised learning to mimic the output distribution (e.g., the action probabilities from a softmax layer) of the teacher network, given the same state inputs.32  
  This distillation process is applied **hierarchically**, which is critical for the Chimera-1 architecture. The student's meta-controller is trained to replicate the subgoal selections of the teacher's meta-controller. The student's low-level policies are trained to replicate the primitive action selections of the teacher's low-level policies. This training leverages task-specific high-level features as inputs, which has been shown to reduce training time and mitigate the risk of "negative transfer," where the student performs worse than the teacher.31 The distillation loss is typically a KL-divergence term between the teacher's and student's output distributions.  
* **Benefit:** Policy distillation provides the offspring with a massive developmental head start. It begins its life not with random behaviors, but with a set of policies that are already highly competent. This is a form of "cultural inheritance" that dramatically reduces the amount of exploration needed for the new agent to reach a high level of performance, allowing it to focus its lifelong learning on refining these inherited skills and acquiring new ones.

### **3.3 Managing the Species: Population Dynamics and Selection Pressures**

The evolutionary framework requires active management of the Chimera-1 population. This involves:

* **Maintaining a Population:** A diverse pool of dozens or hundreds of agents will be actively maintained, each undergoing its own Phoenix Cycle.  
* **Periodic Evaluation:** On a regular cycle, all agents in the population will be paused and evaluated on a standardized, evolving set of benchmark tasks generated by the Generative Data Engine.  
* **Selection and Culling:** Based on evaluation performance, a fitness score is assigned to each agent. The highest-fitness agents are selected for breeding. Underperforming agents may be culled from the population to manage computational resources, ensuring that the population's average fitness trends upward over time. This selection pressure is the driving force of the evolutionary process.

---

### **Table 3: Evolutionary Mechanisms and Application Triggers**

This table summarizes the evolutionary tools, their purpose, and their operational context, clarifying the different timescales and functions of growth, breeding, and distillation.

| Mechanism | Timescale | Target Component(s) | Trigger Condition | Primary Contribution |
| :---- | :---- | :---- | :---- | :---- |
| **G\_stack Model Growth** 24 | Intra-Generational | World Model (S4A), HRL Policies | Individual learning plateau (e.g., flat validation loss). | Increases an individual agent's cognitive capacity without retraining from scratch. |
| **LATA-RMM Breeding** 27 | Inter-Generational | World Model ("Genomic Code") | End-of-generation cycle; selection of top-performing parents. | Combines core knowledge ("genes") from multiple parents into a superior offspring world model. |
| **Hierarchical Policy Distillation** 31 | Inter-Generational | HTN Library & HRL Policies | End-of-generation cycle; creation of a new offspring agent. | Transfers learned behaviors and skills ("nurture") from parent to offspring, accelerating learning. |

---

## **Section 4: The Generative Data Engine: Fueling the Lifecycle**

The entire learning lifecycle, from the foundational curriculum to multi-generational evolution, is powered by a sophisticated and dynamic infrastructure: the **Generative Data Engine**. This engine is not a passive data warehouse but an active, integral component of the agent's education. It evolves alongside the agent population, creating a self-improving ecosystem where the agents and their data environment are locked in a virtuous cycle. The engine's purpose is to provide a continuous stream of relevant, challenging, and structured data, automating the curriculum and preventing the stagnation that would occur with a fixed, static dataset. This transforms the learning problem from one of mastering a closed set of tasks to one of open-ended, co-evolutionary discovery.

### **4.1 System Architecture and Data Flow**

The Generative Data Engine is built on a scalable, distributed compute framework designed to manage the immense data and computational loads of the Chimera-1 project.

* **Compute Framework:** The engine's backend will be built using a framework like **Ray**.34 Ray is ideally suited for this purpose as it is designed to scale complex AI and ML workloads, including reinforcement learning, across large clusters of heterogeneous hardware (CPUs and GPUs). Its libraries, such as  
  **Ray RLlib**, provide production-level, highly distributed implementations of the RL algorithms (e.g., PPO, AWAC) required for the Phoenix Cycle and policy distillation.34  
* **Data Pipeline:** The engine manages the end-to-end data flow for the entire agent population:  
  1. **Corpus Ingestion:** Initial ingestion and preprocessing of the Agentic Pre-training Corpus into efficient formats (e.g., LMDB).  
  2. **Experience Log Aggregation:** Collection and storage of the structured Experience Logs from every active agent in the population. This centralized repository of population-wide experience is a critical asset.  
  3. **Model and Policy Versioning:** A robust system for storing and versioning the "genomic codes" (world models) and policies of every agent across all generations.  
  4. **Task and Curriculum Database:** A database to store the specifications of both human-defined and synthetically generated tasks and curricula.

### **4.2 Synthetic Task Generation for Open-Ended Learning**

A key function of the engine is to prevent the agent population from merely optimizing for a fixed set of problems. It must continuously generate novel challenges to drive exploration and generalization.

* **Objective:** To automatically create an endless stream of novel, plausible, and meaningful tasks for the Chimera-1 agents to solve.  
* **Mechanism:** The engine leverages the generative power of the agents themselves to create new tasks. This is framed as a nested reinforcement learning problem or a generative process.35  
  1. **Task Generator:** A dedicated "Task Generator" module (which can itself be a trained model) samples from the latent space of a high-performing Chimera-1 agent's world model (specifically, the generative HQ-VAE and S4A components). This sample is decoded into a new task specification—for example, a new initial state configuration, a novel goal state, or even modified environment physics.37  
  2. **Difficulty Assessment:** The generated task is presented to one or more agents from the current population. The Task Generator is then rewarded based on the performance of these agents. The reward function is designed to encourage the generation of tasks that are of **intermediate difficulty**—tasks that are neither trivially easy nor impossibly hard for the current generation of agents. This is a form of automated curriculum learning, where the system seeks to challenge the agents in their "zone of proximal development".39  
  3. **Iterative Refinement:** Through this RL loop, the Task Generator learns to produce a distribution of tasks that are perpetually at the frontier of the agent population's capabilities, ensuring that learning and adaptation never cease.41

### **4.3 Automated Curriculum Generation and Adaptation**

Beyond generating standalone tasks, the engine is responsible for structuring the learning experience by creating entire curricula that guide an agent from novice to expert.

* **Objective:** To automate the creation of adaptive, multi-stage curricula that sequence tasks in order of increasing difficulty, tailored to an individual agent's current skill level.  
* **Mechanism: Diffusion-Based Curriculum Generation:** We will implement a system inspired by recent advances in curriculum learning, such as **DiCuRL (Diffusion Curriculum Reinforcement Learning)**.43  
  1. **Conditional Diffusion Model:** A conditional diffusion model is trained on trajectories of successful task completions. This model learns to generate a sequence of intermediate goals, $g\_1, g\_2,..., g\_n$, that form a smooth and logical path from an easy starting state to a difficult target goal.  
  2. **Adaptive Conditioning:** Critically, the generation process is conditioned on the current agent's state and capabilities. In practice, this means conditioning the diffusion model on the agent's current policy or Q-function representation.43  
  3. **Personalized Curriculum:** The result is a curriculum that is dynamically generated and personalized for each agent. An agent struggling with a particular concept will be presented with a more gradual curriculum with more intermediate steps, while a more advanced agent will receive a more challenging, condensed curriculum. This automates the process of shaping an agent's learning trajectory, a key factor in improving sample efficiency and solving hard-exploration problems.40 This approach can be seen as a sophisticated form of "reverse curriculum learning," where the system automatically generates start states that are progressively farther from the goal, adapting to the agent's growing competence.46

### **4.4 The Data-Agent Flywheel: A Self-Improving Ecosystem**

The integration of the agent population with the Generative Data Engine creates a powerful, self-reinforcing feedback loop—a **Data-Agent Flywheel**.

1. **Agents Generate Data:** As agents interact with their environment (real or synthetic), they produce high-quality Experience Logs. Their improving world models also serve as the generative basis for the engine to create new synthetic tasks and curricula.  
2. **Data Improves Agents:** This richer, more diverse, and more challenging pool of data (both real and synthetic) is used to train the next generation of agents via the Foundational Curriculum, the Phoenix Cycle, and the Evolutionary Framework.  
3. **Better Agents Generate Better Data:** These improved agents are more capable. They can solve more complex problems, leading to even richer Experience Logs. Their more sophisticated world models allow the Generative Data Engine to create even more complex and nuanced synthetic tasks.

This flywheel dynamic is the ultimate engine of the Chimera-1 project. It creates a co-evolutionary relationship between the learners and their learning environment. The system does not just learn to solve problems; it learns how to *create* the problems that will make it smarter. This is the fundamental mechanism that drives open-ended, accelerating improvement across the entire Chimera-1 species.

## **Conclusion**

This blueprint has detailed a comprehensive, four-pillar framework for the complete learning lifecycle of the Chimera-1 agent. It provides a principled and actionable path from a randomly initialized state to a continuously improving, evolving species of intelligent agents. The core principles underpinning this design are integration and dynamism.

First, the **Foundational Curriculum** establishes a robust baseline of competence by treating the agent's architecture as a composite of specialists. By pre-training each component—the HQ-VAE for conceptual representation, the S4A Engine for sequential modeling, and the HTN/HRL for executive function—on the data types and tasks they are best suited for, we ensure that the integrated agent begins its life on a solid and efficient foundation.

Second, the **Phoenix Cycle** provides a biologically plausible mechanism for stable, lifelong learning. The clear separation of wakeful, goal-directed experience gathering from sleep-like, conservative offline consolidation allows the agent to continuously adapt and improve from its interactions with the world without succumbing to the instabilities of naive online learning.

Third, the **Evolutionary Framework** transcends the limitations of a single agent's lifespan. It implements a dual-track inheritance system that passes down both innate ability (the "nature" of the world model via LATA-RMM breeding) and learned skills (the "nurture" of policies via distillation). This, combined with intra-generational model growth, creates a powerful ratchet for compounding, population-level improvement over time.

Finally, the **Generative Data Engine** transforms the learning environment from a static resource into a dynamic, co-evolving ecosystem. By synthetically generating novel tasks and adaptively creating personalized curricula, the engine ensures that the agent population is always challenged at the frontier of its capabilities, driving true open-ended learning.

Together, these four pillars do not merely describe a training process; they specify a self-sustaining intellectual ecosystem. This blueprint provides the definitive specification for cultivating intelligence in the Chimera-1 architecture, laying the groundwork for a system capable of continuous, accelerating, and autonomous growth.

#### **Works cited**

1. The Official PyTorch Implementation of "NVAE: A Deep Hierarchical Variational Autoencoder" (NeurIPS 2020 spotlight paper) \- GitHub, accessed July 6, 2025, [https://github.com/NVlabs/NVAE](https://github.com/NVlabs/NVAE)  
2. state-spaces/s4: Structured state space sequence models \- GitHub, accessed July 6, 2025, [https://github.com/state-spaces/s4](https://github.com/state-spaces/s4)  
3. HQ-VAE: Hierarchical Discrete Representation Learning with Variational Bayes \- arXiv, accessed July 6, 2025, [https://arxiv.org/html/2401.00365v2](https://arxiv.org/html/2401.00365v2)  
4. HQ-VAE: Hierarchical Discrete Representation Learning with Variational Bayes, accessed July 6, 2025, [https://openreview.net/forum?id=1rowoeUM5E¬eId=WIokW4YYgL](https://openreview.net/forum?id=1rowoeUM5E&noteId=WIokW4YYgL)  
5. HQ-VAE: Hierarchical Discrete Representation Learning with Variational Bayes | OpenReview, accessed July 6, 2025, [https://openreview.net/forum?id=xqAVkqrLjx](https://openreview.net/forum?id=xqAVkqrLjx)  
6. HQ-VAE: Hierarchical Discrete Representation ... \- OpenReview, accessed July 6, 2025, [https://openreview.net/pdf/a5cb9f0542212c08b7e2902272769e63d4699dab.pdf](https://openreview.net/pdf/a5cb9f0542212c08b7e2902272769e63d4699dab.pdf)  
7. VQ-VAE-2 Explained | Papers With Code, accessed July 6, 2025, [https://paperswithcode.com/method/vq-vae-2](https://paperswithcode.com/method/vq-vae-2)  
8. What are variational autoencoders and to what learning tasks are they used?, accessed July 6, 2025, [https://stats.stackexchange.com/questions/321841/what-are-variational-autoencoders-and-to-what-learning-tasks-are-they-used](https://stats.stackexchange.com/questions/321841/what-are-variational-autoencoders-and-to-what-learning-tasks-are-they-used)  
9. Variational Autoencoder Tutorial: VAEs Explained \- Codecademy, accessed July 6, 2025, [https://www.codecademy.com/article/variational-autoencoder-tutorial-vaes-explained](https://www.codecademy.com/article/variational-autoencoder-tutorial-vaes-explained)  
10. HQ-VAE: Hierarchical Discrete Representation Learning with Variational Bayes \- Sony AI, accessed July 6, 2025, [https://ai.sony/publications/HQ-VAE-Hierarchical-Discrete-Representation-Learning-with-Variational-Bayes/](https://ai.sony/publications/HQ-VAE-Hierarchical-Discrete-Representation-Learning-with-Variational-Bayes/)  
11. From S4 to Mamba: A Comprehensive Survey on Structured ... \- arXiv, accessed July 6, 2025, [https://arxiv.org/pdf/2503.18970?](https://arxiv.org/pdf/2503.18970)  
12. What Can We Learn from State Space Models for Machine Learning on Graphs?, accessed July 6, 2025, [https://openreview.net/forum?id=xAM9VaXZnY](https://openreview.net/forum?id=xAM9VaXZnY)  
13. Beyond Transformers: Structured State Space Sequence Models, accessed July 6, 2025, [https://cnichkawde.github.io/statespacesequencemodels.html](https://cnichkawde.github.io/statespacesequencemodels.html)  
14. Introduction to State Space Models as Natural Language Models \- neptune.ai, accessed July 6, 2025, [https://neptune.ai/blog/state-space-models-as-natural-language-models](https://neptune.ai/blog/state-space-models-as-natural-language-models)  
15. State Space Model (SSM) and Structured State Space For Sequence Modeling (S4) \- Ethan Morgan, accessed July 6, 2025, [https://www.ethanmorgan.io/blog/ML/Learning-ML/State-Space-Model-(SSM)-and-Structured-State-Space-For-Sequence-Modeling-(S4)](https://www.ethanmorgan.io/blog/ML/Learning-ML/State-Space-Model-\(SSM\)-and-Structured-State-Space-For-Sequence-Modeling-\(S4\))  
16. Hierarchical Task Network Planning for Facilitating ... \- arXiv, accessed July 6, 2025, [https://arxiv.org/pdf/2306.08359](https://arxiv.org/pdf/2306.08359)  
17. Hierarchical Task Network (HTN) in AI \- Stack Overflow, accessed July 6, 2025, [https://stackoverflow.com/questions/50771965/hierarchical-task-network-htn-in-ai](https://stackoverflow.com/questions/50771965/hierarchical-task-network-htn-in-ai)  
18. Hierarchical Reinforcement Learning (HRL) in AI \- GeeksforGeeks, accessed July 6, 2025, [https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-reinforcement-learning-hrl-in-ai/](https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-reinforcement-learning-hrl-in-ai/)  
19. Hierarchical Reinforcement Learning (HRL): Breaking Down Complex Tasks | by Hey Amit, accessed July 6, 2025, [https://medium.com/@heyamit10/hierarchical-reinforcement-learning-hrl-breaking-down-complex-tasks-d9798e49c782](https://medium.com/@heyamit10/hierarchical-reinforcement-learning-hrl-breaking-down-complex-tasks-d9798e49c782)  
20. VAEs in Reinforcement Learning. In the realm of machine learning ..., accessed July 6, 2025, [https://medium.com/@nicholsonjm92/vaes-in-reinforcement-learning-932fc2df7026](https://medium.com/@nicholsonjm92/vaes-in-reinforcement-learning-932fc2df7026)  
21. Hierarchical Reinforcement Learning: A Survey and Open Research Challenges \- MDPI, accessed July 6, 2025, [https://www.mdpi.com/2504-4990/4/1/9](https://www.mdpi.com/2504-4990/4/1/9)  
22. FINE-TUNING OFFLINE REINFORCEMENT LEARNING WITH ..., accessed July 6, 2025, [https://www.ri.cmu.edu/app/uploads/2021/09/62.pdf](https://www.ri.cmu.edu/app/uploads/2021/09/62.pdf)  
23. Finetuning Offline World Models in the Real World \- Proceedings of Machine Learning Research, accessed July 6, 2025, [https://proceedings.mlr.press/v229/feng23a/feng23a.pdf](https://proceedings.mlr.press/v229/feng23a/feng23a.pdf)  
24. Stacking Your Transformers: A Closer Look at Model Growth ... \- NIPS, accessed July 6, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/143ea4a156ef64f32d4d905206cf32e1-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/143ea4a156ef64f32d4d905206cf32e1-Paper-Conference.pdf)  
25. (PDF) Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training \- ResearchGate, accessed July 6, 2025, [https://www.researchgate.net/publication/380894778\_Stacking\_Your\_Transformers\_A\_Closer\_Look\_at\_Model\_Growth\_for\_Efficient\_LLM\_Pre-Training](https://www.researchgate.net/publication/380894778_Stacking_Your_Transformers_A_Closer_Look_at_Model_Growth_for_Efficient_LLM_Pre-Training)  
26. How do I merge two trained neural network weight matrices into one? \- Stack Overflow, accessed July 6, 2025, [https://stackoverflow.com/questions/49988009/how-do-i-merge-two-trained-neural-network-weight-matrices-into-one](https://stackoverflow.com/questions/49988009/how-do-i-merge-two-trained-neural-network-weight-matrices-into-one)  
27. Random matrix analysis of deep neural network weight matrices | Phys. Rev. E, accessed July 6, 2025, [https://link.aps.org/doi/10.1103/PhysRevE.106.054124](https://link.aps.org/doi/10.1103/PhysRevE.106.054124)  
28. Model Merging: Combining Different Fine-Tuned LLMs \- Marvik \- Blog, accessed July 6, 2025, [https://blog.marvik.ai/2024/06/19/model-merging-combining-different-fine-tuned-llms/](https://blog.marvik.ai/2024/06/19/model-merging-combining-different-fine-tuned-llms/)  
29. What is Model Merging? Techniques & Challenges \- Deepchecks, accessed July 6, 2025, [https://www.deepchecks.com/glossary/model-merging/](https://www.deepchecks.com/glossary/model-merging/)  
30. What is policy distillation in RL? \- Milvus, accessed July 6, 2025, [https://milvus.io/ai-quick-reference/what-is-policy-distillation-in-rl](https://milvus.io/ai-quick-reference/what-is-policy-distillation-in-rl)  
31. Knowledge Transfer for Deep Reinforcement Learning with ..., accessed July 6, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/10733/10592](https://ojs.aaai.org/index.php/AAAI/article/view/10733/10592)  
32. Real-time Policy Distillation in Deep Reinforcement Learning \- ML For Systems, accessed July 6, 2025, [http://mlforsystems.org/assets/papers/neurips2019/real\_time\_sun\_2019.pdf](http://mlforsystems.org/assets/papers/neurips2019/real_time_sun_2019.pdf)  
33. \[1511.06295\] Policy Distillation \- arXiv, accessed July 6, 2025, [https://arxiv.org/abs/1511.06295](https://arxiv.org/abs/1511.06295)  
34. Scale Machine Learning & AI Computing | Ray by Anyscale, accessed July 6, 2025, [https://www.ray.io/](https://www.ray.io/)  
35. Reinforcement Learning for Generative AI: A Survey \- arXiv, accessed July 6, 2025, [https://arxiv.org/html/2308.14328v3](https://arxiv.org/html/2308.14328v3)  
36. Synthetic Data Generation using RL | by Harsh Bhatt \- Medium, accessed July 6, 2025, [https://medium.com/@harshbhatt7585/synthetic-data-generation-using-rl-e89fe9f966c8](https://medium.com/@harshbhatt7585/synthetic-data-generation-using-rl-e89fe9f966c8)  
37. Boosting Deep Reinforcement Learning Agents with Generative Data Augmentation \- MDPI, accessed July 6, 2025, [https://www.mdpi.com/2076-3417/14/1/330](https://www.mdpi.com/2076-3417/14/1/330)  
38. Genesis-Embodied-AI/Genesis: A generative world for general-purpose robotics & embodied AI learning. \- GitHub, accessed July 6, 2025, [https://github.com/Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis)  
39. \[2003.04664\] Automatic Curriculum Learning For Deep RL: A Short Survey \- arXiv, accessed July 6, 2025, [https://arxiv.org/abs/2003.04664](https://arxiv.org/abs/2003.04664)  
40. Automatic Curriculum Learning For Deep RL: A Short Survey \- IJCAI, accessed July 6, 2025, [https://www.ijcai.org/proceedings/2020/0671.pdf](https://www.ijcai.org/proceedings/2020/0671.pdf)  
41. Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use \- ChatPaper, accessed July 6, 2025, [https://chatpaper.com/chatpaper/paper/127048](https://chatpaper.com/chatpaper/paper/127048)  
42. Paper page \- Synthetic Data RL: Task Definition Is All You Need \- Hugging Face, accessed July 6, 2025, [https://huggingface.co/papers/2505.17063](https://huggingface.co/papers/2505.17063)  
43. Diffusion-based Curriculum Reinforcement Learning, accessed July 6, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/b0e89a49af1fb2ebea69bfc39df0be4a-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/b0e89a49af1fb2ebea69bfc39df0be4a-Paper-Conference.pdf)  
44. Diffusion-based Curriculum Reinforcement Learning \- OpenReview, accessed July 6, 2025, [https://openreview.net/forum?id=yRhrVaDOWE\&referrer=%5Bthe%20profile%20of%20Giovanni%20Iacca%5D(%2Fprofile%3Fid%3D\~Giovanni\_Iacca1)](https://openreview.net/forum?id=yRhrVaDOWE&referrer=%5Bthe+profile+of+Giovanni+Iacca%5D\(/profile?id%3D~Giovanni_Iacca1\))  
45. Curriculum for Reinforcement Learning \- Lil'Log, accessed July 6, 2025, [https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/](https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/)  
46. Reverse Curriculum Generation for Reinforcement Learning, accessed July 6, 2025, [https://www.ri.cmu.edu/app/uploads/2017/11/florensa17a.pdf](https://www.ri.cmu.edu/app/uploads/2017/11/florensa17a.pdf)