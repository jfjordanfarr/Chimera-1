

# **Chimera-1: A Blueprint for a Learning Lifecycle**

Version: 1.0 

---

## **Introduction**

This document provides the definitive master blueprint for the learning lifecycle of the Chimera-1 agent. It synthesizes all preceding research and development efforts into a single, cohesive strategy, explicitly tailored to the finalized cognitive architecture defined by ARC-PERCEPTION, ARC-COGNITIVE, and ARC-CONTROL. This blueprint will serve as the canonical guide for the agent's genesis, its continuous adaptation, and its long-term evolution. This report details the multi-stage pre-training curriculum, the lifelong learning "Phoenix Cycle," the evolutionary framework for generational improvement, and the generative data engine that fuels this entire process. The successful execution of this blueprint will mark a significant milestone in creating truly adaptive, generalist artificial agents.

---

## **Section 1: The Foundational Curriculum: Pre-Training Core Competencies**

This section details the multi-stage, offline pre-training regimen designed to forge the agent's core components from raw data. This foundational phase is critical for seeding the agent with the fundamental priors necessary for all subsequent learning. Each component is trained sequentially, with the output of one stage forming the input for the next, creating a structured pipeline from perception to action.

### **1.1 Forging Perception: Object-Centric Codebook (OCCE) Generation**

The first and most fundamental step is to equip the agent with a robust perceptual system capable of decomposing complex visual scenes into their constituent objects. The ARC-PERCEPTION.OCCE (Object-Centric Codebook Encoder) module is designed to learn a discrete, semantic vocabulary for representing objects and their properties, independent of viewpoint or transient environmental conditions.

#### **1.1.1 Data Corpus: The Objectron Dataset and Multi-View Video Streams**

The quality of the learned perceptual representations is directly dependent on the quality and diversity of the training data. To this end, the primary data source for training the OCCE will be the **Objectron dataset**.1 This large-scale dataset is uniquely suited for our purposes, containing approximately 15,000 object-centric video clips that capture common objects from a multitude of angles.1 The inclusion of AR session metadata, such as camera poses, sparse point-clouds, and 3D bounding boxes, provides rich, multi-view geometric understanding that is essential for learning view-invariant representations.2 The dataset's scale, with over 4 million annotated images, and its geo-diversity, collected from 10 countries across five continents, ensure that the learned model is not biased towards a specific controlled environment and can generalize to objects "in the wild".1

To further enhance generalization and robustness, the Objectron data will be supplemented with large, unconstrained video datasets such as **YouTube-VIS**.3 While lacking the detailed 3D annotations of Objectron, these datasets expose the model to a wider variety of real-world motion patterns, occlusions, and background clutter. This combination ensures the

OCCE learns to segment objects both from clean, multi-view data and from complex, dynamic scenes.

#### **1.1.2 Training Architecture: Vector-Quantized Vision Foundation Models for OCL (VVO)**

The OCCE will be implemented using the **VVO (Vector-Quantized Vision Foundation Models for Object-Centric Learning)** architecture, a state-of-the-art framework that unifies several key advances in object-centric learning.5 The VVO architecture consists of three main components: an encoder, an aggregator, and a decoder.5

* **Encoder:** A pre-trained Vision Foundation Model (VFM), such as DINO, will serve as the encoder backbone. VFMs are highly effective at extracting feature maps with strong "objectness" properties, meaning they naturally separate objects from the background even before explicit training.5 This provides a powerful starting point for the aggregation process.  
* **Aggregator:** A **Slot Attention** module will act as the aggregator.5 This module takes the dense feature map from the VFM encoder and, using a set of learnable query vectors (slots), sparsifies it into a set of object-level feature vectors. Each slot comes to represent a single object or background element in the scene.  
* **Quantizer and Decoder:** The central innovation of the VVO framework lies in its reconstruction target. Instead of attempting to reconstruct the raw input pixels, which is notoriously difficult for scenes with complex textures 5, the  
  OCCE learns to reconstruct a *quantized representation* of the VFM's feature map. The VFM features are passed through a vector-quantization (VQ) layer, which maps them to the closest entries in a learned, discrete codebook. The decoder's objective is then to reconstruct this quantized feature map from the aggregated object slots.5

This design choice has profound implications. By training the system to reconstruct a discretized, semantic representation rather than raw pixels, we force the OCCE to learn a vocabulary that captures abstract properties like texture, shape components, and material qualities. The VQ codebook effectively becomes a "genomic" library of fundamental visual concepts, analogous to the semantic representations learned by models like SVQ.8 This shared quantization process suppresses redundant pixel-level information and provides a consistent, semantically rich training signal across all training samples.5 The output of the

OCCE is thus not an image, but a set of discrete token IDs from this genomic codebook, providing a compact, compositional, and meaningful description of the visual world for the downstream cognitive modules.

#### **1.1.3 Objective Function: VQ-VFM Reconstruction and Semantic Contrastive Loss**

The training of the OCCE is guided by a composite loss function designed to foster robust and invariant object representations.

1. **VQ-Feature Reconstruction Loss:** The primary objective is the reconstruction loss of the VVO architecture, which minimizes the divergence (e.g., using Mean Squared Error or Cross-Entropy) between the decoder's output and the quantized VFM feature target.7 This drives the core learning of the object-centric representation.  
2. **Temporal Feature Similarity Loss:** To improve object discovery and tracking in video sequences, a temporal feature similarity loss, as proposed in the VideoSAUR model, will be incorporated.3 This loss encourages features from corresponding patches in consecutive frames to be similar, introducing a strong motion bias that is highly effective for segmenting moving objects from static or moving backgrounds.  
3. **Object-Level Contrastive Loss:** To ensure the learned object representations are invariant to changes in viewpoint, scale, and position, we will implement an object-level contrastive learning objective inspired by SoCo.11 For each object slot identified in a video, we will treat its representations from different frames (and thus different camera angles, thanks to the Objectron dataset) as a "positive pair." The model will be trained to maximize the similarity of these positive pairs in the embedding space while simultaneously pushing them apart from "negative pairs" (i.e., representations of other objects).11 This explicitly teaches the  
   OCCE to produce a consistent set of genomic codes for an object regardless of how it is viewed.

### **1.2 Building a World Model: Hierarchical Autoregressive State-Space Model (HASSM)**

With a robust perceptual system in place, the next stage is to build the agent's internal world model. The ARC-COGNITIVE.HASSM is designed to learn the temporal dynamics of the environment—how the world evolves and responds to the agent's actions. This model is the foundation of the agent's ability to plan, reason, and "imagine" future outcomes.

#### **1.2.1 Data Corpus: Sequences of OCCE Genomic Codes**

The HASSM is not trained on raw visual data. Instead, its training corpus consists of sequences of discrete token IDs generated by the pre-trained and frozen OCCE module. During an initial data-gathering phase (using a simple random or exploratory policy), the agent collects trajectories of interactions. Each observation in these trajectories is passed through the OCCE, converting a high-dimensional video frame into a compact set of K discrete tokens, {z\_1,..., z\_K}. The resulting dataset for the HASSM is a collection of sequences of the form (z\_0, a\_0, r\_0, d\_0), (z\_1, a\_1, r\_1, d\_1),..., where z represents the tokenized state, a the action, r the reward, and d the episode termination flag.

#### **1.2.2 Training Architecture: A Δ-IRIS-style Autoregressive Transformer**

The HASSM architecture is an autoregressive Transformer, a design that has proven highly effective for sequence modeling and has been successfully applied to building world models.14 Specifically, we adopt the architectural principles of

**IRIS** and its more efficient successor, **Δ-IRIS**.17 This approach casts the problem of learning world dynamics as a next-token prediction task, similar to a large language model.18

A critical architectural choice is the adoption of the **Δ-IRIS** innovation: modeling the *delta* between timesteps.17 Instead of predicting the entire set of

K tokens for the next state z\_{t+1} from scratch, the Transformer is conditioned on the previous state z\_t and action a\_t and predicts only the *change* in the scene's token representation. This is vastly more efficient, as most objects in a scene are often static from one frame to the next. This delta-based prediction drastically reduces the number of tokens the model must generate per timestep, overcoming a key computational bottleneck of the original IRIS model and making long-horizon imagination computationally feasible.17 The architecture is a standard GPT-style Transformer decoder that processes a sequence of interleaved state-delta tokens and action tokens.18

This combination of a semantic OCCE vocabulary and a delta-prediction HASSM creates an exceptionally efficient simulator of *semantic change*. The model does not waste computational capacity predicting the pixel values of a static background. Instead, it focuses entirely on modeling the object-centric events that matter: how actions cause objects to appear (adding new tokens), disappear (removing tokens), or change their properties (swapping tokens for others from the codebook). This is a far more abstract and efficient form of world modeling than traditional pixel-based or latent-space video prediction.

#### **1.2.3 Objective Function: Autoregressive Prediction of Latent States, Rewards, and Terminations**

The HASSM is trained in a purely supervised manner on the tokenized trajectories.18 Given a history of states (represented by

OCCE tokens) and actions, the model is optimized to predict three key elements of the next timestep:

1. **Next State Tokens:** The model predicts the probability distribution over the next set of state tokens, p(z\_{t+1} | z\_{\<=t}, a\_{\<=t}). This is trained using a standard cross-entropy loss against the ground-truth tokens from the dataset.  
2. **Reward:** The model predicts the reward received at the current step, p(r\_t | z\_{\<=t}, a\_{\<=t}). The loss function here can be a mean-squared error for continuous rewards or a cross-entropy loss for discretized reward values.  
3. **Termination:** The model predicts a binary flag indicating whether the episode has terminated, p(d\_t | z\_{\<=t}, a\_{\<=t}), trained with a cross-entropy loss.

This multi-part objective directly trains the HASSM to predict the full consequences of taking an action within the semantic, latent space of the OCCE. This forms the core of the agent's ability to "imagine" and simulate future scenarios, which is essential for model-based reinforcement learning.18

### **1.3 Seeding Action: Hierarchical Policy (ActionSystem) Initialization**

The final stage of the foundational curriculum is to seed the ARC-CONTROL.ActionSystem with a competent initial policy. This provides the agent with a baseline set of skills and a nascent understanding of task structure, preventing it from starting its life with purely random behavior. This is achieved through imitation learning from expert demonstrations.

#### **1.3.1 Data Corpus: Curated Expert Trajectory Datasets**

The ActionSystem is pre-trained using a corpus of **expert demonstration trajectories**.24 These trajectories are sequences of

(observation, action) pairs generated by a proficient expert, which could be a human teleoperating the system or a pre-trained, specialized RL agent.25 It is important that the dataset contains demonstrations for a diverse range of both primitive skills (e.g., "pick up object") and more complex, composite tasks (e.g., "make coffee").

All demonstration data will be standardized and stored as Trajectory objects, following the format defined by the imitation library, which includes observations, actions, rewards, and optional info dictionaries.27 To ensure accessibility, version control, and reproducibility, this entire corpus will be hosted on the HuggingFace Hub.27

#### **1.3.2 Training Architecture: Generative Hierarchical Behavioral Cloning**

A simple behavioral cloning approach that maps states directly to actions often fails on complex, multi-stage tasks.28 Therefore, we will employ a more sophisticated

**generative hierarchical behavioral cloning** framework. This approach is designed to learn both the low-level motor policies (the Hierarchical Reinforcement Learning, or HRL, component) and the high-level task structure (the Hierarchical Task Network, or HTN, component) directly from unsegmented demonstration trajectories.29

The architecture functions like a conditional Variational Autoencoder (VAE). It introduces a discrete latent variable, c, which represents the "macro-intent" or sub-task that the expert is currently pursuing. The model learns three distinct components simultaneously:

1. **Inference Network q(c | s, a):** Given a state-action pair from the expert's trajectory, this network infers the most probable latent sub-task c that the expert was executing.  
2. **High-Level Policy π\_{high}(c | s):** This policy, representing the HTN, learns to select the appropriate macro-intent or sub-task c based on the current state s.  
3. **Low-Level Policies π\_{low}(a | s, c):** This is a set of policies, representing the HRL layer, that generate low-level actions a conditioned on both the current state s and the active macro-intent c.

This framework discovers the underlying task hierarchy from the raw demonstration data by optimizing for a model in which the latent sub-task variables are maximally informative about the observed trajectories.30

#### **1.3.3 Objective Function: Supervised Policy and Latent Intent Learning**

The training objective is a form of supervised learning that aims to maximize the likelihood of the expert's actions under our hierarchical policy model.28 This is expressed as maximizing the log-likelihood

log p(a\_{expert} | s\_{expert}), which is computed by marginalizing over all possible latent intents: log Σ\_c p(a\_{expert} | s\_{expert}, c) p(c | s\_{expert}).

Because the latent intent c is unobserved, this objective is optimized variationally by maximizing an Evidence Lower Bound (ELBO). The ELBO consists of a reconstruction term, which encourages the low-level policies to match the expert's actions for a given inferred intent, and a regularization term (typically a KL-divergence), which ensures the distribution over latent intents remains well-structured. This process effectively performs behavioral cloning at both the high (task selection) and low (action execution) levels of the agent's control hierarchy.

### **Table 1: Foundational Curriculum Summary**

| Architectural Component | Primary Data Corpus | Training Architecture | Core Objective Function |
| :---- | :---- | :---- | :---- |
| **ARC-PERCEPTION.OCCE** | Objectron, YouTube-VIS 1 | VQ-VFM-OCL (VVO) with Slot Attention 5 | VQ-Feature Reconstruction \+ Temporal Similarity Loss \+ Object-Level Contrastive Loss 3 |
| **ARC-COGNITIVE.HASSM** | OCCE Token Sequences from exploration | Δ-IRIS Autoregressive Transformer 17 | Autoregressive Prediction (Cross-Entropy) of next state tokens, reward, and termination 18 |
| **ARC-CONTROL.ActionSystem** | Expert Demonstration Trajectories 24 | Generative Hierarchical Behavioral Cloning 29 | Variational Supervised Learning of latent intents and conditional action policies 29 |

---

## **Section 2: The Phoenix Cycle: A Framework for Lifelong Adaptation**

The foundational curriculum provides the agent with its initial "genetic code" and baseline skills. However, a truly intelligent agent must be able to learn and adapt continuously from its experiences throughout its operational lifetime. The "Phoenix Cycle" is the definitive framework for this lifelong learning process. It is a robust, iterative loop divided into two distinct phases: a "Wake" phase for active data gathering in the environment, and a "Sleep" phase for safe, offline model and policy refinement.

### **2.1 The Wake Phase: Active Exploration and Experience Acquisition**

During the Wake phase, the agent is active in its environment, pursuing tasks and gathering new data. The primary goal of this phase is not just to perform, but to learn—specifically, to collect the most informative data possible to fuel the subsequent refinement phase.

#### **2.1.1 Data Acquisition Protocol and Replay Buffer Management**

The agent interacts with the real environment using its current policy as defined by the ActionSystem. Every interaction, captured as a (observation, action, reward, next\_observation, done) tuple, is stored in a large, persistent, and first-in-first-out replay buffer. This buffer serves as the agent's complete life experience.

As new observations are collected, they are immediately processed by the frozen OCCE module to generate their corresponding tokenized representations (z). These token sets are stored alongside the raw observations in the replay buffer. This pre-processing step ensures that when data is sampled for training the HASSM world model, the required token sequences are readily available without the need for repeated, costly inference through the perceptual system.

#### **2.1.2 Online Active Exploration Strategy for Informative Data Gathering**

Simple exploration strategies, such as ε-greedy, are inefficient for complex, high-dimensional environments. To ensure the agent gathers maximally useful data, it will employ a model-based **active exploration** strategy.31 The core idea is to direct exploration towards areas of the state-action space where the agent's knowledge is weakest.

This strategy will directly leverage the HASSM world model. We will maintain an ensemble of HASSM models, each trained on a different bootstrap sample of the replay buffer. During the Wake phase, the agent can use this ensemble to quantify its uncertainty about the world's dynamics. Exploration will be intrinsically rewarded and directed towards state-action pairs where the predictions of the HASSM ensemble exhibit high variance or disagreement.33 This uncertainty-based approach ensures that the agent actively seeks out experiences that will most effectively reduce its world model's error, leading to a highly efficient learning cycle. This can be conceptualized as an online learning problem, akin to the LEO algorithm, where the agent learns to select the exploration strategy that yields the greatest improvement in its models or downstream performance.32

### **2.2 The Sleep Phase: Conservative Model-Based Refinement**

The Sleep phase is a period of intensive, offline computation where the agent uses the data accumulated in its replay buffer to safely update and improve its internal models and policies. The key challenge in this phase is to learn from new data without destabilizing existing knowledge, a problem known as distributional shift, or catastrophically forgetting past skills.34 Our approach is designed specifically to be conservative and robust.

#### **2.2.1 Architectural Framework: A Hybrid of DreamerV3 and COMBO Principles**

The learning algorithm for the Sleep phase is a novel hybrid that combines the strengths of two leading model-based RL methods: **DreamerV3** and **COMBO**. DreamerV3 provides a set of exceptionally robust techniques for training the world model, while COMBO offers a provably conservative method for updating the policy and value function in an offline setting. This combination creates a powerful and stable learning update.

#### **2.2.2 World Model (HASSM) Update: Symlog Prediction and KL Balancing for Robustness**

The HASSM world model is updated using batches of tokenized trajectories sampled from the replay buffer. To ensure this update process is stable across a wide variety of environments and reward scales, we will adopt the key robustness techniques from the DreamerV3 algorithm.36

* **Symlog Predictions:** All predictive heads within the HASSM (for next-state tokens, rewards, and terminations) will be trained to predict the symlog transformed value of their targets, rather than the raw values.36 The  
  symlog function, sign(x) \* log(|x| \+ 1), compresses large values while leaving small values near zero relatively unchanged. This automatically handles the varying scales of inputs and rewards found in different domains, making the learning process stable without requiring per-task hyperparameter tuning.  
* **KL Balancing:** The loss function for the world model includes KL-divergence terms that ensure the latent state representations are both predictable by the dynamics model and informative about the observations. We will use **KL balancing with free bits** to regularize these terms.36 This technique prevents the KL loss from collapsing the representations to a trivial, uninformative state by providing a minimum "budget" of information (e.g., 1 nat) that the latent state must contain. This focuses the model's capacity on prediction error once the representation is sufficiently rich.

#### **2.2.3 Policy (ActionSystem) Update: Conservative Offline Policy Optimization (COMBO)**

The ActionSystem, which comprises the agent's policy and value function, is updated using the **COMBO (Conservative Offline Model-Based Policy Optimization)** algorithm.40 COMBO is a model-based offline RL algorithm designed to address the core problem of distributional shift. Crucially, it achieves conservatism

**without relying on explicit uncertainty estimation**, which can be difficult and unreliable for complex neural network models.40

The COMBO update works as follows: The critic (value function) is trained on a **mixed batch of data**, containing both real transitions sampled from the replay buffer and synthetic transitions generated by performing rollouts of the current policy within the newly updated HASSM world model.42 The key conservative Q-learning objective includes a regularization term that explicitly penalizes (pushes down) the Q-values of out-of-support state-action pairs from the model-generated data, while simultaneously maximizing (pushing up) the Q-values of state-action pairs from the real offline dataset.42 This procedure guarantees that the learned Q-function is a conservative lower bound on the true policy value, effectively preventing the policy from exploiting errors in the world model and choosing dangerously overestimated actions.35

#### **2.2.4 Mitigating Catastrophic Forgetting via Conservative Updates and Experience Replay**

Catastrophic forgetting, the tendency for a model to abruptly lose previously acquired knowledge when learning new information, is a primary risk in any lifelong learning system.34 The Phoenix Cycle incorporates two explicit mechanisms to combat this threat.

First, the **conservative nature of the COMBO policy update** inherently resists forgetting. By anchoring the value function to the entire history of real data contained in the replay buffer (D), the algorithm is prevented from making drastic policy changes that are inconsistent with past successful behaviors.42

Second, the update process for the HASSM world model will utilize **experience replay**. Training batches will be composed of a mixture of recent data from the latest Wake phase and older data sampled from throughout the replay buffer's history. This replay-based strategy is a standard and highly effective technique in continual learning for ensuring that the model is repeatedly exposed to past data distributions, thereby preventing it from overwriting old knowledge with new.45

The synergy between these components creates a virtuous cycle. The robust world model training from DreamerV3 provides a high-fidelity simulation environment. The COMBO algorithm leverages this simulation to learn a safe, conservative policy. This improved policy then gathers higher-quality, more stable data during the next Wake phase, which in turn feeds back into an even better world model update. This symbiotic relationship between robust world modeling and conservative policy optimization is the engine of stable, continuous improvement.

### **Table 2: Phoenix Cycle Protocol**

| Phase | Step | Action | Data Flow | Learning Objective |
| :---- | :---- | :---- | :---- | :---- |
| **Wake** | 1 | Interact & Explore | Agent \-\> Environment | Maximize reward \+ information gain (model uncertainty reduction) |
|  | 2 | Acquire Experience | Environment \-\> Replay Buffer | Store (s, a, r, s', d) tuples |
|  | 3 | Tokenize | OCCE \-\> Replay Buffer | Store z tokens alongside raw observations |
| **Sleep** | 4 | Sample Batch | Replay Buffer \-\> Training Module | Sample mixed batch of new and old data |
|  | 5 | Update World Model | Batch \-\> HASSM | Minimize DreamerV3-style prediction loss (Symlog, KL-Balancing) 36 |
|  | 6 | Generate Synthetic Data | HASSM \-\> Model Buffer | Rollout current policy in HASSM to get synthetic trajectories |
|  | 7 | Update Policy/Value | Replay Buffer \+ Model Buffer \-\> ActionSystem | Minimize COMBO conservative Q-learning loss 42 |

---

## **Section 3: The Evolutionary Framework: Long-Term Species Development**

While the Phoenix Cycle enables an individual agent to adapt within its lifetime, the Evolutionary Framework provides the mechanisms for the long-term, multi-generational development of the Chimera-1 "species." These processes operate on a much longer timescale, allowing for fundamental increases in agent capacity and the creation of novel agent variants through the combination of skills from a population of specialized individuals.

### **3.1 Generational Growth: Scaling Agent Capacity with G\_stack**

As an agent masters its environment, its progress may eventually be limited by the raw capacity of its neural network architecture. The generational growth mechanism allows us to overcome this limitation by systematically increasing the size and power of our agents over time.

#### **3.1.1 Application to HASSM and ActionSystem Transformer Architectures**

To scale the capacity of our agents, we will employ the **G\_stack (depth-wise stacking) operator**, a model growth technique that is particularly well-suited for Transformer-based architectures.47 This method will be applied to the Transformer modules that form the core of both the

HASSM world model and the policy networks within the ActionSystem.

The G\_stack process is conceptually simple yet powerful. A smaller, fully-trained "parent" agent is selected. A new, larger "child" agent is then created by duplicating the layers of the parent's Transformer modules and stacking them depth-wise.47 For example, a parent agent with a 12-layer

HASSM can be "grown" into a child with a 24-layer HASSM by stacking the parent's entire 12-layer block twice. This allows the child model to inherit the learned representations of the parent at multiple levels of abstraction, dramatically accelerating its training compared to starting from a random initialization.47

#### **3.1.2 A Principled Strategy for Optimal Growth Timing and Factor**

The application of G\_stack is not an ad-hoc process; it is guided by a principled, data-driven strategy to maximize its effectiveness.47

* **Growth Factor (g):** The growth factor g determines how many times the parent's layers are stacked. Empirical studies show that the optimal performance is typically achieved with a growth factor between 2 and 4\. To standardize our process and ensure strong results, we will adopt a fixed **growth factor of g=4** for all generational growth events.47 This means a parent model with  
  L layers will be grown into a child model with 4L layers.  
* **Growth Timing (d):** The success of G\_stack is highly dependent on the training maturity of the parent model. Growing too early or too late can be suboptimal. We will use the empirically derived formula from the foundational research to determine the optimal growth timing d (measured in the number of training tokens the parent has processed): log10(d) \= a log10(N) \+ b/log10(C) \+ c, where N is the parameter count of the target child model and C is our available computational budget for training it.47 This equation provides a formalized method for deciding precisely when to trigger a generational growth event to achieve the maximum possible training acceleration.47

### **3.2 Generational Breeding: Creating Novel Agents with Reinforced Model Merging (RMM)**

In addition to vertical scaling through growth, our evolutionary framework supports horizontal combination of skills through "breeding." This process allows us to create novel, versatile agents by merging the parameters of multiple parent agents that have developed specialized expertise in different domains.

#### **3.2.1 The RMM Environment for Chimera-1 Architectures**

To breed new agents, we will use the **Reinforced Model Merging (RMM)** framework.49 RMM reframes the complex task of model merging as a reinforcement learning problem, thereby avoiding naive and often ineffective techniques like simple parameter averaging.53

We will define a specific merging environment for the Chimera-1 architecture. In this environment, an RL "merging agent" learns a policy for constructing a new child agent's HASSM and ActionSystem on a layer-by-layer basis from a pool of two or more parent agents.50 The state is a "merging map" that tracks the construction of the child model, and the action space provides a rich set of operations for each layer:

1. **Model Actions:** Copy layer k directly from Parent A (or Parent B, etc.).  
2. **Merging Actions:** Combine the corresponding layers from multiple parents using a sophisticated merging operator like TIES-merging, which intelligently resolves parameter conflicts.53  
3. **Layer Actions:** Skip the current layer (creating a residual connection) or return to a previous layer, allowing the RL agent to discover complex, non-linear, and even hierarchical merging strategies.52

#### **3.2.2 The Merging Agent and Dynamic Average Reward (DAR) for Efficient Search**

A dedicated RL agent, trained with a policy gradient algorithm like PPO, will learn to navigate this merging environment.51 Its goal is to discover a sequence of actions—a merging policy—that results in a child model with the highest performance on a diverse, held-out set of evaluation tasks.

A key challenge in this process is the computational cost of evaluating each newly constructed child model to provide a reward signal. To make this search tractable, we will implement the **Dynamic Average Reward (DAR)** mechanism.50 DAR dramatically accelerates the reward feedback loop by using only small, random subsets of the evaluation data to estimate the child's performance in each search episode. This provides a reward signal that is efficient enough for the RL agent to learn effectively, with reported speedups of up to 100x compared to full evaluation.50

### **3.3 Knowledge Inheritance: Hierarchical Policy Distillation**

Whether a new generation agent is created via G\_stack or RMM, its neural architecture will differ from its parent(s), making a direct transfer of weights impossible. To ensure that the valuable knowledge learned by the parent generation is not lost, we will use **policy distillation** to transfer it to the new child agent.55

#### **3.3.1 Distilling HTN and HRL Policies from a Retiring Teacher to a New Student**

In this process, the fully-trained parent agent acts as the "teacher," and the newly created, randomly initialized child agent is the "student." The student is trained not through environmental interaction, but in a supervised manner to mimic the teacher's behavior. This distillation is applied hierarchically to match the structure of our ActionSystem:

* **HTN Distillation:** The student's high-level policy (π\_{high}) is trained to match the teacher's output probability distribution over the latent "macro-intents" or sub-tasks for a given set of states.  
* **HRL Distillation:** The student's low-level policies (π\_{low}) are trained to match the teacher's output action probability distributions, conditioned on both the state and the teacher's inferred macro-intent.

This process effectively "transplants" the decision-making logic of the teacher into the new, more capable architecture of the student, providing it with a massive head start in its own learning journey.56

#### **3.3.2 Real-Time Distillation for Continuous Knowledge Transfer**

To further accelerate development, we can adopt a **real-time distillation** approach.57 Instead of waiting for a teacher agent to be fully trained and "retired," a student agent can be trained in parallel. The teacher's latest policy is continuously distilled into the student as the teacher itself continues to learn via its own Phoenix Cycle. This reduces the total training time and allows the student to benefit immediately from the teacher's most up-to-date knowledge.

These three mechanisms—growth, breeding, and inheritance—are not isolated techniques. They form a single, integrated evolutionary pipeline. A new generation begins with G\_stack or RMM to create a new, more capable agent architecture (the "hardware"). This new architecture is then seeded with the accumulated knowledge of its predecessors via policy distillation (the "software"). Finally, this newly initialized agent is deployed to begin its own Phoenix Cycle of lifelong learning and adaptation. This complete, end-to-end process enables long-term, compounding improvement across agent generations.

### **Table 3: Evolutionary Operator Specifications**

| Operator | Target Module(s) | Input | Output | Key Parameters / Mechanism |
| :---- | :---- | :---- | :---- | :---- |
| **G\_stack** | HASSM, ActionSystem | Trained Parent Agent (Size N) | Larger Child Agent (Size g\*N) | Growth Factor g=4, Growth Timing d from formula 47 |
| **RMM** | HASSM, ActionSystem | 2+ Specialized Parent Agents | Novel Hybrid Child Agent | RL agent searches layer-wise merge actions; DAR for reward 50 |
| **Policy Distillation** | ActionSystem | Teacher Policy π\_T | Student Policy π\_S | Supervised learning to match teacher's intent & action distributions 56 |

---

## **Section 4: The Generative Data Engine: Fueling the Lifecycle**

The entire learning lifecycle, from foundational pre-training to long-term evolution, is critically dependent on a robust and intelligent data infrastructure. The Generative Data Engine is the comprehensive system designed to support all stages of the agent's development. It is not merely a passive repository of data but an active participant in the learning process, responsible for curating real data and generating synthetic new challenges.

### **4.1 Corpus Curation and Management**

The foundation of the engine is its ability to manage and curate the vast amounts of data required for training.

#### **4.1.1 A SELECT-based Framework for Initial Data Sourcing and Quality Control**

Our entire data management strategy is built upon the principles and methodologies of the **SELECT benchmark**.58 SELECT establishes that data curation is not a passive act of collection but an active process of selecting and organizing samples to maximize learning efficiency.58

We will implement a SELECT-inspired framework to evaluate all incoming data sources, from the initial Objectron and expert trajectory corpora to the data continuously collected by agents in the field. Each dataset will be benchmarked using a suite of metrics, including its impact on in-distribution accuracy, out-of-distribution (OOD) robustness, and transfer learning performance on downstream tasks.60 This rigorous quality control process ensures that we only use high-utility data for training and pre-training, guided by the key finding that the quality of data curation is often more important than the sheer quantity of data.60

#### **4.1.2 Metadata Schema and Trajectory Storage on the HuggingFace Hub**

To ensure consistency and accessibility, all datasets will be standardized. Trajectories will be stored using the data structures defined in the imitation library, which provides a standard format for observations, actions, rewards, and auxiliary information dictionaries.27

All curated and benchmarked datasets will be hosted on the **HuggingFace Hub**.27 This provides a centralized, version-controlled, and easily accessible repository for every data asset used in the Chimera-1 project. Each dataset on the Hub will be accompanied by extensive metadata, documenting its source, the curation strategy used, its quality scores from our SELECT evaluation framework, and its intended use within the learning lifecycle.

### **4.2 Synthetic Data and Task Generation**

The most advanced function of the Generative Data Engine is its ability to create novel data and tasks, turning it from a simple data library into a source of endless challenges for the agent.

#### **4.2.1 Leveraging the HASSM World Model for Novel Scenario Generation**

The trained HASSM is not just a predictive model; it is a powerful generative model of the world's dynamics in the OCCE's semantic latent space.14 We will leverage this capability to augment our training data by generating vast quantities of synthetic trajectories. By providing an initial state, sampling a sequence of actions from the agent's policy, and unrolling the

HASSM's dynamics, we can create plausible scenarios and interactions that the agent has never experienced in the real environment. These latent trajectories of OCCE tokens can be used directly to train the policy, or they can be decoded back into video frames to supplement the training data for all modules, significantly increasing data diversity and volume.

#### **4.2.2 Procedural Generation of Hierarchical Task Network (HTN) Goals in Latent Space**

The ultimate function of the Generative Data Engine is to create entirely new tasks for the agent to solve, fostering a form of open-ended, self-directed learning. Instead of requiring human engineers to hand-craft new goals for the HTN planner, we will use the HASSM to procedurally generate them.

The process for automated curriculum generation is as follows:

1. **Sample a Latent Goal:** We sample a latent state, z\_{goal}, from the state space learned by the HASSM. This goal state is specifically chosen from a region of the latent space that the world model evaluates as having low probability (i.e., it is novel) but is still considered reachable from the agent's current distribution of states.  
2. **Define a New Task:** This novel latent state z\_{goal} is designated as the terminal state for a new, synthetically generated task.  
3. **Challenge the Planner:** The agent's ActionSystem is then tasked with finding a plan—a sequence of HTN sub-tasks and HRL actions—that can successfully navigate from a typical starting state to this new, imagined goal state z\_{goal} within the HASSM's simulated environment.

This creates a powerful, automated curriculum. The agent's own understanding of the world, embodied in its HASSM, is used to continuously generate new and challenging problems for its planning and control systems to solve. This closed loop, where the agent's knowledge fuels the creation of challenges that expand that very knowledge, is the key to driving the agent towards ever-greater competence and generalization without constant human intervention. The data engine is thus not a static resource but an active, intelligent partner in the agent's own development.

---

## **Conclusion**

This blueprint, "Chimera-1: A Blueprint for a Learning Lifecycle," represents the culmination of our research into creating a truly adaptive artificial agent. By integrating state-of-the-art techniques for perception, world modeling, and control within a unified framework, we have laid out a comprehensive and actionable plan. The Foundational Curriculum provides the essential starting priors, ensuring the agent begins its life with a robust understanding of objects, dynamics, and tasks. The Phoenix Cycle defines a rigorous process for robust, continuous adaptation, allowing the agent to learn from experience without the risk of catastrophic forgetting. The Evolutionary Framework enables long-term, compounding growth in capability and intelligence, ensuring the Chimera-1 "species" can scale and diversify over time. Finally, the Generative Data Engine transforms data from a static resource into an active, intelligent partner in the agent's development, creating a virtuous cycle of self-driven, open-ended learning. The successful execution of this lifecycle will not only produce the Chimera-1 agent but will also establish a new paradigm for building the next generation of generalist AI.

#### **Works cited**

1. google-research-datasets/Objectron: Objectron is a dataset of short, object-centric video clips. In addition, the videos also contain AR session metadata including camera poses, sparse point-clouds and planes. In each video, the camera moves around and above the object and captures it from different views. \- GitHub, accessed July 7, 2025, [https://github.com/google-research-datasets/Objectron](https://github.com/google-research-datasets/Objectron)  
2. Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild With Pose Annotations \- CVF Open Access, accessed July 7, 2025, [https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmadyan\_Objectron\_A\_Large\_Scale\_Dataset\_of\_Object-Centric\_Videos\_in\_the\_CVPR\_2021\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmadyan_Objectron_A_Large_Scale_Dataset_of_Object-Centric_Videos_in_the_CVPR_2021_paper.pdf)  
3. Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities, accessed July 7, 2025, [https://openreview.net/forum?id=t1jLRFvBqm](https://openreview.net/forum?id=t1jLRFvBqm)  
4. \[2306.04829\] Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities \- arXiv, accessed July 7, 2025, [https://arxiv.org/abs/2306.04829](https://arxiv.org/abs/2306.04829)  
5. Vector-Quantized Vision Foundation Models for Object-Centric Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2502.20263v1](https://arxiv.org/html/2502.20263v1)  
6. Vector-Quantized Vision Foundation Models for Object-Centric Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2502.20263v3](https://arxiv.org/html/2502.20263v3)  
7. \[Literature Review\] Vector-Quantized Vision Foundation Models for Object-Centric Learning, accessed July 7, 2025, [https://www.themoonlight.io/en/review/vector-quantized-vision-foundation-models-for-object-centric-learning](https://www.themoonlight.io/en/review/vector-quantized-vision-foundation-models-for-object-centric-learning)  
8. Object-Centric Semantic Vector Quantization \- OpenReview, accessed July 7, 2025, [https://openreview.net/forum?id=HAymeESPKo](https://openreview.net/forum?id=HAymeESPKo)  
9. Structured World Modeling via Semantic Vector Quantization \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2402.01203v1](https://arxiv.org/html/2402.01203v1)  
10. Object-Centric Semantic Vector Quantization \- OpenReview, accessed July 7, 2025, [https://openreview.net/forum?id=Jl8g3s6dHk\&referrer=%5Bthe%20profile%20of%20Yi-Fu%20Wu%5D(%2Fprofile%3Fid%3D\~Yi-Fu\_Wu1)](https://openreview.net/forum?id=Jl8g3s6dHk&referrer=%5Bthe+profile+of+Yi-Fu+Wu%5D\(/profile?id%3D~Yi-Fu_Wu1\))  
11. Aligning Pretraining for Detection via Object-Level Contrastive Learning \- OpenReview, accessed July 7, 2025, [https://openreview.net/pdf?id=8PA2nX9v\_r2](https://openreview.net/pdf?id=8PA2nX9v_r2)  
12. What is Contrastive Learning? A 2025 Guide \- viso.ai, accessed July 7, 2025, [https://viso.ai/deep-learning/contrastive-learning/](https://viso.ai/deep-learning/contrastive-learning/)  
13. Contrastive Learning. Learning Representations | by Emmanuel Olateju | Rectlabs Inc, accessed July 7, 2025, [https://medium.com/rectlabs/contrastive-learning-c2c557865402](https://medium.com/rectlabs/contrastive-learning-c2c557865402)  
14. Daily Papers \- Hugging Face, accessed July 7, 2025, [https://huggingface.co/papers?q=autoregressive%20action%20world%20model](https://huggingface.co/papers?q=autoregressive+action+world+model)  
15. Types of Autoregressive Language Model & Applications \- Code B, accessed July 7, 2025, [https://code-b.dev/blog/autoregressive-language-model](https://code-b.dev/blog/autoregressive-language-model)  
16. Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning \- arXiv, accessed July 7, 2025, [http://arxiv.org/pdf/2506.18537](http://arxiv.org/pdf/2506.18537)  
17. Towards Efficient World Models \- OpenReview, accessed July 7, 2025, [https://openreview.net/pdf?id=o8IDoZggqO](https://openreview.net/pdf?id=o8IDoZggqO)  
18. Transformers are Sample-Efficient World Models | OpenReview, accessed July 7, 2025, [https://openreview.net/forum?id=vhFu1Acb0xb](https://openreview.net/forum?id=vhFu1Acb0xb)  
19. \[2209.00588\] Transformers are Sample-Efficient World Models \- arXiv, accessed July 7, 2025, [https://arxiv.org/abs/2209.00588](https://arxiv.org/abs/2209.00588)  
20. Some thoughts on autoregressive models \- Wonder's Lab, accessed July 7, 2025, [https://wonderfall.dev/autoregressive/](https://wonderfall.dev/autoregressive/)  
21. \[2406.19320\] Efficient World Models with Context-Aware Tokenization \- arXiv, accessed July 7, 2025, [https://arxiv.org/abs/2406.19320](https://arxiv.org/abs/2406.19320)  
22. What are Autoregressive Models? \- AR Models Explained \- AWS, accessed July 7, 2025, [https://aws.amazon.com/what-is/autoregressive-models/](https://aws.amazon.com/what-is/autoregressive-models/)  
23. (PDF) Transformers are Sample Efficient World Models \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/363209456\_Transformers\_are\_Sample\_Efficient\_World\_Models](https://www.researchgate.net/publication/363209456_Transformers_are_Sample_Efficient_World_Models)  
24. On Learning Informative Trajectory Embeddings for Imitation, Classification and Regression, accessed July 7, 2025, [https://arxiv.org/html/2501.09327v2](https://arxiv.org/html/2501.09327v2)  
25. How can imitation learning data be collected? \- Artificial Intelligence Stack Exchange, accessed July 7, 2025, [https://ai.stackexchange.com/questions/38572/how-can-imitation-learning-data-be-collected](https://ai.stackexchange.com/questions/38572/how-can-imitation-learning-data-be-collected)  
26. Imitation Learning | Papers With Code, accessed July 7, 2025, [https://paperswithcode.com/task/imitation-learning/codeless](https://paperswithcode.com/task/imitation-learning/codeless)  
27. Trajectories \- imitation \- Read the Docs, accessed July 7, 2025, [https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html](https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html)  
28. Behavioral Cloning (BC) \- imitation, accessed July 7, 2025, [https://imitation.readthedocs.io/en/latest/algorithms/bc.html](https://imitation.readthedocs.io/en/latest/algorithms/bc.html)  
29. 1803.07612.pdf \- Caltech Authors, accessed July 7, 2025, [https://authors.library.caltech.edu/records/jc24z-mgw41/preview/1803.07612.pdf](https://authors.library.caltech.edu/records/jc24z-mgw41/preview/1803.07612.pdf)  
30. Learning Hierarchical Policies from Unsegmented Demonstrations ..., accessed July 7, 2025, [https://www.ri.cmu.edu/publications/learning-hierarchical-policies-from-unsegmented-demonstrations-using-causal-information/](https://www.ri.cmu.edu/publications/learning-hierarchical-policies-from-unsegmented-demonstrations-using-causal-information/)  
31. Learning Exploration Strategies in Model-Based Reinforcement Learning \- UT Computer Science, accessed July 7, 2025, [https://www.cs.utexas.edu/\~pstone/Papers/bib2html-links/AAMAS13-hester.pdf](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AAMAS13-hester.pdf)  
32. Learning Exploration Strategies in Model-Based Reinforcement Learning \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/262240531\_Learning\_Exploration\_Strategies\_in\_Model-Based\_Reinforcement\_Learning](https://www.researchgate.net/publication/262240531_Learning_Exploration_Strategies_in_Model-Based_Reinforcement_Learning)  
33. Model-Free Active Exploration in Reinforcement Learning \- NIPS, accessed July 7, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/abbbb25cddb2c2cd08714e6bfa2f0634-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/abbbb25cddb2c2cd08714e6bfa2f0634-Paper-Conference.pdf)  
34. Preventing Catastrophic Forgetting in Hybrid Reinforcement Learning 1\. Introduction \- POLITesi, accessed July 7, 2025, [https://www.politesi.polimi.it/retrieve/24f07b15-20c2-49dd-be9f-1ca3de4d1eb8/Executive\_Summary\_\_\_Leonardo\_De\_Clara.pdf](https://www.politesi.polimi.it/retrieve/24f07b15-20c2-49dd-be9f-1ca3de4d1eb8/Executive_Summary___Leonardo_De_Clara.pdf)  
35. Conservative Q-Learning for Offline Reinforcement Learning \- The VITALab website, accessed July 7, 2025, [https://vitalab.github.io/article/2021/06/09/CQL.html](https://vitalab.github.io/article/2021/06/09/CQL.html)  
36. Mastering Diverse Domains through World Models arXiv ..., accessed July 7, 2025, [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)  
37. DreamerV3: Mastering Diverse Domains \- Emergent Mind, accessed July 7, 2025, [https://www.emergentmind.com/articles/2301.04104](https://www.emergentmind.com/articles/2301.04104)  
38. Dreamer V3 \- EclecticSheep, accessed July 7, 2025, [https://eclecticsheep.ai/2023/08/10/dreamer\_v3.html](https://eclecticsheep.ai/2023/08/10/dreamer_v3.html)  
39. DreamerV3: Mastering Diverse Domains through World Models \- The VITALab website, accessed July 7, 2025, [https://vitalab.github.io/article/2023/01/19/DreamerV3.html](https://vitalab.github.io/article/2023/01/19/DreamerV3.html)  
40. COMBO: Conservative Offline Model-Based Policy Optimization, accessed July 7, 2025, [https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html)  
41. COMBO: Conservative Offline Model-Based Policy Optimization | Request PDF, accessed July 7, 2025, [https://www.researchgate.net/publication/349363400\_COMBO\_Conservative\_Offline\_Model-Based\_Policy\_Optimization](https://www.researchgate.net/publication/349363400_COMBO_Conservative_Offline_Model-Based_Policy_Optimization)  
42. COMBO: Conservative Offline Model-Based Policy Optimization, accessed July 7, 2025, [https://arxiv.org/pdf/2102.08363](https://arxiv.org/pdf/2102.08363)  
43. COMBO: CONSERVATIVE OFFLINE MODEL-BASED POLICY OPTIMIZATION \- OpenReview, accessed July 7, 2025, [https://openreview.net/pdf?id=C3BddQqfrcr](https://openreview.net/pdf?id=C3BddQqfrcr)  
44. Conservative State Value Estimation for Offline Reinforcement Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/pdf/2302.06884](https://arxiv.org/pdf/2302.06884)  
45. A Continual Offline Reinforcement Learning Benchmark for Navigation Tasks \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2506.02883v1](https://arxiv.org/html/2506.02883v1)  
46. OER: Offline Experience Replay for Continual Offline Reinforcement Learning \- IOS Press Ebooks, accessed July 7, 2025, [https://ebooks.iospress.nl/doi/10.3233/FAIA230343](https://ebooks.iospress.nl/doi/10.3233/FAIA230343)  
47. Stacking Your Transformers: A Closer Look at Model Growth ... \- NIPS, accessed July 7, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/143ea4a156ef64f32d4d905206cf32e1-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/143ea4a156ef64f32d4d905206cf32e1-Paper-Conference.pdf)  
48. Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training, accessed July 7, 2025, [https://openreview.net/forum?id=FXJDcriMYH\&referrer=%5Bthe%20profile%20of%20Jie%20Fu%5D(%2Fprofile%3Fid%3D\~Jie\_Fu2)](https://openreview.net/forum?id=FXJDcriMYH&referrer=%5Bthe+profile+of+Jie+Fu%5D\(/profile?id%3D~Jie_Fu2\))  
49. \[2503.21272\] Reinforced Model Merging \- arXiv, accessed July 7, 2025, [https://arxiv.org/abs/2503.21272](https://arxiv.org/abs/2503.21272)  
50. arxiv.org, accessed July 7, 2025, [https://arxiv.org/html/2503.21272v1](https://arxiv.org/html/2503.21272v1)  
51. Reinforced Model Merging \- ChatPaper, accessed July 7, 2025, [https://chatpaper.com/chatpaper/paper/124573](https://chatpaper.com/chatpaper/paper/124573)  
52. Reinforced Model Merging \- arXiv, accessed July 7, 2025, [https://arxiv.org/pdf/2503.21272](https://arxiv.org/pdf/2503.21272)  
53. Papers Explained Review 13: Model Merging | by Ritvik Rastogi, accessed July 7, 2025, [https://ritvik19.medium.com/papers-explained-review-13-model-merging-d0db49797b90](https://ritvik19.medium.com/papers-explained-review-13-model-merging-d0db49797b90)  
54. What is Model Merging? Techniques & Challenges \- Deepchecks, accessed July 7, 2025, [https://www.deepchecks.com/glossary/model-merging/](https://www.deepchecks.com/glossary/model-merging/)  
55. What is policy distillation in RL? \- Milvus, accessed July 7, 2025, [https://milvus.io/ai-quick-reference/what-is-policy-distillation-in-rl](https://milvus.io/ai-quick-reference/what-is-policy-distillation-in-rl)  
56. Knowledge Transfer for Deep Reinforcement Learning with Hierarchical Experience Replay, accessed July 7, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/10733/10592](https://ojs.aaai.org/index.php/AAAI/article/view/10733/10592)  
57. Real-time Policy Distillation in Deep Reinforcement Learning \- ML For Systems, accessed July 7, 2025, [http://mlforsystems.org/assets/papers/neurips2019/real\_time\_sun\_2019.pdf](http://mlforsystems.org/assets/papers/neurips2019/real_time_sun_2019.pdf)  
58. SELECT: A Large-Scale Benchmark of Data Curation Strategies for Image Classification, accessed July 7, 2025, [https://openreview.net/forum?id=rIHx6puY5b](https://openreview.net/forum?id=rIHx6puY5b)  
59. SELECT: A Large-Scale Benchmark of Data Curation Strategies for Image Classification, accessed July 7, 2025, [https://www.researchgate.net/publication/384699820\_SELECT\_A\_Large-Scale\_Benchmark\_of\_Data\_Curation\_Strategies\_for\_Image\_Classification](https://www.researchgate.net/publication/384699820_SELECT_A_Large-Scale_Benchmark_of_Data_Curation_Strategies_for_Image_Classification)  
60. SELECT: A Large-Scale Benchmark of Data Curation Strategies for Image Recognition, accessed July 7, 2025, [https://nyu-dice-lab.github.io/SELECT/](https://nyu-dice-lab.github.io/SELECT/)  
61. SELECT: A Large-Scale Benchmark of Data Curation Strategies for Image Classification, accessed July 7, 2025, [https://arxiv.org/html/2410.05057v1](https://arxiv.org/html/2410.05057v1)  
62. jimmyxu123/SELECT: This is the repository for "SELECT: A Large-Scale Benchmark of Data Curation Strategies for Image Recognition" \- GitHub, accessed July 7, 2025, [https://github.com/jimmyxu123/SELECT](https://github.com/jimmyxu123/SELECT)