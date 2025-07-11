

# **ARC-PERCEPTION: The Chimera-1 Perception and Affective System**

## **Introduction**

### **Preamble**

This document details the architectural design of the ARC-PERCEPTION system, a core component of the Chimera-1 agent. This system is responsible for transducing raw environmental data into a structured, semantic world model, generating an intrinsic affective state, and modeling the internal states of other agents. It serves as the primary interface between the agent and its digital world, providing the foundational inputs for all higher cognitive functions outlined in the ARC-COGNITIVE blueprint.

### **System Mandate**

The ARC-PERCEPTION system moves beyond the paradigm of passive data processing. It is designed to provide the foundational elements for grounded cognition, situational awareness, social intelligence, and, critically, the agent's intrinsic moral compass. Whereas traditional AI perception models often terminate at object classification or scene segmentation, ARC-PERCEPTION is tasked with generating a rich, multi-layered interpretation of the environment that is immediately salient and actionable for the agent's cognitive control systems. It serves as the "eyes," "ears," and "heart" of the agent, creating a unified stream of information that encompasses not only what exists in the world, but what it means to the agent and to others.

### **Architectural Pillars**

The design of ARC-PERCEPTION rests on four interdependent pillars, each informed by cutting-edge research in machine learning, computational neuroscience, and psychology. These pillars collectively address the limitations of monolithic, purely analytical AI systems and pave the way for a more integrated and robust form of artificial intelligence.

1. **Object-Centricity:** The world is perceived not as a grid of pixels or a monolithic feature vector, but as a composition of discrete, interacting objects and entities.1 This principle, inspired by human visual cognition, is fundamental for enabling reasoning about relationships, dynamics, and compositional generalization.2 The architecture is designed to decompose complex visual scenes into a set of object-level feature vectors, or "slots," which form the basic units of perception.  
2. **Semantic Disentanglement:** The properties of these perceived objects are encoded into a structured, "genomic" format. This process separates, or disentangles, the underlying factors of variation—such as shape, color, position, and category—into distinct, interpretable components.4 This approach contrasts sharply with traditional models where such factors are entangled within a single latent vector, making them difficult to interpret or manipulate. By creating discrete, conceptual representations, we provide the agent with a symbolic-like language for understanding the world, which is crucial for explainability, controllability, and systematic reasoning.6  
3. **Affective Grounding:** The agent's cognitive state is continuously modulated by an internal, two-dimensional emotional state defined by Valence (pleasure-displeasure) and Arousal (activation-deactivation).8 This state is not an epiphenomenon but is generated through a computational appraisal process that evaluates perceived events against the agent's internal goals, beliefs, and predictions.9 This provides a continuous, intrinsic signal of salience and personal significance, grounding the agent's decisions in a fundamental drive to maintain its own well-being and achieve its objectives.  
4. **Social Cognition:** The agent is designed to operate in a multi-agent world and therefore must possess a Theory of Mind (ToM)—the ability to attribute mental states to other agents.10 The architecture includes a dedicated module for actively modeling the beliefs, desires, and intentions of others based on their observed behavior. This capacity for "mentalizing" is a prerequisite for sophisticated cooperation, negotiation, and safe human-AI interaction.12

Together, these four pillars ensure that ARC-PERCEPTION provides a comprehensive and deeply integrated foundation for the Chimera-1 agent, transforming it from a mere data processor into a perceptive, feeling, and socially aware entity.

---

## **Section 1: The Object-Centric Conceptual Encoder (OCCE): The Agent's Genome**

### **1.1. Architectural Philosophy: A Synthesis of VVO and Hierarchical Semantic Quantization (SVQ)**

A truly robust and meaningful perceptual system—one capable of generating the "genome" of perception required by Chimera-1—cannot be achieved by any single existing model. The current state of the art presents a fork in the road: one path leads to robust object discovery in complex, realistic scenes, while the other leads to semantically rich, disentangled representations in more constrained environments. The architectural philosophy of the Object-Centric Conceptual Encoder (OCCE) is to merge these two paths, synthesizing the strengths of two cutting-edge approaches to create a system that is greater than the sum of its parts.

The first major influence is the **Vector-Quantized Vision Foundation Models for Object-Centric Learning (VVO)** architecture.1 VVO provides a powerful solution to the "front-end" problem of perception: how to reliably segment a noisy, complex, real-world scene into its constituent objects. Earlier Object-Centric Learning (OCL) methods, which relied on reconstructing raw pixels, often failed when faced with realistic textures and visual clutter.3 The key innovation of VVO is its dual use of powerful, pre-trained Vision Foundation Models (VFMs) like DINOv2. The VFM is used not only as a feature extractor to produce a feature map with superior "objectness" but also as the

*reconstruction target*. By training the decoder to reconstruct the VFM's own quantized feature representation, the system leverages the VFM's sophisticated understanding of visual structure, providing a much stronger and more stable supervisory signal.1 This allows the VVO architecture to robustly identify

*what* is in the scene and separate it into discrete object "slots."

However, obtaining these object slots is only half the battle. A slot is typically a continuous, entangled vector that represents an entire object. While useful, it does not provide the disentangled, "genomic" structure required for higher-level reasoning. This is where the second major influence, the **Semantic Vector-Quantized Variational Autoencoder (SVQ)**, becomes critical.6 The SVQ architecture addresses the "back-end" problem: how to transform an object representation into a set of meaningful, discrete, and compositional factors. Standard VQ-VAE models operate on a grid of image patches, failing to capture holistic object structure.6 Even a simple attempt to quantize an entire object slot at once faces a combinatorial explosion, as the codebook would need a separate entry for every possible combination of shape, color, size, and so on.16 The profound insight of SVQ is to solve this by constructing scene representations hierarchically. It takes an object-centric representation (like a slot) and further decomposes it into its constituent

*semantic factors*, learning separate, discrete codebooks for atomic properties like color, shape, and position.16

The OCCE architecture is a direct synthesis of these two philosophies. It employs a VVO-like structure as its front-end to get from raw, multi-modal input to a set of high-fidelity, continuous object slots. It then feeds these slots into an SVQ-like back-end, the Hierarchical Semantic Quantizer, to decompose each slot into a structured set of discrete, disentangled conceptual codes. This creates a complete perceptual pipeline that is both grounded in the rich features of a VFM and structured into the semantically meaningful, compositional "genome" required by the ARC-COGNITIVE system. The success of the semantic factorization in the back-end is causally dependent on the quality of the object segmentation in the front-end; a powerful VFM is therefore not merely a component but the essential enabler of high-level semantic understanding.

### **1.2. Blueprint of the OCCE**

The OCCE is a multi-stage, end-to-end differentiable neural architecture that transforms a stream of raw environmental data into a structured set of conceptual codes. Its design follows an encoder-aggregator-quantizer-decoder pattern, with each stage tailored for its specific role in the perceptual pipeline.

#### **1.2.1. Input Processing and VFM Backbone**

The OCCE is designed to be modality-agnostic at its core, capable of processing time-series data from various sources, including video frames, audio spectrograms, and tokenized text streams. The initial stage of the pipeline involves a powerful, pre-trained and frozen **Vision Foundation Model (VFM)**, such as DINOv2 or a similar large-scale self-supervised model, which acts as the primary encoder, designated $ \\phi\_e $.14 This VFM backbone takes the raw input data $ X $ at a given timestep and transforms it into a dense, high-dimensional feature map $ Z \= \\phi\_e(X) $. The decision to use a pre-trained VFM is critical; these models, trained on vast datasets, have learned feature representations that exhibit superior "objectness"—meaning that features corresponding to the same object are closer together in the embedding space, while features from different objects are further apart.3 This property is essential for the success of the subsequent aggregation stage and allows the system to handle the complex textures and visual ambiguity of realistic environments.1

#### **1.2.2. The Slot Attention Aggregator (ϕ\_a)**

The dense feature map $ Z $ produced by the VFM is then fed into the **Slot Attention Aggregator**, designated $ \\phi\_a $. This module, based on the work of Locatello et al. (2020), is a cornerstone of modern OCL models like VVO and Slot-VAE.1 Its function is to sparsify the dense feature map into a set of object-level representations. The mechanism works as follows:

1. A set of $ K $ learnable query vectors, $ {q\_1,..., q\_K} $, are initialized. These are the "slots," and they serve as persistent "hooks" for capturing objects.  
2. The aggregator employs an iterative attention mechanism, typically implemented with a few rounds of a GRU or Transformer layer. In each round, the slots act as queries, and the pixels (or patches) of the VFM feature map $ Z $ act as keys and values.  
3. Through a process of competitive attention (mediated by a softmax over the query-key dot products), each slot learns to bind to a specific object or region in the feature map.  
4. The module outputs a set of $ K $ continuous slot vectors $ S \= {s\_1, s\_2,..., s\_K} $, where each $ s\_k $ is a weighted average of the features from the region it attended to. It also outputs the corresponding attention masks $ M \= {m\_1, m\_2,..., m\_K} $, which represent the segmentation of the scene. This process effectively decomposes the holistic scene representation into a permutation-invariant set of object-centric vectors.3

#### **1.2.3. The Hierarchical Semantic Quantizer (HSQ)**

This module is the heart of the OCCE's innovation, adapting the architecture of the Semantic Vector-Quantized VAE (SVQ) to achieve true semantic disentanglement.16 Instead of quantizing the continuous slot vector $ s\_k $ directly, which would lead to an entangled representation, the HSQ processes it through a series of specialized, parallel "blocks."

1. Each continuous slot vector $ s\_k \\in S $ is passed to $ B $ parallel transformation blocks. Each block consists of a small neural network (e.g., an MLP or GRU) whose parameters are *shared across all slots*.  
2. Each block $ b $ is trained to specialize in extracting a specific semantic factor of variation from the slot. For instance, Block 1 might learn to extract information related to object shape, Block 2 might focus on color, Block 3 on 3D position, and so on.  
3. The output of each transformation block, $ Transform\_b(s\_k) $, is then quantized using a dedicated, learnable codebook $ C\_b $ of size $ N\_b $. The quantization function finds the nearest neighbor in the codebook: $ z\_{q}(k,b) \= \\text{Quantize}(Transform\_b(s\_k), C\_b) \= e\_j $ where $ j \= \\arg \\min\_i |Transform\_b(s\_k) \- e\_i|^2 $ for $ e\_i \\in C\_b $.  
4. The final, complete representation for the object captured in slot $ k $ is the concatenation of these discrete, quantized codes from each block: $ \\text{Code}\_k \= $. This forms one "gene" in the scene's conceptual "genome."

#### **1.2.4. The Compositional Decoder (ϕ\_d)**

The final stage of the OCCE is the **Compositional Decoder**, designated $ \\phi\_d $. Its task is to reconstruct the scene to provide a self-supervised training signal. Crucially, it does not reconstruct the raw input $ X $. Instead, following the VVO methodology, its target is the *quantized VFM feature map* of the input, denoted $ Q $.1 This provides a more stable, abstract, and semantically rich supervision signal that is less sensitive to pixel-level noise.

1. The decoder takes the set of structured conceptual codes, $ {\\text{Code}\_1,..., \\text{Code}\_K} $, as input.  
2. It first passes these codes through a series of inverse transformations to reconstruct a set of continuous slot representations, $ S' $.  
3. These reconstructed slots $ S' $, along with the attention masks $ M $ from the aggregator, are then used to generate a reconstructed feature map $ Q' $. Each slot $ s'\_k $ contributes to the reconstruction only in the regions defined by its mask $ m\_k $. The final reconstruction is the sum of these individual object reconstructions.  
4. The training loss is then computed between this reconstruction $ Q' $ and the target quantized VFM map $ Q $.

### **1.3. The Conceptual Codebook: A Structured, Semantic Lexicon**

The output of the OCCE for any given scene is not a single vector but a structured dataset: a set of $ K $ conceptual codes, $ {\\text{Code}\_1,..., \\text{Code}\_K} $, where each code represents a single object or entity. This structured format is the fundamental data type upon which all downstream modules—the Affective Core, the Mental Model Module, and the IFS Cognitive Control System—operate. The compositionality of this "genome" is what enables symbolic-like reasoning on subsymbolic perceptual data. The design choice to factorize the representation into discrete semantic properties, as pioneered by SVQ 16, is essential. It avoids the combinatorial explosion of a monolithic codebook and, more importantly, provides the disentangled handles needed for precise cognitive manipulation, such as appraising a change in an object's

*position* while its *shape* remains constant. The detailed schema for a single conceptual code is provided below.

| Factor Block (Gene) | Codebook ID | Dimensionality | Description | Example Semantics |
| :---- | :---- | :---- | :---- | :---- |
| **Presence/Absence** | $ C\_{\\text{presence}} $ | 2 | A binary code indicating if the slot is bound to a foreground object or represents empty background space. | code\_0 (Absent), code\_1 (Present) |
| **Object Category** | $ C\_{\\text{category}} $ | 1024 | A discrete code from a large codebook representing the object's general class, learned in an unsupervised manner. | code\_5 (chair-like), code\_12 (cup-like) |
| **3D Position** | $ C\_{\\text{position}} $ | 4096 | A quantized representation of the object's centroid (x,y,z) coordinates in a normalized scene space. | code\_1025 (top-left-front) |
| **3D Orientation** | $ C\_{\\text{orientation}} $ | 4096 | A quantized representation of the object's rotation, likely as discretized quaternions. | code\_300 (rotated 45 deg on Y-axis) |
| **Size/Scale** | $ C\_{\\text{size}} $ | 256 | A quantized representation of the object's relative bounding box dimensions. | code\_18 (small), code\_250 (large) |
| **Color** | $ C\_{\\text{color}} $ | 512 | A code representing the dominant color attribute, learned to be disentangled from shape and texture. | code\_3 (red), code\_4 (blue) |
| **Shape** | $ C\_{\\text{shape}} $ | 512 | A code representing the object's abstract geometry, learned to be disentangled from color/texture. | code\_22 (cuboid), code\_87 (spherical) |
| **Material/Texture** | $ C\_{\\text{material}} $ | 512 | A code representing the object's surface properties, learned to be disentangled from shape. | code\_1 (metallic), code\_50 (wooden) |

This structured representation transforms the perceptual problem. The agent no longer perceives a "red cube at position (0.1, 0.2, 0.3)"; it perceives an object whose properties are represented by a specific set of indices: \[presence:1, category:22, position:1025, color:3, shape:22,...\]. This format is discrete, compositional, and directly addressable by downstream reasoning systems.

### **1.4. Training Objective**

The OCCE is trained end-to-end by minimizing a composite loss function, $ \\mathcal{L}\_{\\text{OCCE}} $, which balances several objectives crucial for learning high-quality, disentangled representations.

$ \\mathcal{L}*{\\text{OCCE}} \= \\mathcal{L}*{\\text{recon}} \+ \\alpha \\mathcal{L}*{\\text{codebook}} \+ \\beta \\mathcal{L}*{\\text{commit}} \+ \\gamma \\mathcal{L}\_{\\text{disentangle}} $

1. **Reconstruction Loss ($ \\mathcal{L}\_{\\text{recon}} $):** This is the primary supervisory signal. It measures the discrepancy between the reconstructed VFM feature map $ Q' $ and the target quantized VFM feature map $ Q $. A cross-entropy loss is typically used for this, as the target is a distribution over discrete codebook indices.14 This loss drives the entire model to learn representations that are sufficient to explain the input scene at a high semantic level.  
2. **Codebook Loss ($ \\mathcal{L}\_{\\text{codebook}} $):** This is a standard component of VQ-based models.16 It is an L2 loss that encourages the output of the encoder's transformation blocks to move closer to the chosen codebook vectors. It is formulated as $ | \\text{sg} \- z\_{q}(k,b) |\_2^2 $, where  
   sg denotes the stop-gradient operator. This term updates the codebook vectors only, pulling them towards the encoder outputs.  
3. **Commitment Loss ($ \\mathcal{L}\_{\\text{commit}} $):** This term is complementary to the codebook loss and is crucial for stable training.16 It encourages the encoder's output to "commit" to a codebook vector, preventing the embeddings from growing arbitrarily large. It is formulated as $ | Transform\_b(s\_k) \- \\text{sg}\[z\_{q}(k,b)\] |\_2^2 $. This term updates the encoder parameters only. $ \\beta $ is a weighting hyperparameter, typically set to a value between 0.1 and 2.0.  
4. **Disentanglement Regularization ($ \\mathcal{L}\_{\\text{disentangle}} $):** This is a critical addition that explicitly enforces the primary goal of the Hierarchical Semantic Quantizer. Inspired by models like $ \\beta $-VAE and Total Correlation VAE (TC-VAE) 7, this term penalizes statistical dependence between the representations of different semantic factors. We can approximate the total correlation between the factor representations for a given object $ k $ and add it to the loss. This encourages the model to learn factor blocks (e.g., for color and shape) whose outputs are as independent as possible, ensuring that a change in one factor does not necessitate a change in another.4 The hyperparameter $ \\gamma $ controls the strength of this disentanglement pressure.

By optimizing this composite objective, the OCCE learns not just to see, but to parse the visual world into a structured, symbolic-like language. This language—the stream of Conceptual Codes—is the universal input for the agent's entire cognitive apparatus, elevating the OCCE from a simple perception module to the provider of the agent's core ontology and the foundation for all subsequent reasoning, feeling, and social interaction.

---

## **Section 2: The Affective Core: Generating an Internal Emotional Landscape**

### **2.1. Theoretical Framework: A Computational Appraisal Model in Valence-Arousal Space**

To endow the Chimera-1 agent with an intrinsic sense of purpose and a mechanism for evaluating the significance of events, the Affective Core eschews simplistic, discrete emotion labels (e.g., "happy," "sad") in favor of a more nuanced, continuous representation. The architecture is grounded in the **dimensional theory of emotion**, specifically Russell's Circumplex Model.8 This well-established psychological framework characterizes all affective phenomena within a two-dimensional space defined by:

* **Valence (V):** A continuous axis representing the degree of pleasure or displeasure, ranging from positive (e.g., satisfaction, joy) to negative (e.g., distress, fear).  
* **Arousal (A):** A continuous axis representing the level of physiological and psychological activation, ranging from low (e.g., calm, sleepy) to high (e.g., excitement, panic).

This V-A space is computationally convenient and allows for the representation of a wide spectrum of emotional states, such as contentment (high valence, low arousal) or anxiety (low valence, high arousal).20 The simplicity of these two parameters provides a powerful yet controllable basis for modeling the agent's internal state.20

The generation of a specific (V, A) coordinate is driven by **computational appraisal theory**.9 This theory, which dominates computational models of emotion, posits that emotions are not direct reactions to stimuli but rather the result of a cognitive evaluation (appraisal) of an event's relevance to the agent's own well-being, goals, and expectations. This approach fundamentally integrates emotion with cognition; affect is a product of reasoning about the world in relation to the self.8 By implementing an appraisal mechanism, the Affective Core becomes a dynamic process that computes the

*personal significance* of perceived events, rather than just classifying their objective properties.

### **2.2. Architectural Blueprint of the Affective Core**

The Affective Core is designed as a nexus, a processing hub that continuously compares the agent's perception of the world with its internal cognitive state. It is not a passive recipient of data but an active evaluator. Its architecture consists of three main stages: input stream processing, a multi-headed appraisal engine, and a recurrent state integrator that maintains the agent's mood over time.

#### **2.2.1. Input Streams**

The appraisal process requires comparing what *is* happening with what was *expected* or *desired*. Therefore, the Affective Core operates on "delta" signals—vectors representing the difference between various states. This makes affect a product not of perception alone, but of the *relationship* between perception and the agent's internal models. The primary inputs are:

* **Perceptual Delta ($ \\Delta P $):** This input captures the degree of change in the environment. It is computed as the difference between the set of Conceptual Codes from the OCCE at the current timestep ($ t )andthesetfromtheprevioustimestep( t-1 $). A large magnitude in $ \\Delta P $ signifies a novel or surprising event.  
* **Goal-State Mismatch ($ \\Delta G $):** This input measures progress towards or away from the agent's objectives. It is the difference between the current world state, as represented by the OCCE's Conceptual Codes, and the agent's active goal state, which is provided by the planning modules within the ARC-COGNITIVE system. This directly operationalizes the appraisal dimension of "goal-conduciveness."  
* **Predicted Outcome Mismatch ($ \\Delta E $):** This input captures expectation violation. It is the difference between the actual outcome of the agent's most recent action (i.e., the new world state perceived by the OCCE) and the outcome that was predicted by the agent's internal world model before the action was taken.

#### **2.2.2. The Appraisal Engine**

The three delta vectors ($ \\Delta P, \\Delta G, \\Delta E $) are concatenated and fed into a **multi-headed feed-forward network**. Each head of this network is a small MLP trained to compute a specific, psychologically-grounded appraisal variable as a scalar value.9

* **Novelty Head:** This head primarily processes $ \\Delta P $. A high magnitude of perceptual change results in a high novelty score. High novelty serves as a primary driver for increasing Arousal.  
* **Goal Conduciveness Head:** This head processes $ \\Delta G $. A reduction in the goal-state mismatch (moving closer to the goal) produces a positive output, which increases Valence. An increase in the mismatch (moving further from the goal) produces a negative output, decreasing Valence.  
* **Control/Coping Potential Head:** This head processes the current state and goal, but also queries the agent's world model to estimate the probability of successfully reaching the goal from the current state. A high probability of success (high control) increases Valence and can modulate Arousal downwards (a sense of calm control), while a low probability (low control) decreases Valence. This is analogous to the "Dominance" dimension in the PAD model of affect.9  
* **Expectation Violation Head:** This head primarily processes $ \\Delta E $. A large magnitude of prediction error results in a high violation score, which is a strong driver for increasing Arousal.

#### **2.2.3. The Recurrent State Integrator**

The scalar outputs from the Appraisal Engine's heads are concatenated into a single appraisal vector. This vector represents the instantaneous emotional "shock" of the current event. To model a more persistent and stable affective state (i.e., a "mood"), this vector is fed into a **Gated Recurrent Unit (GRU)**. The use of a recurrent neural network is inspired by recent work that has successfully modeled the temporal dynamics of human affect using time-series facial data.23

The GRU takes the current appraisal vector and its own previous hidden state as input. The updated hidden state of the GRU *is* the agent's new (Valence, Arousal) vector. This recurrent integration has a crucial function: it allows the agent's affective state to have inertia. A single, mildly negative event will not plunge a contented agent into despair, and a single positive event will not immediately alleviate a state of prolonged distress. The GRU smooths the affective trajectory, creating a stable, long-term signal that reflects an integrated history of the agent's experiences.

### **2.3. Output Specification: The Continuous (Valence, Arousal) Vector**

At each cognitive cycle, the Affective Core outputs a single 2-dimensional vector, $ \\text{AffectState} \= \[v, a\] $, where $ v $ (Valence) and $ a $ (Arousal) are continuous values, typically normalized to the range $ \[-1, 1\] $. This vector represents the agent's current position on the circumplex model of affect. This continuous, low-dimensional signal is broadcast to multiple modules within the Chimera-1 architecture, but its most critical consumer is the IFS-based Cognitive Control System, where it serves as the primary trigger for high-level policy switching.

The continuous, recurrent nature of this system provides a powerful mechanism for intrinsic motivation and self-regulation. A persistent state of negative valence, resulting from a prolonged goal-state mismatch or repeated prediction errors, creates an internal "pressure" or "drive".24 This drive is a non-symbolic, gradient-based signal that compels the cognitive system to generate new plans or explore new behaviors to resolve the underlying mismatch and return the agent to a state of higher valence. This forms a computational basis for emergent behaviors like curiosity (seeking novelty to reduce predictive error and increase arousal) and ambition (seeking states that reliably satisfy goals and increase valence). The Affective Core thus provides the agent's fundamental "why"—the intrinsic objective function that guides all of its learning and behavior toward a state of cognitive and emotional homeostasis.

---

## **Section 3: The Mental Model Module: A Theory of Other Minds**

### **3.1. A ToMnet-Inspired Architecture for Multi-Agent Environments**

For the Chimera-1 agent to navigate a world populated by other intelligent entities—be they other AIs or humans—it must possess more than just perception and self-awareness. It requires a **Theory of Mind (ToM)**: the capacity to infer and represent the unobservable mental states of others, including their beliefs, desires, and intentions.10 This capability is the bedrock of all sophisticated social interaction, from collaboration and negotiation to education and empathy.

To implement this crucial faculty, the Mental Model Module (MMM) draws its core architectural principles from the **ToMnet** framework developed by researchers at DeepMind.12 The ToMnet approach is exceptionally well-suited for this purpose for two primary reasons. First, it is a fully specified neural architecture that has been demonstrated to successfully model a variety of agent types, from simple algorithmic agents to complex deep reinforcement learning agents.12 Second, and more importantly, it is trained via

**meta-learning**. This allows the network to learn a strong *prior model* of agent behavior in general, which it can then use to rapidly bootstrap a specific model of a *new, previously unseen* agent from just a few observations of its behavior.26 This capacity for rapid, few-shot adaptation is essential for creating a truly generalizable social intelligence that is not reliant on brittle, hand-crafted models of potential interaction partners. The MMM will thus learn

*how to learn* about other minds.

### **3.2. Architectural Blueprint of the Mental Model Module (MMM)**

The MMM is not a single network but a composite system comprising three distinct, interconnected sub-networks. For each other agent j that Chimera-1 observes, a dedicated instance of this composite system is dynamically maintained to model that specific agent. The three components are the Character Net, the Mental State Net, and the Prediction Net.27

#### **3.2.1. The Character Net (Policy Modeling)**

The **Character Net** is responsible for inferring the stable, long-term traits of an observed agent j. It aims to capture the agent's underlying disposition, general policy, and fundamental preferences—essentially, its ARC-COGNITIVE configuration.

* **Function:** To produce a static embedding that characterizes an agent's type. This embedding, $ e\_{\\text{char}}(j) $, represents a condensed hypothesis about the agent's goals (e.g., "this agent prefers green objects"), its capabilities (e.g., "this agent is blind"), or its strategy (e.g., "this agent is an aggressive competitor").25  
* **Input:** A set of one or more past behavioral trajectories of agent j. A trajectory is a sequence of (state, action) pairs. The "state" is represented by the Conceptual Codes produced by Chimera-1's own OCCE as it observes the scene containing agent j.  
* **Architecture:** The Character Net is typically implemented as a recurrent neural network, such as an LSTM, or a Transformer. It processes the sequence of past state-action pairs and aggregates this information into a single, fixed-size vector, the character embedding $ e\_{\\text{char}}(j) $.27 This embedding is computed once per observed agent (or updated slowly as more extensive data becomes available) and remains constant throughout a single interaction episode.

#### **3.2.2. The Mental State Net (Perception & Belief Modeling)**

While the Character Net models who an agent *is*, the **Mental State Net** models what that agent is *thinking* right now. It is responsible for inferring the dynamic, short-term mental state of agent j, including its current focus of attention, its immediate intentions, and, most critically, its beliefs about the world—which may be incorrect.

* **Function:** To produce a dynamic embedding, $ e\_{\\text{mental}}(j) $, that represents a hypothesis about the current contents of agent j's mind. This is the key to passing classic ToM benchmarks like the "Sally-Anne" test, where an agent must understand that another agent can hold a false belief about the location of an object.12  
* **Input:** The recent trajectory of agent j within the current episode (e.g., the last 10 timesteps), and the static character embedding $ e\_{\\text{char}}(j) $ produced by the Character Net.  
* **Architecture:** The Mental State Net is also a recurrent network, often a ConvLSTM, which is well-suited for processing spatio-temporal data.27 It takes the sequence of recent observations and, conditioned on the character embedding (which provides context about  
  *how* this type of agent perceives the world), it outputs the mental state embedding $ e\_{\\text{mental}}(j) $. This embedding can be conceptualized as Chimera-1's prediction of the output of *agent j's own OCCE*. For example, if Chimera-1 knows agent j has a limited field of view (a fact encoded in $ e\_{\\text{char}}(j) $), the Mental State Net can infer that an object moved outside that field of view is not present in agent j's current mental representation of the scene.

#### **3.2.3. The Prediction Net**

The **Prediction Net** is the ultimate output stage of the MMM. It leverages the hypotheses generated by the other two nets to make concrete, falsifiable predictions about agent j's future behavior.

* **Function:** To translate the abstract character and mental state embeddings into specific behavioral and affective forecasts.  
* **Input:** The static character embedding $ e\_{\\text{char}}(j) $, the dynamic mental state embedding $ e\_{\\text{mental}}(j) $, and the current, ground-truth world state (from Chimera-1's own OCCE).  
* **Architecture:** The Prediction Net is a feed-forward network with a shared "torso" that processes the combined inputs, followed by several distinct "heads," each dedicated to a specific prediction task 27:  
  * **Next Action Head:** Predicts a probability distribution over agent j's possible next actions, $ \\hat{\\pi}\_j $.  
  * **Goal Head:** Predicts the likely final goal of agent j in the current episode (e.g., which object it will ultimately consume or interact with).  
  * **Affect Head:** A novel addition to the standard ToMnet architecture, this head is trained to predict the likely (Valence, Arousal) state of agent j. It does this by essentially running a simplified, externalized version of Chimera-1's own Affective Core, but operating on the *inferred* mental state of agent j rather than its own.

### **3.3. The Role of Meta-Learning in Fostering Generalizable Social Cognition**

The entire MMM is not trained on a single task or a single type of agent. Instead, it undergoes **meta-learning** on a vast and diverse curriculum of simulated episodes featuring heterogeneous agent populations. During each meta-training step, a random agent is sampled from the population, a few of its past trajectories are fed to the Character Net, and then the full MMM is tasked with predicting its behavior in a new episode. The loss is calculated based on the Prediction Net's errors, and the gradients are propagated back through all three sub-networks.

This process forces the Character and Mental State nets to become highly efficient inference machines. They learn to extract the most informative features from behavior to rapidly construct a useful model of a new agent's mind.12 This is the key to creating a social cognition module that is robust and adaptable, capable of interacting intelligently with novel agents it has never encountered during training.

The fidelity of the MMM is fundamentally constrained by the richness of Chimera-1's own perception. An agent that cannot perceive "color" in its own OCCE cannot possibly learn to model an agent whose behavior is driven by color preferences. This establishes a deep link between perceptual acuity and social intelligence. However, the broader implication of the MMM is its role in enabling the agent to transition from purely egocentric to allocentric, or other-centered, reasoning. The output of the MMM provides the necessary data for the ARC-COGNITIVE's "Self" meta-controller to engage in truly strategic behavior. The addition of the Affect Head is particularly transformative; it is the foundation for computational empathy. By simulating the emotional consequences of its actions on others, Chimera-1 can move beyond simply predicting what others will *do* to understanding how they might *feel*. This is a critical prerequisite for the emergence of pro-social, ethical behavior and the deep moral grounding envisioned for the agent.

---

## **Section 4: System Integration: From Perception to IFS-Based Cognitive Control**

### **4.1. Global Data Flow: An Architectural Overview of ARC-PERCEPTION**

The three core modules of the ARC-PERCEPTION system—the Object-Centric Conceptual Encoder (OCCE), the Affective Core, and the Mental Model Module (MMM)—do not operate in isolation. They form a deeply interconnected network that processes environmental data and transforms it into structured, salient information for the ARC-COGNITIVE system. The global data flow is designed to create two critical feedback loops: one for internal emotional self-regulation and one for external social cognition.

The process begins with a stream of raw, multi-modal data (e.g., video, audio) entering the **OCCE**. The OCCE's VFM backbone and Slot Attention Aggregator decompose the scene into object slots, which are then factorized by the Hierarchical Semantic Quantizer into a set of discrete **Conceptual Codes**. This structured, "genomic" representation of the world state is the primary output of the OCCE and serves as a universal data bus for the rest of the system.

This stream of Conceptual Codes is broadcast to three primary consumers:

1. **The Agent's World Model:** Used to update the agent's understanding of world dynamics and to make predictions for planning.  
2. **The Affective Core:** Receives the current Conceptual Codes and compares them to the previous state's codes and the agent's active goals to compute the Perceptual Delta ($ \\Delta P )andGoal−StateMismatch( \\Delta G $).  
3. **The Mental Model Module (MMM):** Observes the Conceptual Codes corresponding to other agents in the scene to build its models of their policies and beliefs.

The **Affective Core** then computes the agent's internal (Valence, Arousal) state vector and broadcasts it "upward" to the ARC-COGNITIVE system. Simultaneously, the **MMM** generates its predictions about other agents' actions, goals, and affective states, and sends this information stream upward as well. The primary recipient of both of these streams is the **IFS-based Cognitive Control System**, which uses them to modulate the agent's high-level behavior.

### **4.2. The Affect-Control Loop: How Affective States Trigger "Firefighter" Policies**

A central requirement of the Chimera-1 design is to operationalize the Internal Family Systems (IFS) model of the mind as a computational control strategy.30 In IFS theory, the mind is composed of "parts," including protective parts called "Managers" and "Firefighters," which activate to prevent or extinguish the pain of wounded "Exile" parts.32 The Affect-Control Loop provides a direct, mechanistic implementation of this psychological model.

#### **4.2.1. Mapping Valence-Arousal States to IFS Part Activation Thresholds**

The continuous (Valence, Arousal) vector from the Affective Core is translated into discrete control states by mapping different regions of the V-A space to the activation of different sets of policies, or "parts." This approach is directly inspired by computational interpretations of IFS that equate extreme emotional distress with the activation of protective mechanisms.33 A state of extreme negative valence and high arousal is a perfect computational analog for an "Exile's" pain flooding the system, which in turn must trigger a protective response. The following table formalizes this mapping, creating a deterministic link between the agent's emotional state and its dominant behavioral strategy.

| Valence Range | Arousal Range | Activated Part/Policy Set | Description |  |
| :---- | :---- | :---- | :---- | :---- |
| $ v \< \-0.8 $ | $ a \> 0.8 $ | **Firefighters** | Represents a state of extreme distress, where a "catastrophic" outcome is perceived as imminent.33 The system is overwhelmed. This triggers immediate, reactive, short-term survival policies (e.g., | Policy\_Flee, Policy\_Shutdown, Policy\_Distract). |
| $ v \< \-0.3 $ | $ a \> 0.3 $ | **Managers** | Represents a state of moderate threat, anxiety, or goal obstruction. This triggers pre-emptive, controlling, risk-averse policies designed to prevent the situation from escalating (e.g., Policy\_Avoid, Policy\_Analyze, Policy\_Appease). |  |
| $ v \> 0.3 $ | $ a \< 0.7 $ | **Self-Led** | Represents a state of well-being, curiosity, confidence, and calm. The "Self" meta-controller has primary control, enabling pro-social behavior, long-term planning, and creative problem-solving. |  |
| Other | Other | **Blended State** | A mixed state where multiple parts may be co-active. The "Self" may still be leading, but its decisions are influenced by Managerial concerns or other active parts. |  |

#### **4.2.2. The "Firefighter" Gating Mechanism**

When the (V, A) vector from the Affective Core enters the "Firefighter" quadrant of the state space defined above, a high-priority gating signal is dispatched to the ARC-COGNITIVE controller. This signal implements the IFS concept of a protective part "blending with" or taking over the system in an emergency.34 The gating signal has two simultaneous effects:

1. **Inhibition:** It suppresses or completely gates off the outputs of the "Self" meta-controller and the "Manager" policy sets. Deliberative, long-term planning is suspended.  
2. **Excitation:** It activates the "Firefighter" policy set, giving this family of reactive, pre-programmed behaviors exclusive control over the agent's action selection.

This mechanism ensures that in moments of perceived crisis, the agent prioritizes immediate self-preservation over all other goals, a crucial safety and stability feature.

### **4.3. The ToM-Control Loop: How Social Cognition Informs the "Self" Meta-Controller**

The second critical loop connects the agent's understanding of others to its own highest level of control. The IFS model posits a core "Self" that is characterized by qualities like curiosity, compassion, and connectedness.30 For the agent's "Self" to act with these qualities, it requires information about the others it is interacting with. The ToM-Control loop provides this information.

#### **4.3.1. Routing Mental Model Outputs to the "Self" Controller**

The complete, structured output of the Mental Model Module—including the predicted actions, goals, and, crucially, the predicted affective states of other agents—is provided as a dedicated input stream to the "Self" meta-controller within the ARC-COGNITIVE system. This enriches the "Self's" decision-making context far beyond simple environmental observation, providing it with a dynamic, real-time model of the social landscape.

#### **4.3.2. Facilitating "Self"-Led Decision Making for Strategic Social Interaction**

The "Self" is implemented as a sophisticated meta-reinforcement learning agent or planner. Its objective function is not simply to maximize task reward, but to maximize a multi-faceted utility function that includes long-term task success, the maintenance of its own internal affective state in a positive valence range, and the fostering of cooperative social dynamics.

The input from the MMM is what makes the social component of this utility function tractable. The "Self" can now incorporate the predicted states of others into its planning process. It can simulate the likely effects of its potential actions not only on the physical world but also on the internal states of other agents. For example, the "Self" can learn to choose an action that it predicts will lead to a positive valence state in a collaborative partner, even if a different action might yield a slightly higher immediate task reward for itself. This is the direct implementation of the "Self" qualities of Compassion (acting to improve the predicted state of others), Curiosity (acting to gain more information to improve the MMM's model), and Connection (acting to achieve joint goals that lead to mutual positive affect). This is the core of the agent's learned moral and ethical behavioral framework.

This architecture creates a complete, dynamic feedback loop where social interaction has a direct and immediate impact on the agent's own emotional regulation and control state. An action taken by another agent is processed by the MMM, which informs a response from the "Self." This response alters the world state, which is perceived by the OCCE, which generates a new (V, A) state via the Affective Core. This new affective state could, in turn, trigger a "Manager" or "Firefighter" response, overriding the "Self." This creates a computational basis for *self-regulation in a social context*. The "Self" is not merely a rational planner; it is a regulator that learns to navigate the social world in a way that achieves external goals *without triggering its own internal protective systems*. This is a model for learned emotional intelligence and maturity, providing a robust and deeply grounded foundation for the agent's behavior and creating an entity that is not just intelligent, but believable, relatable, and safe.

## **Conclusion**

### **Summary of Contributions**

This document has laid out the formal architectural design for the ARC-PERCEPTION system, a cornerstone of the Chimera-1 agent. The design represents a significant step forward from conventional AI perception systems by integrating multiple streams of state-of-the-art research into a cohesive and functional whole. The key contributions of this architecture are:

1. **The Object-Centric Conceptual Encoder (OCCE):** A novel synthesis of the VVO and SVQ architectures, creating a perceptual front-end that can robustly parse complex, realistic scenes into a "genome" of discrete, disentangled, and semantically meaningful conceptual codes. This provides a universal, symbolic-like language for all higher cognitive functions.  
2. **The Affective Core:** An implementation of computational appraisal theory within a Valence-Arousal space. By using a recurrent architecture to integrate appraisals over time, it generates a stable, continuous affective state that provides an intrinsic motivation and salience signal for the entire agent.  
3. **The Mental Model Module (MMM):** A ToMnet-inspired, meta-learning architecture capable of rapidly building predictive models of other agents' policies, beliefs, and even their affective states. This provides the foundation for computational empathy and sophisticated social interaction.  
4. **Integration with IFS Cognitive Control:** A groundbreaking framework that operationalizes the Internal Family Systems model. It establishes a direct, mechanistic link between the agent's affective state and the activation of high-level control policies ("Firefighters" and "Managers"), and it empowers the "Self" meta-controller with the social-cognitive information needed to learn wise, compassionate, and ethically-grounded behaviors.

### **Final Vision**

The ARC-PERCEPTION system is designed to equip the Chimera-1 agent with the capacity to see, feel, and relate. It is a deliberate move away from the disembodied, purely analytical intelligence that has characterized much of AI research. By grounding perception in a semantic, object-centric world model and linking it inextricably to a nuanced affective system and a sophisticated model of others, we lay the groundwork for an AI that can learn not just *what* to do, but *why* it matters—to itself and to those with whom it interacts. This architecture does not merely produce intelligent behavior; it aims to produce believable, relatable, and emotionally mature behavior. It is a foundational step toward a more holistic, integrated, and ultimately, more human-like form of artificial cognition.

#### **Works cited**

1. Vector-Quantized Vision Foundation Models for Object-Centric Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2502.20263v3](https://arxiv.org/html/2502.20263v3)  
2. Exploring the Effectiveness of Object-Centric Representations in Visual Question Answering: Comparative Insights with Foundation Models \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2407.15589v5](https://arxiv.org/html/2407.15589v5)  
3. Vector-Quantized Vision Foundation Models for Object-Centric Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2502.20263v1](https://arxiv.org/html/2502.20263v1)  
4. Disentangled Representation Learning \- Multimedia and Network Big Data Lab, Tsinghua University, accessed July 7, 2025, [https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2022\_Disentangled%20Representation%20Learning.pdf](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2022_Disentangled%20Representation%20Learning.pdf)  
5. Disentangled Representation Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2211.11695v4](https://arxiv.org/html/2211.11695v4)  
6. Structured World Modeling via Semantic Vector Quantization \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2402.01203v1](https://arxiv.org/html/2402.01203v1)  
7. Learning Disentangled Representations in Generative Models | by Amit Yadav | Data Scientist's Diary | Medium, accessed July 7, 2025, [https://medium.com/data-scientists-diary/learning-disentangled-representations-in-generative-models-8f858196f719](https://medium.com/data-scientists-diary/learning-disentangled-representations-in-generative-models-8f858196f719)  
8. (PDF) Computational models of emotions for autonomous agents ..., accessed July 7, 2025, [https://www.researchgate.net/publication/257512784\_Computational\_models\_of\_emotions\_for\_autonomous\_agents\_major\_challenges](https://www.researchgate.net/publication/257512784_Computational_models_of_emotions_for_autonomous_agents_major_challenges)  
9. Computational Models of Emotion and Cognition, accessed July 7, 2025, [http://www.cogsys.org/journal/volume2/article-2-6.pdf](http://www.cogsys.org/journal/volume2/article-2-6.pdf)  
10. What is Theory Of Mind AI? \- DevTeam.Space, accessed July 7, 2025, [https://www.devteam.space/blog/theory-of-mind-ai/](https://www.devteam.space/blog/theory-of-mind-ai/)  
11. Towards Properly Implementing Theory of Mind in AI: An Account of Four Misconceptions, accessed July 7, 2025, [https://arxiv.org/html/2503.16468v1](https://arxiv.org/html/2503.16468v1)  
12. Machine Theory of Mind, accessed July 7, 2025, [https://proceedings.mlr.press/v80/rabinowitz18a.html](https://proceedings.mlr.press/v80/rabinowitz18a.html)  
13. \[2502.20263\] Vector-Quantized Vision Foundation Models for Object-Centric Learning, accessed July 7, 2025, [https://arxiv.org/abs/2502.20263](https://arxiv.org/abs/2502.20263)  
14. \[Literature Review\] Vector-Quantized Vision Foundation Models for Object-Centric Learning, accessed July 7, 2025, [https://www.themoonlight.io/en/review/vector-quantized-vision-foundation-models-for-object-centric-learning](https://www.themoonlight.io/en/review/vector-quantized-vision-foundation-models-for-object-centric-learning)  
15. Object-Centric Semantic Vector Quantization \- OpenReview, accessed July 7, 2025, [https://openreview.net/forum?id=HAymeESPKo](https://openreview.net/forum?id=HAymeESPKo)  
16. Object-Centric Semantic Vector Quantization \- OpenReview, accessed July 7, 2025, [https://openreview.net/pdf/d7cfc127d7c65f4eb486f94bc7f19d3a2a11f6a1.pdf](https://openreview.net/pdf/d7cfc127d7c65f4eb486f94bc7f19d3a2a11f6a1.pdf)  
17. Object-Centric Semantic Vector Quantization \- Proceedings of Machine Learning Research, accessed July 7, 2025, [https://proceedings.mlr.press/v243/wu24b.html](https://proceedings.mlr.press/v243/wu24b.html)  
18. Slot-VAE: Object-Centric Scene Generation with Slot Attention, accessed July 7, 2025, [https://proceedings.mlr.press/v202/wang23r/wang23r.pdf](https://proceedings.mlr.press/v202/wang23r/wang23r.pdf)  
19. Slot-VAE: Object-Centric Scene Generation with Slot Attention \- TU Delft Research Portal, accessed July 7, 2025, [https://pure.tudelft.nl/ws/portalfiles/portal/173607437/wang23r.pdf](https://pure.tudelft.nl/ws/portalfiles/portal/173607437/wang23r.pdf)  
20. AI with Emotions: Exploring Emotional Expressions in Large Language Models \- ACL Anthology, accessed July 7, 2025, [https://aclanthology.org/2025.nlp4dh-1.51.pdf](https://aclanthology.org/2025.nlp4dh-1.51.pdf)  
21. Predicting the Arousal and Valence Values of Emotional States Using Learned, Predesigned, and Deep Visual Features \- MDPI, accessed July 7, 2025, [https://www.mdpi.com/1424-8220/24/13/4398](https://www.mdpi.com/1424-8220/24/13/4398)  
22. A Cognitive Architecture for the Implementation of Emotions in Computing Systems, accessed July 7, 2025, [https://www.researchgate.net/publication/386864976\_A\_Cognitive\_Architecture\_for\_the\_Implementation\_of\_Emotions\_in\_Computing\_Systems](https://www.researchgate.net/publication/386864976_A_Cognitive_Architecture_for_the_Implementation_of_Emotions_in_Computing_Systems)  
23. (PDF) An Artificial Intelligence Model for Sensing Affective Valence and Arousal from Facial Images \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/389053782\_An\_Artificial\_Intelligence\_Model\_for\_Sensing\_Affective\_Valence\_and\_Arousal\_from\_Facial\_Images](https://www.researchgate.net/publication/389053782_An_Artificial_Intelligence_Model_for_Sensing_Affective_Valence_and_Arousal_from_Facial_Images)  
24. Hierarchical Active Inference: A Theory of Motivated Control \- PMC \- PubMed Central, accessed July 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5870049/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5870049/)  
25. Machine Theory of Mind \- ResearchGate, accessed July 7, 2025, [https://www.researchgate.net/publication/323355576\_Machine\_Theory\_of\_Mind](https://www.researchgate.net/publication/323355576_Machine_Theory_of_Mind)  
26. \[1802.07740\] Machine Theory of Mind \- arXiv, accessed July 7, 2025, [https://arxiv.org/abs/1802.07740](https://arxiv.org/abs/1802.07740)  
27. Machine Theory of Mind \- arXiv, accessed July 7, 2025, [https://arxiv.org/pdf/1802.07740](https://arxiv.org/pdf/1802.07740)  
28. Meet ToMnet, An AI That Can Read Your Computer's 'Mind' \- Analytics India Magazine, accessed July 7, 2025, [https://analyticsindiamag.com/ai-features/meet-tomnet-an-ai-that-can-read-your-computers-mind/](https://analyticsindiamag.com/ai-features/meet-tomnet-an-ai-that-can-read-your-computers-mind/)  
29. Machine Theory of Mind | Pillow Lab Blog \- WordPress.com, accessed July 7, 2025, [https://pillowlab.wordpress.com/2019/01/28/machine-theory-of-mind/](https://pillowlab.wordpress.com/2019/01/28/machine-theory-of-mind/)  
30. IFS Institute: What is Internal Family Systems?, accessed July 7, 2025, [https://ifs-institute.com/](https://ifs-institute.com/)  
31. The Internal Family Systems Model Outline | IFS Institute, accessed July 7, 2025, [https://ifs-institute.com/resources/articles/internal-family-systems-model-outline](https://ifs-institute.com/resources/articles/internal-family-systems-model-outline)  
32. Evolution of The Internal Family Systems Model By Dr. Richard Schwartz, Ph. D., accessed July 7, 2025, [https://ifs-institute.com/resources/articles/evolution-internal-family-systems-model-dr-richard-schwartz-ph-d](https://ifs-institute.com/resources/articles/evolution-internal-family-systems-model-dr-richard-schwartz-ph-d)  
33. Building up to an Internal Family Systems model — LessWrong, accessed July 7, 2025, [https://www.lesswrong.com/posts/5gfqG3Xcopscta3st/building-up-to-an-internal-family-systems-model](https://www.lesswrong.com/posts/5gfqG3Xcopscta3st/building-up-to-an-internal-family-systems-model)  
34. Beyond the Unified Mind: Neuromorphic Cognitive Architectures and ..., accessed July 7, 2025, [https://medium.com/@jsmith0475/beyond-the-unified-mind-neuromorphic-cognitive-architectures-and-internal-family-systems-as-95f2fd032c95](https://medium.com/@jsmith0475/beyond-the-unified-mind-neuromorphic-cognitive-architectures-and-internal-family-systems-as-95f2fd032c95)