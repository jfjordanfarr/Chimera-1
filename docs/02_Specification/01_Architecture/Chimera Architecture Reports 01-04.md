------
------
--01--
------
------





# **ARC-COGNITIVE-v2: The Chimera-1 Core Cognitive Engine**

## **Introduction & Foundational Principles**

### **Mandate and Vision for the Cognitive Engine**

The CognitiveEngine constitutes the central nervous system of the Chimera-1 agent architecture. It is the locus of reasoning, planning, and memory, responsible for transforming the continuous stream of abstract perceptual data into coherent, goal-directed action sequences. This document, **ARC-COGNITIVE**, serves as the definitive architectural blueprint for its design and implementation.

The vision for the CognitiveEngine is one of unparalleled performance and cognitive sophistication. It is designed to operate within the tight loop of perception, cognition, and actuation, where latency and inefficiency are not merely performance degradations but fundamental failures of agent viability. The architecture detailed herein is predicated on four inviolable principles:

1. **Extreme Performance:** Every architectural choice is optimized for low-latency processing and high-throughput data handling. This includes in-process communication, zero-copy data exchange, and the selection of highly optimized, Rust-native components.  
2. **Data Integrity and Robustness:** The engine must be resilient to failure. It employs ACID-compliant transactional systems for memory persistence and leverages Rust's strong type system and ownership model to eliminate entire classes of common programming errors, ensuring the agent's cognitive state remains consistent and uncorrupted.1  
3. **Cognitive Scalability:** The agent must be capable of reasoning over vast temporal horizons, integrating experiences that span seconds, hours, and days. The architecture explicitly addresses the challenge of long-range dependencies, moving beyond the limitations of traditional sequence models.3  
4. **Architectural Integrity:** The design must be maintainable and extensible over the long term. This is achieved through strict modularity, clear encapsulation, and well-defined public contracts, preventing the system from devolving into an unmanageable "big ball of mud".5

This document provides the engineering team with an exhaustive, implementation-ready blueprint. It defines not only the high-level concepts but also the concrete crate structures, public APIs, data schemas, and technology stack required to build the CognitiveEngine in strict accordance with the foundational ARC-META specification.

### **Adherence to ARC-META: The Modular Monolith Philosophy**

The foundational ARC-META document mandates a **Modular Monolith** architecture for the Chimera-1 system, with core components implemented in Rust. This choice is not an architectural compromise but a strategic decision to prioritize performance, simplicity, and robustness for the agent's core operational loop.

A modular monolith architecture provides the primary benefits of a microservices approach—namely, decoupled and cohesive modules with clear boundaries—without incurring the significant drawbacks of a distributed system.5 For an AI agent, the feedback loop between perceiving the world, reasoning about it, and acting upon it must be as tight as possible. Introducing network latency, serialization/deserialization overhead, and the complexities of distributed consensus and failure modes into this critical path would be detrimental to the agent's responsiveness and reliability. A monolithic architecture, where modules communicate via direct in-process function calls, will always be more performant for these tightly-coupled interactions than a distributed system.7 This performance differential is not a minor optimization; it is a fundamental enabler of real-time intelligent behavior.

The success of a modular monolith hinges on two core principles: **organization** and **encapsulation**.5

* **Organization** refers to the logical structure of the codebase, where the system is divided into modules that represent distinct domain or business capabilities. In Chimera-1, these modules are Perception, CognitiveEngine, and Actuation.  
* **Encapsulation** is the degree of independence between these modules. It is enforced by ensuring that modules interact only through stable, well-defined public interfaces or contracts, hiding their internal implementation details.8 This loose coupling is critical for maintainability; a change to the internal workings of one module should not break another, so long as the public contract is upheld.5

While many programming languages rely on developer discipline to maintain these boundaries, Rust provides a powerful advantage. Its strict ownership and privacy system allows architectural principles to be enforced by the compiler itself. By judiciously using visibility modifiers like pub and pub(crate), we can make the internal components of a module inaccessible to others, preventing improper coupling at compile time.10 This transforms an architectural guideline into a verifiable guarantee, ensuring the long-term integrity of the system and making Rust a superior language for implementing a robust modular monolith.

This architecture also provides a pragmatic path for future evolution. Should a specific module, such as a computationally intensive perception pipeline, require independent scaling, its clear boundaries and contracted interface make it a prime candidate for extraction into a separate service. The modular monolith serves as a high-performance starting point that does not preclude a future transition to a more distributed topology if and when the need arises.6 For the core of Chimera-1, however, the focus remains on maximizing in-process performance and transactional simplicity.

### **High-Level System Data Flow**

To contextualize the detailed design of the CognitiveEngine, it is essential to understand its place within the broader Chimera-1 data flow. The engine acts as the central hub, mediating between perception and action.

The data flow follows a logical progression:

1. **Perception Input: The Object-Centric Genomic Code.** The Perception module processes raw sensory data, but it moves beyond simple patch-based analysis, which struggles with complex textures and semantic understanding.42 Instead, it generates a true "genomic" code through a process of  
   **Object-Centric Semantic Vector Quantization**. First, a mechanism like **Slot Attention** decomposes the visual scene into a set of abstract "slots," where each slot corresponds to a distinct object or entity in the input.43 Following this,  
   **Semantic Vector Quantization (SVQ)** is applied not to raw feature patches, but hierarchically at the object level.45 Each object slot is decomposed into low-level, discrete concept schemas (e.g., color, shape, texture) which are then composed into the final object representation.47 The resulting  
   **Conceptual Codes** are structured, disentangled representations that capture the semantic properties of individual objects, providing a rich, compositional input stream to the CognitiveEngine that is suitable for complex reasoning.47 These codes are streamed to the  
   CognitiveEngine in batches.  
2. **Cognitive Processing:** The CognitiveEngine receives the ConceptualCode stream. Its primary internal component, the **HASSM Planner**, integrates this new information into its generative world model. This process involves updating its internal latent state and storing salient information in the long-term **Memory System**. When tasked with a goal, the HASSM planner projects a sequence of future states, forming a coherent plan.  
3. **Memory Interaction:** Throughout its operation, the CognitiveEngine continuously interacts with its memory subsystem. It writes new experiences (derived from percepts) to a persistent database and reads past knowledge—both semantic (facts) and procedural (skills)—to inform its planning and reasoning processes.  
4. **Actuation Output:** The output of the CognitiveEngine is a **Plan**, which is a structured sequence of intended actions or sub-goals. This plan is passed to the Actuation module.  
5. **Action Execution:** The Actuation module is responsible for translating the abstract plan from the CognitiveEngine into concrete, low-level commands that can be executed by the agent's physical or virtual effectors.

This flow creates a continuous loop where the agent perceives, thinks, and acts, with the CognitiveEngine serving as the critical reasoning core that connects sensory input to purposeful output. The following sections will dissect the internal architecture of this engine in exhaustive detail.

## **Module Architecture and Public API**

In adherence to the modular monolith philosophy, the CognitiveEngine is implemented as a distinct Rust crate within the Chimera-1 workspace. This crate encapsulates all logic related to planning, memory, and reasoning. Interaction with this module from other parts of the system is strictly limited to a single, well-defined public interface, the CognitiveEngineFacade.

### **Crate Structure: cognitive\_engine**

The cognitive\_engine crate is organized to reflect its internal sub-components, promoting a clear separation of concerns. This structure ensures that developers can work on specific areas like memory persistence or planner logic with minimal interference from other parts of the engine.

The directory structure for the crate is as follows:

chimera-1/  
└── src/  
    ├── cognitive\_engine/  
    │   ├── planner/  
    │   │   ├── mod.rs          // Public interface for the planner sub-module  
    │   │   └── hassm.rs        // HASSM implementation details  
    │   ├── memory/  
    │   │   ├── mod.rs          // Public interface for the memory sub-module  
    │   │   ├── persistence.rs  // redb database interaction logic  
    │   │   └── cache.rs        // moka in-memory cache logic  
    │   ├── ffi/  
    │   │   ├── mod.rs          // FFI module definition (private to the crate)  
    │   │   └── arrow\_bridge.rs // PyO3 and Apache Arrow FFI implementation  
    │   ├── error.rs            // Crate-specific error types  
    │   └── lib.rs              // Defines the public CognitiveEngine Facade  
    ├── perception/             // Placeholder for the Perception module crate  
    ├── actuation/              // Placeholder for the Actuation module crate  
    └── main.rs                 // The main application bootstrapper

The corresponding Cargo.toml for the cognitive\_engine crate will declare its primary dependencies, which form the technical foundation of the module:

Ini, TOML

\[package\]  
name \= "cognitive\_engine"  
version \= "0.1.0"  
edition \= "2021"

\[dependencies\]  
\# Core asynchronous runtime  
tokio \= { version \= "1", features \= \["full"\] }

\# Persistence Layer  
redb \= "2.1"

\# In-Memory Caching  
moka \= { version \= "0.12", features \= \["future"\] }

\# Python FFI Bridge  
pyo3 \= { version \= "0.21", features \= \["extension-module"\] }

\# Apache Arrow for Zero-Copy Data Exchange  
arrow \= "52.0"  
arrow-schema \= "52.0"  
arrow-array \= "52.0"  
pyo3-arrow \= "0.10"

\# Error Handling  
thiserror \= "1.0"

\# Serialization (for data stored in redb)  
serde \= { version \= "1.0", features \= \["derive"\] }  
bincode \= "1.3"

This structure establishes a clear boundary. The lib.rs file acts as the gatekeeper, exposing only the intended public API (CognitiveEngine). The internal modules like ffi are not declared as pub, making them private to the cognitive\_engine crate and enforcing the encapsulation principle at the compiler level.11

### **The CognitiveEngineFacade: A Unified Interface**

To manage the complexity of the CognitiveEngine's internal systems, its public API is implemented using the **Facade** design pattern.9 The

CognitiveEngine struct serves as this facade, providing a simplified, high-level interface to the rest of the Chimera-1 application. It encapsulates the intricate interactions between the planner, the memory system, and the Python FFI bridge, presenting a clean and stable contract to its clients (e.g., the main application loop or other modules).

This approach offers several key advantages:

* **Decoupling:** Client modules do not need to know about the existence of redb, moka, or the HASSM planner. They interact only with the CognitiveEngine, which hides these implementation details.9  
* **Simplicity:** The facade reduces the number of objects that clients need to handle. Instead of managing separate handles for a planner, a database connection, and a cache, clients interact with a single, unified object.  
* **Centralized Control:** All interactions are routed through the facade, making it an ideal place to manage cross-cutting concerns such as logging, metrics, and transactional control.

The design of the CognitiveEngine struct and its public methods adheres strictly to the Rust API Guidelines.13 This includes using

snake\_case for method names, comprehensive Result-based error handling, and ensuring that all public-facing structs have private fields to maintain strong encapsulation (Guideline C-STRUCT-PRIVATE).13

The conceptual implementation of the facade is as follows:

Rust

// in cognitive\_engine/lib.rs

// Publicly expose types needed for the API contract, but not implementations.  
pub use self::error::{EngineError, Plan, Goal, MemoryQuery, MemoryResult, PerceptBatch};

mod planner;  
mod memory;  
mod error;

// The FFI module is kept private to the crate.  
mod ffi;

/// Configuration for initializing the CognitiveEngine.  
pub struct EngineConfig {  
    pub database\_path: std::path::PathBuf,  
    pub semantic\_cache\_size\_mb: u64,  
    pub procedural\_cache\_size\_mb: u64,  
}

/// The public facade for the entire cognitive module.  
pub struct CognitiveEngine {  
    // Internal state is private, enforcing encapsulation.  
    planner\_handle: planner::PlannerHandle,  
    memory\_handle: memory::MemoryHandle,  
    // Other internal state, e.g., tokio runtime handles  
}

impl CognitiveEngine {  
    /// Creates and initializes a new instance of the CognitiveEngine.  
    /// This is the sole entry point for creating the engine.  
    pub async fn new(config: EngineConfig) \-\> Result\<Self, EngineError\> {  
        // Complex initialization logic for memory, planner, and FFI is hidden here.  
        //...  
    }

    // \--- Public API Methods \---

    /// Ingests a batch of new perceptual information into the engine's world model.  
    pub async fn process\_percepts(&self, percepts: PerceptBatch) \-\> Result\<(), EngineError\> {  
        // Delegates to internal memory and planner components.  
        //...  
    }

    /// Generates a high-level plan to achieve the specified goal.  
    pub async fn generate\_plan(&self, goal: Goal) \-\> Result\<Plan, EngineError\> {  
        // Invokes the HASSM planner via its internal handle.  
        //...  
    }

    /// Executes a structured query against the agent's long-term memory.  
    pub async fn query\_memory(&self, query: MemoryQuery) \-\> Result\<MemoryResult, EngineError\> {  
        // Delegates to the internal memory handle, which manages caching and persistence.  
        //...  
    }  
}

The formal contract provided by this facade is summarized in the table below. This table is the canonical reference for any developer integrating with the CognitiveEngine. Its existence enforces the principle of communicating via well-defined interfaces, which is paramount in a modular monolith architecture.5

**Table 2.2.1: CognitiveEngine Public API Methods**

| Method Signature | Parameters | Return Value | Description | Key Error Conditions |
| :---- | :---- | :---- | :---- | :---- |
| pub async fn new(config: EngineConfig) \-\> Result\<Self, EngineError\> | config: EngineConfig: Specifies the database file path and in-memory cache sizes. | Result\<Self, EngineError\> | Initializes all sub-components of the engine, including opening the redb database, configuring moka caches, and setting up the Python FFI bridge. | EngineError::DbOpenFailed, EngineError::FfiInitFailed |
| pub async fn process\_percepts(\&self, percepts: PerceptBatch) \-\> Result\<(), EngineError\> | percepts: PerceptBatch: An Apache Arrow RecordBatch containing new conceptual codes from the Perception module. | Result\<(), EngineError\> | Ingests new information about the world. This triggers updates to the HASSM's internal state and writes salient memories to the persistence layer. | EngineError::InvalidPerceptFormat, EngineError::MemoryWriteFailed |
| pub async fn generate\_plan(\&self, goal: Goal) \-\> Result\<Plan, EngineError\> | goal: Goal: A high-level, structured representation of the agent's current objective. | Result\<Plan, EngineError\> | Invokes the HASSM planner to generate a sequence of abstract actions or sub-goals required to achieve the provided goal. | EngineError::PlanningFailed, EngineError::UnreachableGoal |
| pub async fn query\_memory(\&self, query: MemoryQuery) \-\> Result\<MemoryResult, EngineError\> | query: MemoryQuery: A structured query for retrieving semantic (factual) or procedural (skill-based) information. | Result\<MemoryResult, EngineError\> | Retrieves information from the agent's long-term memory store, transparently handling cache lookups and database queries. | EngineError::QueryFailed, EngineError::MemoryReadFailed |

## **The Generative World Model: HASSM Planner Architecture**

The core of the CognitiveEngine is its generative world model and planner, named the **Hierarchical Attentive State-Space Model (HASSM)**. This model is responsible for understanding the temporal flow of events, predicting future states, and generating plans to achieve goals. Its design is a synthesis of recent advances in sequence modeling, drawing inspiration from hierarchical cognition in humans and leveraging the computational efficiency of state-space models.

### **Conceptual Framework: Hierarchical Planning**

Human cognition excels at managing complexity by decomposing large problems into smaller, more manageable sub-problems organized in a hierarchy.15 We plan a trip not by listing every footstep, but by outlining high-level goals ("fly to Paris," "visit the Louvre") which are then broken down into finer-grained actions. The HASSM architecture formalizes this intuitive process for the Chimera-1 agent.

The model operates on the principle that an agent's experience can be understood at multiple temporal scales. There are fine-grained, moment-to-moment events, and there are coarse-grained, long-term narratives that connect these events. A planner that can reason at both levels is fundamentally more powerful and efficient. To achieve this, HASSM adopts a hierarchical structure, stacking state-space models to create a temporal hierarchy that can process and predict sequences at different levels of abstraction.16

### **HASSM Two-Level Architecture**

The HASSM planner is composed of two distinct but interconnected state-space models (SSMs), each operating at a different temporal resolution. This structure is directly inspired by the Hierarchical State-Space Models (HiSS) architecture, which has demonstrated superior performance in continuous sequence-to-sequence prediction tasks.16

**1\. Low-Level SSM: The Abstraction Layer**

* **Function:** This model's primary role is temporal abstraction. It consumes the high-frequency stream of ConceptualCode vectors provided by the Perception module. It processes this stream in fixed-size temporal "chunks" (e.g., a window of 256 consecutive codes). For each chunk, it produces a single, compressed LocalFeature vector.  
* **Analogy:** This is analogous to the HiSS architecture's low-level SSM, which processes raw sensor data into abstract "chunk features".16 In HASSM, this layer transforms a rapid sequence of "what is being seen" into a more abstract representation of "what just happened."  
* **Output:** The output is a new sequence of LocalFeature vectors, which is temporally down-sampled from the original input stream. This sequence represents the agent's experience at a coarser, more abstract level.

**2\. High-Level SSM: The Reasoning and Planning Layer**

* **Function:** This model takes the sequence of LocalFeature vectors from the low-level model as its input. Operating at a lower temporal frequency but over a much longer history, its purpose is to model the long-range dependencies and causal relationships between these abstract events. It learns the "narrative" of the agent's experience. When given a goal, this is the layer that performs forward-looking simulation to generate a plan.  
* **Analogy:** This mirrors the high-level SSM in the HiSS architecture, which maps the sequence of chunk features to a global prediction.16 In HASSM, this layer reasons about the sequence of "what just happened" to predict "what will happen next" and to decide "what to do next."

This two-level structure provides a crucial computational advantage. The expensive, long-range reasoning required for planning is performed on the compressed, abstract sequence produced by the low-level model, not on the raw, high-frequency perceptual stream. This separation of concerns allows each layer to be specialized for its task: the low-level model for efficient local feature extraction, and the high-level model for deep, long-term temporal reasoning.

### **Core Sequence Model: S4A (Structured State-Space with Attention)**

Both the low-level and high-level models in HASSM are built upon a powerful sequence modeling backbone called **S4A (Structured State-Space with Attention)**. This architecture is a hybrid that combines the efficiency of Structured State-Space Models (S4) with the contextual flexibility of an attention mechanism.

#### **S4 for Long-Range Dependencies**

The choice of S4 as the base model is a direct response to the limitations of other architectures. Transformer models, while powerful, exhibit computational and memory complexity that scales quadratically with sequence length (O(L2)).3 For an agent designed to operate continuously over potentially unbounded time horizons, this quadratic scaling is untenable.

Structured State-Space Models (SSMs) like S4, S5, and Mamba have emerged as a superior alternative for long-sequence tasks.3 They are derived from classical state-space models in control theory and are formulated to capture long-range dependencies with near-linear computational complexity (

O(LlogL) or O(L)).3 S4, in particular, has demonstrated state-of-the-art performance on benchmarks requiring reasoning over sequences of tens of thousands of elements, a capability that is essential for Chimera-1's long-term memory and planning.4 S4 achieves this by representing the sequence transformation as a continuous-time system, which can be efficiently computed as a global convolution during training and as a highly parallelizable recurrent system during inference.

#### **The "Attentive" Component: Memoryful Dynamics with ALiBi**

While the base S4 model is highly efficient, its state transitions are fundamentally Markovian—the next state depends only on the current state and the current input. To enable more sophisticated, context-dependent reasoning, HASSM incorporates an attention mechanism, drawing inspiration from Attentive State-Space Models.22 This creates

**"memoryful" dynamics**, where the transition to the next state is a function of the *entire* history of past states, not just the most recent one.

In the high-level S4A model, the state transition is augmented by an attention mechanism that looks back over the history of LocalFeature vectors. The influence of each past feature on the current transition is determined by a learned attention weight. This allows the model to dynamically refer to salient past events when planning for the future.

To ground this attention mechanism in a robust sense of time, we employ **Attention with Linear Biases (ALiBi)**.24 ALiBi is a simple yet remarkably effective technique that modifies the standard attention calculation. Instead of adding complex positional embeddings to the input, ALiBi adds a static, non-learned bias directly to the query-key attention scores. This bias is a linear penalty proportional to the distance between the query and key tokens. The attention score calculation becomes:

Attention(qi​,K)=softmax(dk​​qi​KT​+m⋅\[−(i−1),…,−1,0\])

Here, m is a fixed, head-specific scalar that determines the steepness of the penalty. This simple addition provides the model with a strong inductive bias towards recency, as closer tokens receive less of a penalty.

The use of ALiBi provides a critical advantage for an autonomous agent: **length extrapolation**. Models using traditional positional encodings often fail catastrophically when presented with sequences longer than those they were trained on.26 In contrast, ALiBi has been shown to allow models to generalize effectively to sequences that are many times longer than their training context.24 This property is not a mere convenience; it is a fundamental requirement for agent robustness. Chimera-1 will inevitably encounter situations that are longer and more complex than any single training scenario. ALiBi ensures that its core reasoning capability will not degrade or collapse in these novel, long-duration situations, providing a level of resilience that is essential for deployment in open-ended environments.

### **Reasoning and Planning Flow**

The HASSM planner operates in a continuous cycle of abstraction and prediction. The data flow for a single planning operation can be broken down into the following steps:

1. **Percept Ingestion:** The Perception module emits a continuous stream of ConceptualCode vectors. These are buffered by the CognitiveEngine.  
2. **Low-Level Abstraction:** The Low-Level S4 model consumes these codes in fixed-size chunks (e.g., a chunk of 256 codes). For each chunk, it runs its forward pass and outputs a single LocalFeature vector. This vector is a compressed, abstract representation of the events that occurred during that time window. This process runs continuously, building up a history of LocalFeature vectors.  
3. **High-Level State Update:** The High-Level S4A model receives this new sequence of LocalFeature vectors. It updates its internal latent state to reflect the latest abstract events, effectively integrating the new information into its understanding of the world's narrative.  
4. **Goal-Conditioned Planning:** When the CognitiveEngine receives a generate\_plan request with a specific Goal embedding, the planning phase begins. The High-Level S4A model, conditioned on its current latent state and the Goal vector, operates in a generative, auto-regressive mode.  
5. **Future Projection:** The model projects a sequence of *future* LocalFeature vectors. It predicts the next most likely feature vector, appends it to its history, and uses this new, extended history to predict the subsequent vector, and so on. This forward simulation is guided by the Goal and the model's learned dynamics of the world. The attention mechanism, powered by ALiBi, is active during this process, allowing the model to reference its entire past experience when deciding on the next step.  
6. **Plan Finalization:** The resulting sequence of predicted future LocalFeature vectors constitutes the **Plan**. This is an abstract plan, representing a trajectory through the space of high-level world states.  
7. **Plan Execution:** This abstract plan is returned by the CognitiveEngineFacade. It is then passed to the Actuation module, which contains a simpler decoder model responsible for translating each abstract LocalFeature vector in the plan into a sequence of concrete, low-level motor commands.

This entire process, from abstraction to prediction, is handled within the Python-based AI components, managed and orchestrated by the Rust core via the FFI boundary.

### **HASSM Integration with the Internal Family Systems (IFS) Cognitive Control System**

The HASSM planner does not operate in isolation. It is a critical component within a broader cognitive control architecture inspired by the Internal Family Systems (IFS) psychological model.50 This architecture provides a robust framework for managing the interplay between deliberative planning and reactive responses, ensuring both goal-directed behavior and rapid, protective actions. The system consists of three key roles: Managers, Firefighters, and the Self.51

The HASSM Planner as "Manager"  
The HASSM planner embodies the Manager part of the IFS model.52 It is the proactive, deliberative component responsible for long-term, goal-oriented planning. It analyzes the world state, simulates future outcomes, and generates complex plans to achieve high-level objectives, as described in the previous sections. Its function is to maintain control and guide the agent towards desirable future states in a structured, thoughtful manner.54  
Reactive "Firefighter" Policies  
In parallel to the Manager, the system incorporates a set of Firefighter policies.55 These are highly-optimized, reactive policies that are triggered by specific, urgent environmental cues (e.g., imminent threats, system integrity warnings) detected by the  
Perception module. Unlike the Manager's computationally intensive deliberative process, Firefighters execute immediate, pre-defined action sequences to "extinguish the fire" and protect the agent from immediate harm, acting as an emergency response system.52 Their goal is to impulsively distract from or suppress overwhelming negative states, prioritizing immediate safety over long-term consequences.57

The "Self" as Meta-Controller and Arbitrator  
The Self acts as the central meta-controller, governing the overall cognitive flow but not engaging in planning itself.51 Its primary function is  
**arbitration**: deciding which part—the Manager (HASSM) or a Firefighter—has control over the agent's actions at any given moment.58 This arbitration is not static but is a dynamic process based on the current context, ensuring a balance between thoughtful planning and necessary reactivity.

Architectural Hooks and Control Flow  
The integration of these parts is achieved through a clear control flow within the CognitiveEngine, creating a hybrid architecture that balances deliberative and reactive control 59:

* **Default State:** Under normal operating conditions, the Self delegates control to the Manager (HASSM planner). The generate\_plan method proceeds as a deliberative process, allowing for thoughtful, goal-oriented behavior.  
* **Firefighter Preemption:** The Self meta-controller continuously monitors the incoming PerceptBatch for pre-defined Firefighter trigger conditions. If a high-priority trigger is detected (e.g., a signal indicating imminent danger), the Self immediately preempts any ongoing HASSM planning process.60 Control is passed to the relevant Firefighter policy, which outputs a reactive plan directly to the  
  Actuation module for immediate execution.  
* **Uncertainty-Based Arbitration:** The Self also modulates control based on the Manager's confidence. If the HASSM planner returns a plan with a low confidence score (as indicated by the confidence\_score in the hassm\_output\_v1 schema) or if the agent is in a highly uncertain state, the Self may arbitrate in favor of a more conservative, pre-defined Firefighter policy over the Manager's novel but potentially unreliable plan.58 This provides a crucial safety layer, balancing goal-directed exploration with robust, predictable behavior when faced with ambiguity.

## **The Memory System: Persistence and Caching**

An agent's intelligence is inextricably linked to its memory. The Chimera-1 memory system is designed to be a robust, high-performance, and tiered architecture that supports the agent's cognitive functions. It consists of a durable persistence layer for long-term knowledge, built on a Rust-native embedded database, and a high-speed in-memory caching layer that serves as the agent's working memory.

### **Persistence Layer: redb Embedded Database**

The foundation of the agent's long-term memory is a persistent key-value store. After a thorough evaluation of available technologies, **redb** has been selected as the persistence layer.

#### **Rationale for redb**

redb is a pure Rust, embedded key-value database that offers a compelling combination of features perfectly suited to the needs of the CognitiveEngine 1:

* **Performance and Architecture:** redb is architecturally inspired by LMDB, utilizing copy-on-write (CoW) B-trees. This design is highly optimized for read performance, which is the dominant workload for the CognitiveEngine's planner as it queries memory for contextual information. Write performance, while slightly lower than Log-Structured Merge-tree (LSM) based stores like RocksDB, is more than sufficient for ingesting new memories, and the architecture avoids the high write amplification and compaction stalls associated with LSM-trees.2  
* **ACID Compliance and Crash Safety:** redb provides fully ACID-compliant transactions, guaranteeing that the agent's memory remains consistent even in the event of a crash or power loss.1 This is a non-negotiable requirement for a robust autonomous agent.  
* **Concurrency with MVCC:** redb implements Multi-Version Concurrency Control (MVCC). This allows multiple concurrent read transactions to proceed without blocking a single, concurrent write transaction.1 This feature is architecturally critical for Chimera-1. It enables the planning thread to read from a stable, consistent snapshot of the database while a separate memory consolidation thread simultaneously writes new experiences from the perception stream. This decoupling prevents the planning process from being stalled by memory updates, ensuring agent responsiveness.  
* **Type Safety and Ergonomics:** Unlike many key-value stores that operate on raw byte arrays (\[u8\]), redb provides a type-safe, BTreeMap\<K, V\>-like API.2 This allows us to define tables with specific Rust types for keys and values, leveraging the Rust compiler to prevent data type mismatches and serialization errors at compile time.

#### **Database Schema and Tables**

The entire memory of the agent will be stored in a single database file, chimera\_memory.redb. Within this file, we will define several tables to organize different types of knowledge. The schema is designed to be simple, robust, and aligned with the agent's needs.

**Table 4.1.1: semantic\_memory Table**

This table stores the agent's repository of general, factual knowledge about the world—concepts, entities, and their relationships. This is analogous to human semantic memory.

| Key | Value | Description |
| :---- | :---- | :---- |
| ConceptID (u64) | ConceptData (struct) | The primary key is a unique 64-bit identifier for a concept. The value is a serde-serializable Rust struct containing the concept's name, a textual description, pre-computed vector embeddings, and a list of (RelationType, ConceptID) tuples linking it to other concepts. |

**Table 4.1.2: procedural\_memory Table**

This table stores the agent's learned skills and action sequences. It is the repository of "how-to" knowledge, analogous to human procedural memory.

| Key | Value | Description |
| :---- | :---- | :---- |
| SkillID (u64) | SkillData (struct) | The primary key is a unique 64-bit identifier for a skill. The value is a serde-serializable struct containing the skill's name, its applicability pre-conditions, expected post-conditions, and the abstract action sequence that defines it. |

**Table 4.1.3: episodic\_buffer Multimap Table**

This table serves as the agent's long-term, time-ordered log of experiences. It stores the sequence of LocalFeature vectors produced by the HASSM planner's low-level model. This buffer is the raw material for episodic memory recall and offline learning. redb's support for multimap tables is used here, allowing multiple entries to be associated with a single key if necessary, though in this case, the timestamp key will be unique.

| Key | Value | Description |
| :---- | :---- | :---- |
| Timestamp (u128) | LocalFeatureVector (Vec\<f32\>) | The key is a high-resolution nanosecond timestamp, ensuring a strict chronological ordering of events. The value is the abstract feature vector produced by the low-level SSM, representing the agent's experience at that moment. |

### **In-Memory Caching Strategy: The Working Memory**

To bridge the significant performance gap between CPU-speed reasoning and disk-based I/O from redb, a sophisticated in-memory caching layer is essential. This layer functions as the agent's "working memory," keeping frequently and recently accessed information readily available to the planner.

#### **Technology Choice and Rationale**

The caching layer will be implemented using the **moka** crate. moka is a high-performance, concurrent cache library for Rust, heavily inspired by the well-regarded Java Caffeine library.30 It is the ideal choice for several reasons:

* **Concurrency:** moka is designed from the ground up for use in multi-threaded, asynchronous applications. It provides thread-safe access to the cache without coarse-grained locking, making it suitable for our concurrent planning and memory-writing threads.30  
* **Rich Eviction Policies:** Unlike simpler LRU (Least Recently Used) caches, moka supports a variety of eviction policies, including LRU, LFU (Least Frequently Used), and time-to-live/time-to-idle expiration.30 This allows us to tailor the caching strategy to the specific access patterns of different data types.  
* **Performance:** It is one of the fastest cache implementations available in the Rust ecosystem, designed to minimize contention and maximize hit rates in demanding workloads.

#### **Cache Configuration and Access Pattern**

The CognitiveEngine will maintain two primary caches to serve different types of memory queries:

1. **Semantic Cache:** An **LRU (Least Recently Used)** cache will be used for ConceptData. This is based on the assumption that concepts relevant to the agent's current context are likely to be accessed repeatedly in a short period. The cache will have a configurable memory size limit (e.g., 512 MB).  
2. **Procedural Cache:** An **LFU (Least Frequently Used)** cache will be used for SkillData. This policy is better suited for skills. A core, fundamental skill might be used very frequently over the agent's lifetime, but not necessarily in the immediate past. An LFU policy ensures that these valuable, frequently-used skills are retained in the cache even if they haven't been accessed recently, preventing them from being evicted by more transient, task-specific skills. This cache will also have a configurable size limit (e.g., 256 MB).

The memory access pattern will follow the standard **cache-aside** strategy:

1. When the CognitiveEngine's query\_memory method is called, it first attempts to retrieve the requested data from the appropriate moka cache.  
2. **On a cache hit,** the data is returned immediately at in-memory speed.  
3. **On a cache miss,** the engine proceeds to query the redb persistence layer.  
4. The data retrieved from redb is then inserted into the moka cache before being returned to the caller. This ensures that subsequent requests for the same data will result in a cache hit.

This tiered architecture, combining the non-blocking concurrency of redb's MVCC with the high-performance, thread-safe caching of moka, creates a memory system that is simultaneously durable, consistent, and highly responsive, capable of meeting the stringent performance demands of the Chimera-1 agent.

## **Polyglot Integration Boundary (Rust-Python FFI)**

A core tenet of the ARC-META blueprint is to leverage the "best tool for the job." While Rust provides unparalleled performance and safety for the system's core logic and data management, the Python ecosystem offers an unrivaled, mature suite of tools for deep learning research and development, most notably PyTorch. The CognitiveEngine architecture therefore embraces a polyglot design, establishing a clean, efficient, and robust Foreign Function Interface (FFI) between the Rust-based core and the Python-based AI models.

### **FFI Bridge Architecture: PyO3**

The bridge between Rust and Python will be built using **PyO3**. PyO3 is the de-facto standard and most mature library for creating Rust bindings for the Python interpreter.32 It provides a comprehensive and ergonomic toolkit for seamless interoperability:

* **Bidirectional Calls:** PyO3 allows Rust code to call Python functions and classes, and for Python code to call Rust functions.34  
* **Type Conversion:** It handles the conversion of native types between the two languages (e.g., Rust String to Python str, Rust Vec\<T\> to Python list).  
* **Error Handling:** It provides a mechanism to translate Rust's Result\<T, E\> into Python exceptions, allowing for natural error handling on both sides of the boundary.33  
* **GIL Management:** PyO3 correctly manages the Python Global Interpreter Lock (GIL), ensuring that all calls into the Python runtime are safe and conform to Python's concurrency model.33

To maintain strict encapsulation, all FFI-related code will be isolated within a dedicated, private ffi module inside the cognitive\_engine crate. This module will contain all \#\[pyfunction\] and \#\[pymodule\] definitions, creating a single, explicit, and well-audited surface of interaction between the two languages.35

### **Zero-Copy Data Exchange with Apache Arrow**

For a high-performance agent, the cost of data transfer across the FFI boundary is a major concern. Serializing large data structures (like batches of tensors) in Rust, copying the bytes to Python, and then deserializing them is computationally expensive and introduces significant latency. To eliminate this overhead, the Chimera-1 architecture mandates the use of **Apache Arrow** for all large data transfers across the FFI.

Apache Arrow is a language-agnostic, columnar in-memory data format specification.37 Its key advantage is that it enables

**zero-copy** data sharing. Because libraries in different languages (like Rust's arrow-rs and Python's pyarrow) implement the exact same memory layout, it is possible to pass a pointer to an Arrow data structure from one language to another without copying the underlying data buffers.38 The receiving language can read the data directly from the memory location where the sending language created it.

The **pyo3-arrow** crate provides the critical glue to make this possible between PyO3 and pyarrow.40 It implements the Arrow PyCapsule Interface, a standard protocol for sharing Arrow objects in Python. When a Rust function passes an

arrow-rs object to Python via pyo3-arrow, it is exposed as a Python object that pyarrow can understand and use directly, achieving a true zero-copy transfer.40

This choice has a profound, positive impact on the entire system. By defining the CognitiveEngine's data interfaces in terms of Arrow, it incentivizes upstream modules like Perception and downstream modules like Actuation to also adopt an Arrow-native data format. For example, the Perception module will be most efficient if it generates its ConceptualCode vectors directly into an Arrow buffer, which can then be passed through the CognitiveEngine and across the FFI to Python without any copies. This propagates a high-performance, zero-copy data pipeline throughout the entire agent architecture, a systemic benefit that stems directly from this FFI design decision.

### **Arrow Data Schemas**

To ensure stability and prevent integration errors, the structure of the data crossing the FFI boundary must be formally and strictly defined. These Apache Arrow schemas serve as a machine-readable contract between the Rust core and the Python AI models. Any change to these schemas requires a version increment and coordinated updates on both sides of the bridge.

**Table 5.3.1: Arrow Schema for HASSM Input (hassm\_input\_v1)**

| Field Name | Data Type | Nullable | Description |
| :---- | :---- | :---- | :---- |
| timestamp\_ns | Timestamp(Nanosecond, None) | No | High-precision UNIX timestamp of the primary percept in the batch. |
| conceptual\_codes | List(Field(Float32)) | No | A list of one or more vector embeddings representing the percepts from the Perception module. |
| memory\_context | List(Field(Float32)) | Yes | An optional context vector retrieved from memory to augment the input. NULL if no context is provided. |
| goal\_embedding | List(Field(Float32)) | Yes | The vector embedding of the current goal. This field is only populated for generate\_plan calls; otherwise, it is NULL. |

**Table 5.3.2: Arrow Schema for HASSM Output (hassm\_output\_v1)**

| Field Name | Data Type | Nullable | Description |
| :---- | :---- | :---- | :---- |
| plan\_feature\_sequence | List(Field(List(Field(Float32)))) | No | A list of LocalFeature vectors. Each inner list represents one step in the generated abstract plan. |
| attention\_weights | List(Field(List(Field(Float32)))) | Yes | The attention weight matrices from the final planning step. Provided for introspection, debuggability, and explainability. Can be NULL to save bandwidth. |
| confidence\_score | Float32 | No | A score between 0.0 and 1.0 representing the model's confidence in the validity and reachability of the generated plan. |

### **Example FFI Workflow**

The complete zero-copy data exchange process follows these steps:

1. **Rust (Data Preparation):** The CognitiveEngine's generate\_plan method assembles the necessary data: the latest conceptual\_codes, a relevant memory\_context vector, and the goal\_embedding.  
2. **Rust (arrow-rs):** Using arrow-rs builders, it constructs an in-memory Arrow RecordBatch that conforms precisely to the hassm\_input\_v1 schema. This operation is fast as it involves writing data into pre-allocated buffers.  
3. **Rust to Python (pyo3-arrow):** The Rust code calls the Python FFI function (e.g., \_hassm\_predict(...)), passing the RecordBatch. The pyo3-arrow conversion logic takes over, creating a Python object that wraps the Rust RecordBatch and exposes the Arrow PyCapsule Interface. This step does not copy the gigabytes of tensor data; it only transfers ownership and pointers.  
4. **Python (pyarrow):** The Python function receives the object. It can be immediately used as a pyarrow.RecordBatch without any conversion or deserialization cost.  
5. **Python (torch):** Data from the pyarrow.RecordBatch can be converted to PyTorch tensors. Libraries like torch have native support for creating tensors from objects that support Python's buffer protocol, which pyarrow arrays do. This conversion can also be zero-copy in many cases, especially for numerical data.  
6. **Python (Model Inference):** The tensors are fed into the PyTorch HASSM model for the forward pass.  
7. **Python (pyarrow):** The output tensors from the model (the plan, attention weights, etc.) are used to construct a new pyarrow.RecordBatch that conforms to the hassm\_output\_v1 schema.  
8. **Python to Rust (pyo3-arrow):** The Python function returns this output RecordBatch. pyo3-arrow intercepts the return value and performs the reverse zero-copy conversion, providing the Rust side with a native arrow-rs RecordBatch.  
9. **Rust (Processing):** The CognitiveEngine can now read the plan directly from the returned RecordBatch and pass it to the Actuation module for execution.

This tightly integrated, zero-copy workflow is fundamental to achieving the performance targets of the Chimera-1 project, enabling the system to harness the power of Python's AI ecosystem without sacrificing the performance of its Rust core.

## **Conclusion: A Cohesive Cognitive Core**

The architecture detailed in this document, **ARC-COGNITIVE**, provides a comprehensive and robust blueprint for the mind of the Chimera-1 agent. It is not merely a collection of disparate technologies but a cohesive, synergistic system where each component is chosen and designed to reinforce the others, working in concert to meet the project's demanding requirements for performance, scalability, and intelligence.

The four principal pillars of this design form an integrated whole:

1. **The Modular Monolith in Rust:** This foundational choice prioritizes raw performance and transactional simplicity for the critical perception-cognition-action loop by eliminating network overhead.7 Rust's strong compiler guarantees enforce the modular boundaries, ensuring long-term architectural integrity and preventing the decay into a "big ball of mud" that plagues less disciplined systems.5  
2. **The HASSM Planner:** This hierarchical, attentive state-space model provides the engine with a sophisticated mechanism for reasoning. By stacking S4 models, it creates a temporal hierarchy that can abstract fine-grained events into coarse-grained narratives, enabling efficient, long-range planning.16 The integration of ALiBi-powered attention makes its dynamics "memoryful" and robustly resilient to the length-extrapolation failures common in other sequence models, a critical feature for an agent in an open-ended world.22  
3. **The redb and moka Memory System:** This tiered memory architecture provides a durable, high-performance foundation for agent knowledge. redb offers ACID-compliant, crash-safe persistence with a read-optimized B-tree structure and non-blocking MVCC, which is perfectly suited for concurrent reading (planning) and writing (consolidation).1 The  
   moka caching layer provides a high-speed working memory, with tailored eviction policies (LRU and LFU) to intelligently manage the most relevant semantic and procedural knowledge.30  
4. **The Apache Arrow FFI:** This zero-copy integration boundary forms a high-bandwidth, low-latency bridge to the Python AI ecosystem. By leveraging Apache Arrow's columnar format via pyo3-arrow, we can seamlessly pass large tensor payloads to and from PyTorch models without the crippling overhead of serialization.39 This decision propagates a culture of high-performance, columnar data handling throughout the entire agent architecture.

Together, these components create a cognitive engine that is performant by design, robust by construction, and intelligent by its very structure. This blueprint provides the Chimera-1 development team with a clear, unambiguous, and extensible foundation upon which to build the next generation of autonomous agent intelligence.

#### **Works cited**

1. Crate redb \- Rust \- Docs.rs, accessed July 7, 2025, [https://docs.rs/redb](https://docs.rs/redb)  
2. redb: high performance, embedded, key-value database in pure Rust \- Reddit, accessed July 7, 2025, [https://www.reddit.com/r/rust/comments/uahh4y/redb\_high\_performance\_embedded\_keyvalue\_database/](https://www.reddit.com/r/rust/comments/uahh4y/redb_high_performance_embedded_keyvalue_database/)  
3. A Survey on Structured State Space Sequence (S4) Models \- arXiv, accessed July 7, 2025, [https://arxiv.org/pdf/2503.18970?](https://arxiv.org/pdf/2503.18970)  
4. \[R\] The Annotated S4: Efficiently Modeling Long Sequences with Structured State Spaces, accessed July 7, 2025, [https://www.reddit.com/r/MachineLearning/comments/s5hajb/r\_the\_annotated\_s4\_efficiently\_modeling\_long/](https://www.reddit.com/r/MachineLearning/comments/s5hajb/r_the_annotated_s4_efficiently_modeling_long/)  
5. May I Interest You In a Modular Monolith? | Frontend at Scale, accessed July 7, 2025, [https://frontendatscale.com/issues/45/](https://frontendatscale.com/issues/45/)  
6. Building a Modular Monolith With Vertical Slice Architecture in .NET : r/dotnet \- Reddit, accessed July 7, 2025, [https://www.reddit.com/r/dotnet/comments/1kda70x/building\_a\_modular\_monolith\_with\_vertical\_slice/](https://www.reddit.com/r/dotnet/comments/1kda70x/building_a_modular_monolith_with_vertical_slice/)  
7. Modular Monolith: Architectural Drivers \- Kamil Grzybek, accessed July 7, 2025, [https://www.kamilgrzybek.com/blog/posts/modular-monolith-architectural-drivers](https://www.kamilgrzybek.com/blog/posts/modular-monolith-architectural-drivers)  
8. Why Break It If It Works? The Case for Modular Monoliths | by Nakul Shukla \- Medium, accessed July 7, 2025, [https://nakulshukla.medium.com/why-break-it-if-it-works-the-case-for-modular-monoliths-58bdad671fe8](https://nakulshukla.medium.com/why-break-it-if-it-works-the-case-for-modular-monoliths-58bdad671fe8)  
9. Facade \- Refactoring.Guru, accessed July 7, 2025, [https://refactoring.guru/design-patterns/facade](https://refactoring.guru/design-patterns/facade)  
10. Modular Monoliths Are a Good Idea \- Hacker News, accessed July 7, 2025, [https://news.ycombinator.com/item?id=41534179](https://news.ycombinator.com/item?id=41534179)  
11. Revisiting Rust's modules, part 2 \- \#95 by newpavlov \- language design, accessed July 7, 2025, [https://internals.rust-lang.org/t/revisiting-rust-s-modules-part-2/5700/95?u=newpavlov](https://internals.rust-lang.org/t/revisiting-rust-s-modules-part-2/5700/95?u=newpavlov)  
12. Facade in Rust / Design Patterns \- Refactoring.Guru, accessed July 7, 2025, [https://refactoring.guru/design-patterns/facade/rust/example](https://refactoring.guru/design-patterns/facade/rust/example)  
13. Rust API Guidelines Checklist, accessed July 7, 2025, [https://rust-lang.github.io/api-guidelines/checklist.html](https://rust-lang.github.io/api-guidelines/checklist.html)  
14. About \- Rust API Guidelines, accessed July 7, 2025, [https://rust-lang.github.io/api-guidelines/about.html](https://rust-lang.github.io/api-guidelines/about.html)  
15. Discovery of hierarchical representations for efficient planning \- Gershman Lab, accessed July 7, 2025, [https://gershmanlab.com/pubs/Tomov20.pdf](https://gershmanlab.com/pubs/Tomov20.pdf)  
16. Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling, accessed July 7, 2025, [https://hiss-csp.github.io/](https://hiss-csp.github.io/)  
17. Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2402.10211v3](https://arxiv.org/html/2402.10211v3)  
18. Hierarchical State Space Models for Continuous Sequence ... \- GitHub, accessed July 7, 2025, [https://raw.githubusercontent.com/mlresearch/v235/main/assets/bhirangi24a/bhirangi24a.pdf](https://raw.githubusercontent.com/mlresearch/v235/main/assets/bhirangi24a/bhirangi24a.pdf)  
19. \[2503.18970\] From S4 to Mamba: A Comprehensive Survey on Structured State Space Models \- arXiv, accessed July 7, 2025, [https://arxiv.org/abs/2503.18970](https://arxiv.org/abs/2503.18970)  
20. Paper page \- A Survey on Structured State Space Sequence (S4) Models, accessed July 7, 2025, [https://huggingface.co/papers/2503.18970](https://huggingface.co/papers/2503.18970)  
21. Structured State Space Models for In-Context Reinforcement Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2303.03982](https://arxiv.org/html/2303.03982)  
22. Attentive State-Space Modeling of Disease Progression, accessed July 7, 2025, [http://papers.neurips.cc/paper/9311-attentive-state-space-modeling-of-disease-progression.pdf](http://papers.neurips.cc/paper/9311-attentive-state-space-modeling-of-disease-progression.pdf)  
23. Attentive State-Space Modeling of Disease Progression \- NIPS, accessed July 7, 2025, [https://papers.nips.cc/paper/9311-attentive-state-space-modeling-of-disease-progression](https://papers.nips.cc/paper/9311-attentive-state-space-modeling-of-disease-progression)  
24. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, accessed July 7, 2025, [https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)  
25. Train Short, Test Long: Attention with Linear Biases Enables Input ..., accessed July 7, 2025, [https://arxiv.org/pdf/2108.12409](https://arxiv.org/pdf/2108.12409)  
26. ALiBi: Attention with Linear Biases | by Amy Pajak \- Medium, accessed July 7, 2025, [https://medium.com/@pajakamy/alibi-attention-with-linear-biases-942abe042e9f](https://medium.com/@pajakamy/alibi-attention-with-linear-biases-942abe042e9f)  
27. Context-aware Biases for Length Extrapolation \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2503.08067v1](https://arxiv.org/html/2503.08067v1)  
28. 1.0 release\! \- redb, accessed July 7, 2025, [https://www.redb.org/post/2023/06/16/1-0-stable-release/](https://www.redb.org/post/2023/06/16/1-0-stable-release/)  
29. cberner/redb: An embedded key-value database in pure Rust \- GitHub, accessed July 7, 2025, [https://github.com/cberner/redb](https://github.com/cberner/redb)  
30. Caching — list of Rust libraries/crates // Lib.rs, accessed July 7, 2025, [https://lib.rs/caching](https://lib.rs/caching)  
31. Caching \- Categories \- crates.io: Rust Package Registry, accessed July 7, 2025, [https://crates.io/categories/caching](https://crates.io/categories/caching)  
32. Introduction \- PyO3 user guide, accessed July 7, 2025, [https://pyo3.rs/](https://pyo3.rs/)  
33. pyo3 \- Rust \- Docs.rs, accessed July 7, 2025, [https://docs.rs/pyo3/latest/pyo3/](https://docs.rs/pyo3/latest/pyo3/)  
34. PyO3/pyo3: Rust bindings for the Python interpreter \- GitHub, accessed July 7, 2025, [https://github.com/PyO3/pyo3](https://github.com/PyO3/pyo3)  
35. Rust-Python FFI | dora-rs, accessed July 7, 2025, [https://dora-rs.ai/blog/rust-python/](https://dora-rs.ai/blog/rust-python/)  
36. How I published my 1st Rust-Python binding package | by Senhaji Rhazi hamza \- Medium, accessed July 7, 2025, [https://hamza-senhajirhazi.medium.com/how-i-published-my-1st-rust-python-binding-package-cb44bc4e2e94](https://hamza-senhajirhazi.medium.com/how-i-published-my-1st-rust-python-binding-package-cb44bc4e2e94)  
37. Apache arrow array implementation : r/rust \- Reddit, accessed July 7, 2025, [https://www.reddit.com/r/rust/comments/15o53ei/apache\_arrow\_array\_implementation/](https://www.reddit.com/r/rust/comments/15o53ei/apache_arrow_array_implementation/)  
38. Memory and IO Interfaces — Apache Arrow v20.0.0, accessed July 7, 2025, [https://arrow.apache.org/docs/python/memory.html](https://arrow.apache.org/docs/python/memory.html)  
39. Calling Rust from Python using PyO3 \- Hacker News, accessed July 7, 2025, [https://news.ycombinator.com/item?id=29368530](https://news.ycombinator.com/item?id=29368530)  
40. pyo3\_arrow \- Rust \- Docs.rs, accessed July 7, 2025, [https://docs.rs/pyo3-arrow](https://docs.rs/pyo3-arrow)  
41. kylebarron/arro3: A minimal Python library for Apache Arrow, connecting to the Rust arrow crate \- GitHub, accessed July 7, 2025, [https://github.com/kylebarron/arro3](https://github.com/kylebarron/arro3)  
42. Vector-Quantized Vision Foundation Models for Object-Centric Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2502.20263v1](https://arxiv.org/html/2502.20263v1)  
43. Slot Attention Explained \- Papers With Code, accessed July 7, 2025, [https://paperswithcode.com/method/slot-attention](https://paperswithcode.com/method/slot-attention)  
44. Object-Centric Learning with Slot Attention \- NIPS, accessed July 7, 2025, [https://papers.neurips.cc/paper\_files/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf](https://papers.neurips.cc/paper_files/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf)  
45. Structured World Modeling via Semantic Vector Quantization \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2402.01203v1](https://arxiv.org/html/2402.01203v1)  
46. Object-Centric Semantic Vector Quantization \- OpenReview, accessed July 7, 2025, [https://openreview.net/forum?id=HAymeESPKo](https://openreview.net/forum?id=HAymeESPKo)  
47. Object-Centric Semantic Vector Quantization \- Proceedings of Machine Learning Research, accessed July 7, 2025, [https://proceedings.mlr.press/v243/wu24b/wu24b.pdf](https://proceedings.mlr.press/v243/wu24b/wu24b.pdf)  
48. \[2211.11695\] Disentangled Representation Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/abs/2211.11695](https://arxiv.org/abs/2211.11695)  
49. Explicitly Disentangled Representations in Object-Centric Learning \- arXiv, accessed July 7, 2025, [https://arxiv.org/html/2401.10148v1](https://arxiv.org/html/2401.10148v1)  
50. Beyond the Unified Mind: Neuromorphic Cognitive Architectures and Internal Family Systems as Convergent Models of Distributed Consciousness | by Dr. Jerry A. Smith \- Medium, accessed July 7, 2025, [https://medium.com/@jsmith0475/beyond-the-unified-mind-neuromorphic-cognitive-architectures-and-internal-family-systems-as-95f2fd032c95](https://medium.com/@jsmith0475/beyond-the-unified-mind-neuromorphic-cognitive-architectures-and-internal-family-systems-as-95f2fd032c95)  
51. The Internal Family Systems Model Outline | IFS Institute, accessed July 7, 2025, [https://ifs-institute.com/resources/articles/internal-family-systems-model-outline](https://ifs-institute.com/resources/articles/internal-family-systems-model-outline)  
52. IFS Coaching 101: Firefighter Parts \- Cressana LLC, accessed July 7, 2025, [https://www.cressanallc.com/blog/ifs-coaching-101-firefighter-parts](https://www.cressanallc.com/blog/ifs-coaching-101-firefighter-parts)  
53. IFS Protector Parts (Managers and Firefighters) \- what are they? \- Therapy with Alessio, accessed July 7, 2025, [https://www.therapywithalessio.com/articles/ifs-protector-parts-what-are-they](https://www.therapywithalessio.com/articles/ifs-protector-parts-what-are-they)  
54. Our inner system of managers, firefighters, and exiles and the dynamics of their relationships, accessed July 7, 2025, [https://sequencewiz.org/2023/04/26/our-inner-system-of-managers-firefighters-and-exiles-and-the-dynamics-of-their-relationships/](https://sequencewiz.org/2023/04/26/our-inner-system-of-managers-firefighters-and-exiles-and-the-dynamics-of-their-relationships/)  
55. Understanding Firefighter Parts in Internal Family Systems (IFS) Therapy, accessed July 7, 2025, [https://www.themindfultherapist.co/post/understanding-firefighter-parts-in-internal-family-systems-therapy](https://www.themindfultherapist.co/post/understanding-firefighter-parts-in-internal-family-systems-therapy)  
56. Reactive and Deliberative AI agents \- Vikas Goyal, accessed July 7, 2025, [https://vikasgoyal.github.io/agentic/reactivedeliberativeagents.html](https://vikasgoyal.github.io/agentic/reactivedeliberativeagents.html)  
57. Understanding Internal Family Systems (IFS): Exploring the Role of Firefighters \- Medium, accessed July 7, 2025, [https://medium.com/@compassionatetalktherapy/understanding-internal-family-systems-ifs-exploring-the-role-of-firefighters-a33f4d6474bd](https://medium.com/@compassionatetalktherapy/understanding-internal-family-systems-ifs-exploring-the-role-of-firefighters-a33f4d6474bd)  
58. Neural computations underlying arbitration between model-based and model-free learning, accessed July 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3968946/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3968946/)  
59. Understanding Agent Architecture: The Frameworks Powering AI Systems \- HatchWorks AI, accessed July 7, 2025, [https://hatchworks.com/blog/ai-agents/agent-architecture/](https://hatchworks.com/blog/ai-agents/agent-architecture/)  
60. Building up to an Internal Family Systems model \- LessWrong, accessed July 7, 2025, [https://www.lesswrong.com/posts/5gfqG3Xcopscta3st/building-up-to-an-internal-family-systems-model](https://www.lesswrong.com/posts/5gfqG3Xcopscta3st/building-up-to-an-internal-family-systems-model)




------
------
--02--
------
------


{Your Report}


#### **Revised Architectural Commission 2: The Perception & Affective System**

**The Prompt:**

Hi Gemini!

I'm an enterprise full-stack developer for a legal/financial firm (I don't love the industry, but I do find myself loving the work), a published life scientist (bioinformatics work in Neuron, some stem cell work that got used by another group that landed a Cell paper), an indie game developer of over 10 years, a father of a five year old, and a person who loves learning. I span many backgrounds and find myself in AI work today at my day job. I've been commissioning a series of Gemini Deep Research reports, building up the ingredients and vision for a larger overall project. We are continuing that process. As such, you'll see many reports attached.

I am providing you with the complete project history, including our foundational `ARC-META` blueprint and the newly generated `ARC-COGNITIVE` design. We are continuing the **"Architecture"** phase.

Your next mission is to design the complete **Perception and Affective System**. This system is the agent's interface to its digital world and, critically, the source of its intrinsic moral grounding. The design must be fully integrated with the `ARC-COGNITIVE` blueprint.

Your research must produce a document that details:

1.  **The Object-Centric Conceptual Encoder:** This is a critical enhancement. The design for the **HQ-VAE** must go beyond simple reconstruction. It must incorporate principles from **Semantic Vector Quantization (SVQ)** and **object-centric representation learning** (e.g., Slot Attention). Detail the architecture that ensures the learned "conceptual codes" correspond to disentangled, meaningful objects and their properties, not just arbitrary data patches. This is the foundation of the agent's "genome."
2.  **The Affective Core Architecture:** Design the architecture for the module that computes the agent's internal affective state (Valence, Arousal). How does it process inputs to generate this state?
3.  **The Mental Model Module:** Design the architecture for the agent's Theory of Mind capability. How does it build and maintain a model of other agents' beliefs, desires, and intentions?
4.  **Integration with the "Internal Family System":** Detail the data flow that shows how the outputs of these modules (object-centric concepts, affect, and mental models) are fed into the **IFS-based Cognitive Control System**. Specifically, how does the Affective Core's output trigger the "Firefighter" policies, and how does the Mental Model Module inform the "Self" meta-controller?

**The Deliverable:** A formal architectural document titled **"ARC-PERCEPTION: The Chimera-1 Perception and Affective System."**





------
------
--03--
------
------




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




------
------
--04--
------
------




# **Chimera-1: A Blueprint for a Learning Lifecycle**

Document ID: CHIMERA-BP-LL-001  
Version: 1.0 (Final)  
Status: Approved

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