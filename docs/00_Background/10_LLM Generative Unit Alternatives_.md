

# **Alternative Generative Units for LLMs: A Comparative Analysis for the Chimera-1 Architecture**

## **1\. Executive Summary & Introduction**

### **1.1 The Autoregressive Paradigm and Its Discontents**

The modern era of large language models (LLMs) has been overwhelmingly defined by a single, remarkably effective training objective: autoregressive next-token prediction (NTP).1 In this paradigm, a model learns to predict the next token in a sequence given all preceding tokens. This simple, self-supervised objective, when applied at unprecedented scale, has unlocked emergent capabilities in language understanding, generation, and reasoning that have fundamentally reshaped the field of artificial intelligence.2 The success of architectures from GPT to PaLM is a testament to the power of this approach.1

However, as the field matures and the demands placed on these models grow more complex, the inherent limitations of the standard NTP paradigm have become increasingly apparent and acute. These limitations can be categorized into three primary areas of concern:

* **Computational Inefficiency:** The autoregressive nature of generation is fundamentally sequential. To generate a sequence of length N, the model must perform N separate forward passes, with each pass generating a single token. This process is memory-bandwidth bound and represents a significant bottleneck for inference latency and throughput.5 As models grow larger and applications demand real-time interaction, this sequential dependency becomes a critical barrier to deployment and scalability. Efforts to parallelize decoding are thus a major focus of current research.8  
* **Cognitive Myopia:** The NTP objective incentivizes the model to make a locally optimal choice at each step—predicting the most probable next token. While effective for learning grammar and local semantics, this myopic focus can hinder the development of long-range coherence, global consistency, and sophisticated planning.11 When generating long-form text, such as a technical report or a multi-turn dialogue, models can meander, repeat themselves, or lose the logical thread of an argument. This is because the model lacks an explicit mechanism to plan its future output beyond the immediate next token, a process analogous to the distinction between fast, intuitive "System 1" thinking and slow, deliberate "System 2" reasoning.11 This deficit is a primary reason why even state-of-the-art LLMs struggle with complex, multi-step planning tasks that require foresight.13  
* **Representational Brittleness:** The "token" in next-token prediction is not a natural linguistic unit but an artifact of a preprocessing step called tokenization. Common subword tokenization schemes like Byte-Pair Encoding (BPE) introduce a fixed, discrete vocabulary that creates a fundamental disconnect between the model and the raw data.16 This leads to several problems. "Tokenization bias" can render perfectly valid outputs (e.g., a code snippet ending mid-token) impossible for the model to generate because they do not align with the tokenizer's segmentation.18 Models struggle with out-of-vocabulary words, misspellings, and novel jargon. Most critically, tokenization is a profound obstacle to true, end-to-end multimodality. Current systems that claim to be multimodal often use separate, specialized encoders for different data types (e.g., a vision transformer for images, a text model for language), which are then "stitched" together. This is not a unified understanding of data but a federation of specialists. A truly universal model must operate on a more fundamental representation of information.19

### **1.2 The Generative Unit as a Foundational Architectural Choice**

In light of these challenges, this report argues that the choice of the **generative unit**—the fundamental quantum of information the model is trained to predict—is a foundational architectural decision, as critical as the choice of attention mechanism or network depth. Moving beyond the single, subword token opens a design space of alternative paradigms that can directly address the limitations of standard autoregression. The unit of generation dictates the model's inductive biases, its computational performance profile, the nature of its internal representations, and ultimately, the ceiling of its capabilities. By rethinking *what* a model predicts, we can fundamentally alter *how* it "thinks."

### **1.3 Report Objectives and Structure**

This report provides a deep research analysis of the bleeding edge of alternative generative units in LLM engineering. Its purpose is to conduct a rigorous, comparative investigation to inform the final architectural design of the Chimera-1 model. The analysis will critically examine four distinct paradigms, evaluating their core mechanisms, strengths, weaknesses, and, most importantly, their potential integration with the specified components of the Chimera-1 blueprint.

The report is structured as follows:

* **Section 2** investigates **Multi-Token Prediction**, a paradigm focused on accelerating inference by generating multiple tokens in parallel, and its synergy with the Chimera-1 **Hydra Layer**.  
* **Section 3** explores **Next-Byte Prediction**, a quest for a universal, tokenizer-free model, and its profound implications for the **ALiBi-SSM core** and true multimodality.  
* **Section 4** examines **Next-"Chunk" Prediction**, which elevates the generative unit to a semantically coherent concept, and its relationship with the **multimodal sensory system** and **joint embedding space**.  
* **Section 5** delves into **Hierarchical & Multi-Scale Prediction**, a paradigm aimed at instilling long-range planning and reasoning, and how a "planning loss" could be integrated into the training of the **Semantic Core**.  
* **Section 6** concludes with a **Synthesis and Prescriptive Recommendation**, weaving together the insights from the preceding sections to propose a novel, hybrid generative architecture for Chimera-1, designed to achieve state-of-the-art efficiency and reasoning capabilities.

## **2\. Paradigm I: Multi-Token Prediction — Accelerating Inference through Parallelism**

The most direct and widely explored alternative to single-token generation is Multi-Token Prediction (MTP). The core motivation behind MTP is to overcome the primary efficiency bottleneck of autoregressive models: the sequential nature of decoding. By enabling the model to generate multiple tokens in a single forward pass, MTP aims to significantly increase inference throughput and reduce latency.6 This paradigm is less about changing the fundamental nature of the model's reasoning and more about accelerating the expression of that reasoning.

### **2.1 Core Mechanisms: From Parallel Heads to Speculative Decoding**

The implementation of MTP has evolved from direct architectural modifications to more flexible, algorithm-driven approaches.

* **Parallel Prediction Heads:** An intuitive approach to MTP involves augmenting the standard transformer architecture with multiple "prediction heads." Instead of a single output layer that predicts the next token at position t+1, the model is fitted with k heads, each trained to predict the token at positions t+1,t+2,...,t+k simultaneously.21 While straightforward, this method faces a significant challenge: the predictions for future tokens (  
  t+2 and beyond) are made without knowledge of the actual preceding tokens, leading to a compounding of errors and a degradation in output quality as the prediction horizon increases. This makes naive parallel prediction difficult to scale effectively.  
* **Speculative Decoding: The "Draft-then-Verify" Paradigm:** The dominant and most successful MTP technique is speculative decoding.6 This is an algorithmic framework that elegantly sidesteps the quality degradation issue of parallel heads by using a "draft-then-verify" process. It requires two models: a large, high-quality  
  **target model** (Mq​), which represents the desired output distribution, and a smaller, faster **draft model** (Mp​).7  
  The process unfolds in two stages at each decoding step:  
  1. **Drafting:** The efficient draft model Mp​ is run autoregressively for K steps to generate a sequence of candidate future tokens. This is computationally cheap due to the small size of the draft model.7  
  2. **Verification:** The powerful target model Mq​ then takes the original context and the K drafted tokens and performs a *single forward pass* to compute the probability distributions for all K+1 positions in parallel. Each drafted token is then compared against the target model's prediction. If the draft matches the target's likely output (based on a specific verification criterion), it is accepted. This continues until a mismatch is found.7

The result is that multiple tokens can be accepted and decoded in a single pass of the expensive target model, leading to significant speedups, often in the range of 2-4x, while mathematically preserving the exact output distribution of the original target model.6 The primary trade-off is between the latency of the draft model and the acceptance rate of its speculative tokens; a more capable (and thus slower) drafter will have a higher acceptance rate, and finding the optimal balance is key.23

* **Advanced Drafting and Verification Strategies:** The research landscape is rich with variations on this core theme.  
  * **Drafter Selection:** Drafters can be smaller, off-the-shelf models from the same family as the target (e.g., a 1B parameter model drafting for a 7B model), which benefits from shared tokenizers and training data.7 Alternatively, specialized drafters can be fine-tuned or trained from scratch, sometimes using knowledge distillation from the target model to improve alignment.7 Self-drafting techniques even use the target model itself, through mechanisms like early-exiting or layer-skipping, to generate drafts.7  
  * **Verification Strategies:** The simplest verification is greedy, where a draft is accepted only if it is the single most likely token according to the target model. More sophisticated methods like **speculative sampling** use a probabilistic acceptance criterion (r\<min(1,q(xe​)/p(xe​))) that allows for stochastic sampling while preserving the target distribution.7  
    **Token Tree Verification** further enhances parallelism by having the drafter propose multiple candidate sequences, which are arranged in a tree and verified in a single pass by the target model using a specialized attention mask.7

### **2.2 Advanced Frontiers in Multi-Token Prediction**

While standard speculative decoding is a powerful optimization, recent work has begun to push the MTP paradigm beyond pure acceleration, exploring methods that also influence the model's capabilities.

* **Leap Multi-Token Prediction (L-MTP):** This innovative technique, proposed in 21, modifies the MTP objective to predict non-sequential tokens. Instead of predicting tokens at positions  
  t+1,t+2,t+3, an L-MTP model might predict tokens at t+1,t+3,t+5, strategically skipping intermediate positions. This "leap-based" mechanism provides a broader training signal, forcing the model to capture longer-range dependencies than adjacent-token MTP allows. During inference, the skipped tokens can be filled in from previous steps, and the method integrates well with speculative decoding to further accelerate generation.21 This approach hints at how an MTP-like objective can be used not just for speed, but to encourage the model to learn more abstract, long-range relationships.  
* **Register-Based Prediction (MuToR):** The MuToR framework 26 offers a simple yet effective way to achieve MTP without any architectural changes to a pretrained model. It works by interleaving special, learnable "register tokens" into the input sequence. Each register token is then trained to predict a specific future token in the output. This approach is highly practical as it is compatible with off-the-shelf LLMs, introduces negligible parameter overhead, and remains aligned with the standard next-token pretraining objective.26  
* **Reward-Guided Speculative Decoding (RSD):** RSD 27 represents a significant conceptual shift. It challenges the dogma that speculative decoding must perfectly preserve the target model's output distribution. Instead, RSD incorporates a process reward model (PRM) to evaluate the quality of the draft model's intermediate outputs. If a draft is deemed "good enough" by the reward model, it can be accepted even if it's not what the target model would have generated. This introduces a controlled bias to prioritize high-reward outputs. The system dynamically mixes outputs from the draft and target models, using the fast draft model for high-reward (often easier) steps and reserving the powerful target model for "speculative correction" on low-reward (harder) steps. This optimizes the trade-off between computational cost and output quality in a more nuanced way than standard speculative decoding.27  
* **Hardware-Aware Parallelism:** Recognizing that decoding performance is tied to hardware, some methods explicitly co-design the algorithm and the system. PAPI 8 proposes a heterogeneous architecture with both standard accelerators and Processing-In-Memory (PIM) units, dynamically scheduling compute-bound and memory-bound decoding kernels to the most suitable hardware at runtime. Parallel Prompt Decoding (PPD) 10 uses a hardware-aware dynamic sparse tree to optimize its multi-token generation scheme for different GPUs, achieving significant speedups with minimal trainable parameters and memory overhead.

### **2.3 Analysis: The Speed, Quality, and Complexity Trade-off**

The primary motivation for MTP is clear and well-supported by empirical evidence.

* **Pros:** The most significant advantage is a substantial **inference speedup**, with reports of 2x to 4x improvements in tokens per second across various tasks and models.6 This directly translates to reduced latency and higher throughput, critical for real-world applications. Advanced methods like L-MTP also show potential for improving the model's ability to capture long-range dependencies.21  
* **Cons:** The benefits of MTP are not universal. The actual speedup is highly dependent on the specific task, the sampling temperature (higher temperatures reduce acceptance rates), the quality of the draft model, and the underlying hardware.7 Furthermore, MTP introduces additional complexity into the inference pipeline, requiring the management of one or more draft models and sophisticated verification logic.

Fundamentally, MTP in its most common form is a performance optimization, not a cognitive enhancement. The core mechanisms, particularly standard speculative decoding, are explicitly designed to produce an output distribution that is identical or nearly identical to the original single-token target model. The goal is to generate *more tokens per second*, not necessarily *better* or *more reasoned* tokens. The quality ceiling of the generated output is still defined by the capability of the base target model. While techniques like L-MTP and RSD begin to push this boundary by altering the training signal or introducing a reward-based bias, the primary function of the paradigm remains acceleration. This implies that while MTP is a crucial component for an efficient Chimera-1, it cannot be the sole solution for enhancing its core reasoning capabilities. It must be paired with other paradigms that address the cognitive myopia of autoregression.

### **2.4 Architectural Integration: Synergies with the Chimera-1 Hydra Layer**

The proposed "Hydra Layer" in the Chimera-1 blueprint is a conceptually perfect match for the MTP paradigm. This layer can be designed to be a flexible, powerful "speculation engine" that encapsulates the draft-then-verify logic. The evolution of MTP from architectural modifications to algorithmic overlays suggests that the Hydra Layer should be designed for flexibility.

* **Hydra as a Parallel Head Module:** In its simplest form, the Hydra Layer could be implemented as a set of parallel prediction heads, similar to the approaches used in Medusa or L-MTP.7 One head would predict the token at  
  t+1, while other heads would predict tokens at t+2,t+3,... (for standard MTP) or t+k,t+2k,... (for L-MTP). This would bake the MTP capability directly into the model's architecture.  
* **Hydra as an Integrated Speculation Engine:** A more sophisticated and flexible design would be to implement the Hydra Layer as a module that orchestrates the entire speculative decoding process. Within this design, one of the Hydra's "heads" could be a lightweight, distilled version of the main model, acting as an internal drafter. The other heads could then act in parallel to verify multiple candidate sequences generated by the drafter, akin to token tree verification.7 This would create a self-contained, highly parallel speculation engine within the main model, eliminating the need to manage a separate draft model during deployment.  
* **Hydra with Reward-Guided Control:** Drawing inspiration from RSD 27, the Hydra Layer could be designed to incorporate a connection to Chimera-1's Semantic Core or an external reward model. This would allow the layer to dynamically adjust its strategy. For routine, high-confidence generation, it could rely heavily on its internal drafter. When the reward model signals a complex or low-confidence step, the Hydra Layer could trigger a more comprehensive verification from the full model or even request a new plan from the Semantic Core, thus dynamically trading off speed for quality where it matters most. This trend of moving complexity from static hardware to dynamic algorithms suggests that designing the Hydra Layer as a flexible, algorithm-driven component, rather than a fixed set of heads, will allow Chimera-1 to more easily adapt to future advances in parallel decoding.

**Table 1: Comparative Overview of Multi-Token Prediction Techniques**

| Technique | Core Mechanism | Architectural Impact | Key Advantage | Key Limitation |
| :---- | :---- | :---- | :---- | :---- |
| **Parallel Heads** | Multiple output heads predict several future tokens in a single forward pass. | Requires architectural modification (adding heads) and specialized training. | Conceptually simple and highly parallel. | Quality degrades rapidly with prediction horizon due to compounding errors. |
| **Standard Speculative Decoding** | A small "draft" model generates candidate tokens; a large "target" model verifies them in parallel. | Algorithmic; can use off-the-shelf models. No architectural change needed. | Significant speedup (2-4x) while preserving the target model's output distribution.6 | Requires managing a separate, well-aligned draft model. Speedup is task/temperature dependent. |
| **Leap MTP (L-MTP)** | Predicts non-sequential tokens, skipping intermediate ones in a single pass.21 | Requires architectural modification (reassigning heads) and a two-stage training recipe. | Enhances long-range dependency capture by providing a broader training signal.21 | Inference is more complex, requiring reconstruction of the full sequence. |
| **Register-Based (MuToR)** | Interleaves learnable "register tokens" into the input, each tasked with predicting a future token.26 | Minimal impact; no architectural changes to the base model. Adds a small number of learnable parameters. | High compatibility with existing pretrained models; simple and effective. | The prediction horizon is tied to the number of register tokens introduced. |
| **Reward-Guided SD (RSD)** | Uses a reward model to accept "good enough" drafts, introducing a controlled bias for quality/cost optimization.27 | Algorithmic; requires a trained process reward model in addition to draft/target models. | Optimizes the cost-quality trade-off more effectively; can accept valuable drafts that standard SD would reject. | Moves away from strict distribution preservation; performance depends on reward model quality. |

## **3\. Paradigm II: Next-Byte Prediction — The Quest for a Universal, Tokenizer-Free Model**

While MTP seeks to accelerate the existing token-based paradigm, next-byte prediction challenges its very foundation. This approach argues that the "token" is an artificial and problematic abstraction. By operating on the most fundamental unit of digital information—the byte—a model can achieve true universality, robustness, and a deeper form of multimodality. This paradigm shift, however, comes at a significant computational cost that necessitates a co-design of the generative unit and the core model architecture.

### **3.1 The Tokenization Problem: Bias, Brittleness, and the Multimodal Lie**

Tokenizers, while a practical necessity for managing vocabulary size and sequence length in standard transformers, introduce a cascade of subtle but significant issues.

* **Tokenization Bias and Artifacts:** The process of splitting text into a fixed vocabulary of subwords is deterministic and lossy. This can lead to what is termed "tokenization bias," where the model's predictive distribution is skewed by the arbitrary boundaries of the tokenizer.18 For example, in code generation, if the correct completion  
  a is part of a larger token like class, a standard model may assign it zero probability if the prompt ends at cl. A byte-level model, which sees c and l as individual units, would not suffer from this blindness. Furthermore, tokenizers can assign positive probability to "noncanonical" token sequences that decode to valid text but never appear in the training data, wasting probability mass.19  
* **Brittleness vs. Universality and Robustness:** Token-based models are inherently brittle when faced with data outside their pre-defined vocabulary. They struggle with proper nouns, technical jargon, code, and even simple misspellings, often breaking them into meaningless sub-units. In contrast, byte-level models possess a fixed, universal vocabulary of 256 possible byte values.20 This makes them language-agnostic by default; a single model can process any language encoded in UTF-8 without specialized, language-specific tokenizers.16 This universality provides innate robustness to any novel sequence of characters, as it can be perfectly represented as a sequence of known bytes.  
* **The Multimodal Lie:** Perhaps the most profound limitation of tokenization is that it erects a barrier to true, end-to-end multimodality. Current "multimodal" models typically use separate, powerful encoders for each modality (e.g., a ViT for images, a text model for language) and learn a projection layer to align their representations.28 This is not a unified model but a committee of experts. Next-byte prediction offers a path to a fundamentally more unified approach. As demonstrated by bGPT 29, a model trained on next-byte prediction can learn to process  
  *any* digital data stream, from text files and audio waveforms to images and even the binary instructions of a CPU.29 By treating all information as a sequence of bytes, the model learns the underlying structure of digital information itself, rather than the structure of a single modality. This represents a paradigm shift from modeling language to modeling the "physics" of the digital world.

### **3.2 Core Mechanisms: Taming Sequence Length**

The primary and most formidable challenge of the byte-level paradigm is the explosion in sequence length. A document that is 1,000 tokens long could be 4,000-5,000 bytes long, making it computationally infeasible for a standard transformer architecture with its O(N2) attention complexity. Consequently, the success of byte-level models is inextricably linked to the development of architectures that can efficiently handle ultra-long sequences.

* **Hierarchical Architectures:** The key insight is to process information at multiple scales. A local model can handle fine-grained byte-level details, while a global model can reason about higher-level semantic patterns.  
  * **MegaByte and MambaByte:** These architectures exemplify this approach. MegaByte uses a large, global autoregressive transformer that operates on patches of bytes, and a smaller, local transformer to model the bytes within each patch. MambaByte replaces the transformer with the Mamba state-space model, which has linear-time complexity, making it even more efficient for byte-level modeling.1  
  * **Autoregressive U-Nets:** This architecture, proposed in "From Bytes to Ideas" 17, creates an explicit multi-scale representation. Its "contracting path" progressively pools raw bytes into representations of words, then word-pairs, and so on. This allows deeper layers to predict further into the future (e.g., the next few words) and focus on semantics. The "expanding path" then uses these high-level plans, combined with skip connections from the contracting path, to generate the final, detailed byte sequence. This design allows the model to handle fine syntactic details at shallow layers while being guided by broad semantic patterns from deeper layers.17  
  * **Patching-Based Models:** Architectures like SpaceByte 20 introduce a dynamic, content-aware form of patching. Instead of fixed-size patches, they apply the expensive global transformer blocks only at meaningful boundaries, such as after a space character. This aligns the processing of high-level information with natural word boundaries, significantly improving efficiency and performance over standard byte-level transformers.20

### **3.3 Analysis: Universality vs. the Computational Burden**

The decision to adopt a byte-level approach involves a fundamental architectural trade-off.

* **Pros:** The advantages are profound and foundational. The model gains a **truly universal vocabulary**, eliminating all tokenization-related artifacts and biases.18 It becomes inherently  
  **multilingual and robust** to any form of text.16 Most importantly, it enables the modeling of  
  **any digital data format** in a unified framework, paving the way for a deeper and more general form of artificial intelligence that can simulate the digital world.1 The data preprocessing pipeline is also radically simplified.  
* **Cons:** The primary disadvantage is the **massively increased sequence length**. This shifts the complexity from the external tokenizer into the core model architecture itself. Without specialized, long-context-efficient architectures, the training and inference costs of byte-level models are prohibitive.20 Information is also less compressed at the input level, meaning the model must learn to perform this compression itself, which may require more parameters or training data to achieve the same level of performance as a token-based model on a narrow task like English text generation.

### **3.4 Architectural Integration: The ALiBi-SSM Core as a Natural Enabler**

The choice of a byte-level generative unit is not made in a vacuum; it must be considered in concert with the core model architecture. The specified **ALiBi-SSM core** for Chimera-1 is not merely compatible with the byte-level paradigm; it is a powerful enabler that makes it uniquely viable and attractive.

This synergy arises because the State-Space Model (SSM) component, such as Mamba, directly and efficiently addresses the single greatest weakness of byte-level prediction. SSMs exhibit **linear-time complexity (O(N))** with respect to sequence length, in stark contrast to the quadratic complexity (O(N2)) of standard transformer attention. This means that the computational burden of processing sequences that are 4-5 times longer is manageable and scales gracefully. The MambaByte paper provides direct empirical evidence of this synergy, demonstrating that an SSM-based model can excel at byte-level language modeling, even outperforming subword models.1

The ALiBi (Attention with Linear Biases) component further strengthens this pairing. ALiBi introduces a simple, static bias to attention scores based on the distance between tokens. This inductive bias, which penalizes attention to distant tokens, helps the model maintain local coherence, a crucial property when the "tokens" are individual bytes and local context is extremely fine-grained.

Therefore, the decision to use an ALiBi-SSM core is a powerful architectural prior that transforms the byte-level paradigm from a computationally expensive curiosity into a practical and compelling foundation for Chimera-1. This co-design of the generative unit and the core architecture is a central finding of this analysis.

**Table 2: Byte-Level vs. Token-Level Models: A Fundamental Trade-off**

| Feature | Token-Level Approach | Byte-Level Approach |
| :---- | :---- | :---- |
| **Vocabulary** | Learned, large (32k-256k), and data-dependent (e.g., BPE, SentencePiece). | Fixed, universal (256 values), and data-agnostic. |
| **Robustness** | Brittle to out-of-vocabulary words, misspellings, and novel jargon. | Inherently robust; can represent any character sequence perfectly.20 |
| **Multilinguality** | Requires large, multilingual tokenizers or language-specific models. | Inherently multilingual; a single model can process any UTF-8 encoded language.16 |
| **Multimodality** | Requires separate encoders for each modality, which are then aligned. | Enables true, end-to-end multimodality by treating all digital data as a unified byte stream.29 |
| **Bias & Artifacts** | Suffers from tokenization bias, where segmentation can make valid outputs impossible.18 | Eliminates tokenization bias and artifacts entirely. |
| **Sequence Length** | Shorter, more compressed sequences. | Significantly longer sequences (typically 4-5x), less compressed information at input. |
| **Computational Core** | Standard Transformer attention (O(N2)) is feasible. | Standard attention is computationally infeasible. Requires long-context architectures like SSMs (O(N)) or hierarchical models.1 |

## **4\. Paradigm III: Next-"Chunk" Prediction — Generating Semantically Coherent Units**

Moving up the hierarchy of abstraction, the next-chunk prediction paradigm proposes elevating the generative unit from a syntactic fragment (a token or byte) to a semantically complete unit of meaning. This approach aligns the model's generative process more closely with human cognition, where we compose thoughts not word-by-word, but in concepts, phrases, and ideas.31 Instead of predicting the next word, a chunk-based model would predict the next

*paragraph*, the next *argument*, or the next *multimodal concept*. While this promises significant gains in coherence and reasoning, it introduces the formidable challenge of defining and managing a potentially infinite vocabulary of "chunks."

### **4.1 Defining the Vocabulary: A Taxonomy of Chunking Strategies for Generative Units**

The first critical step is to define what constitutes a "chunk." The extensive literature on chunking for Retrieval-Augmented Generation (RAG) provides a rich toolbox for this task.33 While RAG uses chunking as a preprocessing step to find the most relevant passage in an

*existing* corpus, we can adapt these techniques to define the units for a *generative* vocabulary. These strategies can be organized into a taxonomy of increasing sophistication.

* **Fixed-Size & Recursive Chunking:** The simplest methods involve splitting text into chunks of a fixed token length or using a recursive process with a set of separators to create roughly uniform chunks.33 While computationally cheap, these methods are naive and often break apart semantically coherent units, making them poor candidates for a generative vocabulary.  
* **Syntactic & Structural Chunking:** A more principled approach is to use the natural structure of a document. Chunks can be defined as sentences, paragraphs, or sections based on delimiters. For structured documents like Markdown or LaTeX files, the chunking can align with logical sections, headers, or list items.36 This preserves authorial intent and logical flow.  
* **Semantic Chunking:** This is a significant leap forward, using the meaning of the text to define boundaries. These methods typically involve encoding sentences into a vector space and then grouping them based on semantic similarity.36 Two main approaches exist:  
  * **Breakpoint-based:** This method scans through adjacent sentences and inserts a chunk boundary when the semantic distance between them exceeds a certain threshold, indicating a topic shift.33  
  * **Clustering-based:** This method takes a more global view, using clustering algorithms to group semantically related sentences together, even if they are not contiguous in the original text.33  
* **Layout-Aware Chunking (S2 Chunking):** For visually complex documents like PDFs or reports, text-only methods are insufficient. Layout-aware chunking, as proposed in S2 Chunking 42, is a hybrid approach. It constructs a graph where nodes represent text elements and edges are weighted by a combination of their semantic similarity (from text embeddings) and their spatial proximity (from bounding box information). Clustering this graph yields chunks that respect both the meaning and the visual layout of the document.  
* **Multimodal (Vision-Guided) Chunking:** This is the most advanced form of chunking, leveraging Large Multimodal Models (LMMs) to understand the document as a whole. The LMM processes pages as images, allowing it to interpret the structure, tables, figures, and their relationship to the surrounding text.44 This enables the creation of truly multimodal chunks that might, for example, group a paragraph of text with the specific figure it describes, forming a single, self-contained unit of multimodal meaning.

### **4.2 Core Mechanisms: From Retrieval-Augmentation to True Generative Chunking**

Defining the chunks is only half the problem. The core challenge is how a model can generate from a vocabulary that is potentially vast, dynamic, and sparse. A simple softmax over all possible chunks in a corpus is intractable. Recent research provides a viable mechanism.

* **Chunk-Distilled Language Modeling (CD-LM):** This framework 47 offers a concrete and powerful solution. Instead of generating a chunk from a fixed vocabulary, the model's task is simplified. At each step, a  
  **retrieval module** uses the current context embedding to query a large, external **datastore of chunks**. The most relevant chunk is retrieved and proposed to the main model. The model then makes a much simpler decision: whether to accept the proposed chunk or to reject it and generate a single token instead. This transforms the problem from "predicting one of millions of chunks" to a simple binary choice augmented by an efficient vector search. Crucially, the chunk datastore can be built from various sources—a powerful teacher model, the model's own high-confidence outputs (self-distillation), or external, expert-curated corpora. This allows for flexible and continuous adaptation to new knowledge without requiring expensive retraining of the core model.47  
* **Chunk AutoRegressive Modeling (CAR):** While developed for recommendation systems, the CAR model 48 provides a compelling conceptual parallel. It packs multiple related pieces of information (e.g., semantic attributes of an item and the user's action) into a single conceptual "chunk." The model then performs autoregression at this chunk level. This "act-with-think" paradigm, where a semantic unit is modeled holistically, is directly analogous to generating a complete idea rather than a sequence of words.48  
* **LLM-based Chunking:** The observation that LLMs themselves can be used to define chunking rules (e.g., by generating regular expressions 50) suggests a fascinating meta-level capability. A sufficiently advanced model could potentially learn to define its own optimal generative units based on the context, a form of learned, dynamic vocabulary creation.

### **4.3 Analysis: Semantic Cohesion vs. Vocabulary Explosion**

The chunk-based paradigm offers a tantalizing promise but also presents a significant technical hurdle.

* **Pros:** By generating in semantically coherent units, the model's outputs could exhibit far greater **long-range consistency and logical flow**. The generative process would more closely align with human conceptual composition, potentially leading to improved compositional reasoning.31 The CD-LM framework, in particular, allows for the  
  **direct and efficient injection of new knowledge** into the model's generative process by simply updating the external chunk datastore.47  
* **Cons:** The primary and seemingly prohibitive challenge is **vocabulary explosion**. The set of all possible meaningful chunks of text is effectively infinite. Even if limited to chunks from a large corpus, the vocabulary would be enormous and extremely sparse, making training a standard generative model with a final softmax layer impossible.

However, the CD-LM framework provides a powerful resolution to this problem. By reframing the task as retrieval-then-verification, it sidesteps the vocabulary explosion issue entirely. The "vocabulary" is no longer a fixed part of the model's architecture but a dynamic, external vector database of chunks. This database can be massive and can be updated, expanded, or swapped out without retraining the core LLM. This makes the chunk-based paradigm not only feasible but also highly scalable and adaptable.

### **4.4 Architectural Integration: A Natively Multimodal Vocabulary for Chimera-1's Sensory System**

The architecture of Chimera-1, with its specified **multimodal sensory system** and **joint embedding space**, is perfectly positioned to leverage a sophisticated, chunk-based generative unit. This paradigm allows for the creation of a system where multimodality is not an add-on but a native feature of the generative process itself.

The proposed integration is as follows:

1. **The Sensory System as a Multimodal Chunker:** The "multimodal sensory system" would be tasked with more than just encoding individual modalities. Its primary role would be to act as a **multimodal chunker**, using a vision-guided approach.44 It would process raw data streams (e.g., PDFs, web pages, code repositories with documentation) and create a datastore of  
   *multimodal chunks*. Each entry in this datastore would be a self-contained unit of meaning, such as a tuple of (text\_paragraph\_embedding, associated\_image\_embedding, structured\_table\_data).  
2. **The Joint Embedding Space as a Retrieval Space:** The "joint embedding space" would not just align different modalities but would be trained to produce a single, holistic vector representation for each *multimodal chunk*. This space becomes a rich semantic space optimized for *retrieval*.  
3. **Generation as Chunk Composition:** The core generative task for Chimera-1 would then become predicting the next multimodal chunk. Following the CD-LM mechanism, the model would use its current state to form a query into the vector database of multimodal chunks. Upon retrieving the best candidate chunk, the model would decide whether to generate it. This transforms the act of generation from stringing together words to composing a response from a palette of rich, multimodal concepts. This architecture makes Chimera-1 natively multimodal at its most fundamental generative level.

**Table 3: A Taxonomy of Chunking Strategies for Generative Vocabulary Definition**

| Chunking Strategy | Core Principle | Required Inputs | Ideal Use Case | Key Research |
| :---- | :---- | :---- | :---- | :---- |
| **Syntactic / Structural** | Segment text based on natural or formal boundaries like sentences, paragraphs, or Markdown headers. | Plain Text | Generating structured documents or text where sentence/paragraph integrity is paramount. | 36 |
| **Semantic (Breakpoint)** | Insert chunk boundaries where the semantic similarity between adjacent sentences drops, indicating a topic shift. | Text, Sentence Embeddings | Long-form narrative or argumentative text generation, ensuring topic coherence within chunks. | 33 |
| **Semantic (Clustering)** | Group semantically related sentences together globally, even if non-contiguous, using clustering algorithms. | Text, Sentence Embeddings | Generating topical summaries or knowledge syntheses where related facts should be grouped. | 33 |
| **Layout-Aware (S2)** | Combine semantic similarity with spatial proximity from bounding boxes in a graph-based clustering approach. | Text, Embeddings, Bounding Boxes | Generating text that respects the layout of complex documents like reports or scientific papers. | 42 |
| **Vision-Guided Multimodal** | Use an LMM to process document pages as images to understand and group related text, tables, and figures. | Images/PDFs of pages | Creating a truly multimodal generative vocabulary from rich documents for complex Q\&A or analysis. | 44 |

## **5\. Paradigm IV: Hierarchical & Multi-Scale Prediction — Planning for Coherent Reasoning**

The final paradigm, hierarchical and multi-scale prediction, is the most ambitious. It directly confronts the cognitive myopia of standard autoregressive models by attempting to build in mechanisms for long-horizon planning and structured reasoning. This approach seeks to endow LLMs with capabilities analogous to human "System 2" thinking—a slow, deliberate, and strategic process—as a complement to their existing fast, intuitive "System 1" pattern matching.11 The motivation is clear: despite their fluency, current LLMs often fail at tasks requiring robust planning, logical consistency, and adherence to constraints, a limitation that has spurred the development of what are being termed "Large Reasoning Models" (LRMs).53

### **5.1 The Need for Planning: Escaping Autoregressive Myopia with System 2 Capabilities**

A growing body of evidence demonstrates the planning deficiencies of standard LLMs. Systematic evaluations on benchmarks like PlanBench show that even state-of-the-art models struggle with classical planning problems that are trivial for dedicated symbolic solvers.14 They produce plans that are infeasible, suboptimal, or fail to generalize.14 This is a direct consequence of the next-token objective, which provides no explicit signal for foresight or goal-directed reasoning. The model learns to produce plausible continuations, not necessarily correct or coherent long-term solutions. The recent focus on LRMs, such as OpenAI's o1, represents a major industry effort to address this fundamental limitation by building models specifically constructed and trained for complex, multi-step reasoning.53 This paradigm aims to achieve that goal by making planning an explicit part of the generative process.

### **5.2 Core Mechanisms: Latent Plans, Multi-Scale Models, and Geometric Priors**

Approaches to integrating planning can be broadly categorized into those that modify the inference process and those that build planning into the model's architecture and training objective.

* **Explicit Planning Modules at Inference:** Systems like LEAP 56 augment the standard inference loop with an explicit planning mechanism. At each step of a reasoning process, the model "looks ahead" by simulating the future outcomes of several possible actions (e.g., which facts to select, what new fact to deduce). It scores these potential futures and chooses the action that leads to the most promising path. This turns a greedy, step-by-step process into a more strategic search, informed by foresight. These systems are often trained using reinforcement learning or contrastive methods to teach the model how to score good reasoning paths over bad ones.56  
* **Latent Plan Generation:** A more deeply integrated approach is to factorize the generation process itself into two stages: planning and realization. This is typically implemented using a hierarchical latent variable model.60  
  1. **Plan Generation:** A high-level **planner module** first generates an abstract, latent plan for the entire output. This plan can take many forms: a sequence of "writing actions" 11, a series of "anchor words" for key sentences 60, a logical skeleton for an argument 62, or a sequence of paragraph-level content plans.64  
  2. Text Realization: A lower-level decoder module then generates the final text, conditioned on the latent plan.  
     This architecture forces the model to first establish a global structure for its output before filling in the local details, dramatically improving coherence and consistency. The planning and generation can be interleaved, with the plan for the next section being informed by the text generated for the previous one, creating a dynamic feedback loop between high-level strategy and low-level execution.64  
* **Hierarchical Architectures and Geometric Priors:** This family of methods builds hierarchy directly into the model's structure or its representational geometry.  
  * **Hierarchical Models:** Models can be architected with explicit levels of scale. For instance, a hierarchical model might generate text at the word and character levels simultaneously, with the word-level model guiding the finer-grained character-level generation.65 Other approaches use hierarchical topic models to discover a tree of concepts from a corpus and then use this hierarchy to guide text generation, ensuring semantic organization at multiple levels of granularity.66  
  * **Geometric Priors (Hyperbolic Space):** A particularly elegant approach is to impose a hierarchical structure on the model's embedding space itself. Hyperbolic geometry, which has a greater capacity to embed tree-like structures with low distortion compared to Euclidean space, is a natural fit for this.68 The Hierarchy Transformer encoders (HiTs) model 68 achieves this by retraining a standard transformer encoder with additional hyperbolic clustering and centripetal losses. These losses force the model to organize its representations of concepts into a coherent hierarchy within a Poincaré ball, explicitly encoding relationships like "a beagle is a type of dog" into the geometry of the space. Other work has shown how hierarchical relationships can be encoded as orthogonal subspaces in a model's representation space.70

### **5.3 Analysis: Enhanced Reasoning vs. Architectural and Training Complexity**

The hierarchical planning paradigm directly targets the most significant weakness of current LLMs, but this comes at a cost.

* **Pros:** This approach offers a principled path toward **improving long-range coherence, planning, and robust reasoning**. By making the plan explicit, generation becomes more **interpretable and controllable**. A user could potentially inspect or even edit the latent plan to guide the final output. This paradigm has been shown to significantly improve performance on complex, multi-step reasoning tasks.57 A key benefit is that it provides a natural mechanism for  
  **variable compute time**. As proposed in 11, by sampling multiple plans from the planner and conditioning the generator on the resulting distribution of futures, the model can spend more computational effort on difficult problems by increasing the number of samples. This allows the model to "think" more when needed, a core feature attributed to advanced LRMs.53  
* **Cons:** The primary drawback is a significant increase in **architectural and training complexity**. These models often rely on complex latent variable frameworks that can be difficult to train and optimize. A major research challenge is defining the plan space itself—what are the right primitives for a "writing plan" or a "reasoning plan"? Furthermore, this paradigm requires a specialized training objective, a **"planning loss,"** to train the planner module, in addition to the standard generation loss.

### **5.4 Architectural Integration: The "Planning Loss" as a Training Objective for the Chimera-1 Semantic Core**

The specified **Semantic Core** of the Chimera-1 model is the ideal architectural locus for a hierarchical planning module. This component would not be a passive encoder but an active planner responsible for generating the high-level strategy that guides the entire generative process.

To make this concrete, a novel training objective for Chimera-1 can be formulated. The total loss function, Ltotal​, would be a weighted sum of a standard generation loss, Lgen​, and a dedicated **planning loss**, Lplan​:

Ltotal​=Lgen​+αLplan​

* Lgen​ would be the cross-entropy loss for predicting the chosen generative unit (e.g., the next byte or chunk), conditioned on both the context and the plan.  
* Lplan​ is the crucial new component. Its goal is to ensure the latent plan generated by the Semantic Core is meaningful and predictive of the final output's structure. Following the framework of "Learning to Plan Long-Term for Language Modeling" 11, this can be implemented by:  
  1. **Inferring Abstract Plans:** First, an offline process analyzes a large text corpus to infer sequences of abstract "writing actions" (e.g., introduce\_topic, provide\_example, conclude\_argument).  
  2. **Training the Planner:** The Semantic Core is then trained, via Lplan​, to predict these action sequences from the text context.  
  3. **Training the Generator:** The main Chimera-1 generator is trained, via Lgen​, to produce its output (bytes or chunks) conditioned on the plan predicted by the Semantic Core.

This makes the abstract plan an explicit, supervised part of the end-to-end training process.

Furthermore, drawing from the HiTs paper 68, the internal embedding space of the Semantic Core itself could be structured with a hyperbolic clustering loss during pre-training. This would force the Core to not only learn to plan but to organize its world knowledge into a coherent, navigable hierarchy, providing a powerful inductive bias for reasoning.

## **6\. Synthesis & Prescriptive Recommendation for Chimera-1**

The preceding analysis has examined four distinct paradigms for generative units, each with unique strengths, weaknesses, and architectural implications. Multi-Token Prediction offers speed; Next-Byte Prediction offers universality; Next-Chunk Prediction offers semantic cohesion; and Hierarchical Prediction offers reasoning. A critical realization is that these paradigms are not mutually exclusive competitors. Instead, they can be viewed as layers in a cognitive stack, operating at different levels of abstraction. This perspective allows for the design of a novel, hybrid architecture for Chimera-1 that synergistically combines the advantages of all four.

### **6.1 Recapitulation: The Four Paradigms as a Cognitive Stack**

The four paradigms can be organized into a conceptual hierarchy that mirrors a plausible model of intelligent generation:

1. **Cognitive Layer (The "Why"): Hierarchical Prediction.** At the highest level of abstraction, the model must decide *what* it wants to say. This is the domain of planning, strategy, and reasoning. The Hierarchical Prediction paradigm, with its explicit or latent plans, operates here, generating a high-level goal or a sequence of communicative intents.  
2. **Semantic Layer (The "What"): Next-Chunk Prediction.** Once a high-level goal is established, the model must select a concrete concept or unit of meaning to express it. This is the domain of the Next-Chunk Prediction paradigm. The model selects a semantically or multimodally coherent chunk that instantiates the current step of the plan.  
3. **Expression Layer (The "How"): Multi-Token Prediction.** With a specific chunk of content selected, the model must articulate it efficiently. This is the domain of Multi-Token Prediction. It accelerates the rendering of the chunk's textual component into a sequence of tokens, optimizing for speed and throughput.  
4. **Physical Layer (The "Medium"): Next-Byte Prediction.** At the most fundamental level, all generated content must be represented as raw data. This is the domain of Next-Byte Prediction. It provides the universal, tokenizer-free foundation upon which any modality—text, image, audio, or code—can be constructed.

This layered view provides the blueprint for a composite model that does not have to choose between these paradigms but can instead integrate them into a single, multi-level generative process.

### **6.2 A Hybrid Generative Architecture for Chimera-1**

Based on this synthesis, the following prescriptive recommendation is made for the Chimera-1 architecture. It is a hybrid model that leverages a different generative paradigm at each layer of its operation, mapping directly onto the specified components of the Chimera-1 blueprint.

* **Foundation (Physical Layer): Next-Byte Prediction.** Chimera-1 should be a **tokenizer-free, byte-level model**. This is the most foundational recommendation. This choice provides true universality, eliminates all tokenization artifacts, and enables a profoundly deep and unified form of multimodality. This is made practical and efficient by the specified **ALiBi-SSM core**, which is perfectly suited to handle the long sequences inherent to byte-level processing.1  
* **Core Generative Unit (Semantic Layer): Next Multimodal Chunk.** The model's primary generative task should not be to predict the next byte, but to predict the **next multimodal chunk**. This is achieved using a retrieval-based mechanism inspired by CD-LM.47 The  
  **multimodal sensory system** is responsible for processing diverse data sources and creating a massive, external vector database of multimodal chunks (e.g., (text, image, data) tuples). The **joint embedding space** is used to represent these chunks for efficient retrieval. The model's core task is to use its context to query this database and select the next chunk to generate.  
* **High-Level Control (Cognitive Layer): Hierarchical Planning.** The **Semantic Core** should be implemented as an active **planner**. It is trained with a dedicated **planning loss** to generate a sequence of latent, abstract plan steps (e.g., \[identify\_premise, draw\_inference, state\_conclusion\]). This latent plan guides the chunk retrieval process at the semantic layer, ensuring the generated output is globally coherent and goal-directed. This layer incorporates the variable-compute mechanism, allowing the model to sample more plans for more difficult problems, enabling it to "think" longer when necessary.11  
* **Inference Acceleration (Expression Layer): Multi-Token Prediction.** The **Hydra Layer** is responsible for efficiently rendering the selected chunks. For the textual component of a retrieved multimodal chunk, the Hydra Layer employs a state-of-the-art speculative decoding algorithm (such as RSD or L-MTP) to generate the corresponding byte sequence with maximum parallelism and speed.21 Non-textual components of the chunk (e.g., an image embedding) are passed through directly.

### **6.3 Training and Inference Workflow**

This hybrid architecture implies a sophisticated, multi-stage training and inference process.

Training Workflow:  
The model is trained end-to-end with a composite loss function:  
Ltotal​=w1​Lplan​+w2​Lchunk​+w3​Lbyte​

* **Lplan​ (Planning Loss):** Trains the Semantic Core to predict abstract plan sequences inferred from the training corpus, as described in Section 5.4.  
* **Lchunk​ (Chunk Loss):** A retrieval-based loss that trains the model to query and select the correct ground-truth chunk from the datastore given the context and the ground-truth plan. This could be a contrastive loss that pulls the context embedding closer to the correct chunk embedding and pushes it away from others.  
* **Lbyte​ (Generation Loss):** A standard cross-entropy loss on the byte-level representation of the textual part of the ground-truth chunk, conditioned on the context and the chunk's identity.

Inference Workflow:  
At each generative step, the model executes the following sequence:

1. **Plan:** The Semantic Core observes the current context and generates the next latent step in its high-level plan.  
2. **Retrieve:** The model uses the current context and the new plan step to form a query to the external multimodal chunk database. The top-k most relevant chunks are retrieved.  
3. **Select & Generate:** The model scores the retrieved chunks and selects the best one to generate next.  
4. **Accelerate:** The Hydra Layer takes the textual component of the selected chunk and uses speculative decoding to rapidly generate the full byte sequence. The non-textual components (e.g., image data) are decoded via their respective pathways.  
5. **Update:** The generated chunk is added to the context, and the process repeats.

### **6.4 Concluding Remarks**

The standard next-token prediction paradigm has served the field well, but it is a developmental stage, not an endpoint. To build models that are not only fluent but also efficient, robust, truly multimodal, and capable of sophisticated reasoning, we must fundamentally rethink the unit of generation.

The proposed hybrid architecture for Chimera-1 represents a principled synthesis of the four leading alternative paradigms. It is not a compromise but a composition, leveraging each paradigm for the task to which it is best suited. By building on a universal **byte-level** foundation, composing with coherent **multimodal chunks**, guiding generation with a **hierarchical planner**, and accelerating output with **multi-token prediction**, this architecture provides a clear and ambitious path forward. It is a design that directly addresses the deepest limitations of current models and is tailored to exploit the specified strengths of the Chimera-1 blueprint. This approach offers the most promising path to creating a model that "punches ludicrously above its weight," achieving a new level of synergy between generative efficiency and cognitive capability.

#### **Works cited**

1. Beyond Language Models: Byte Models are Digital World Simulators \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2402.19155v1](https://arxiv.org/html/2402.19155v1)  
2. Scaling Language Models: Methods, Analysis & Insights from Training Gopher \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2112.11446](https://arxiv.org/abs/2112.11446)  
3. Pretrained Language Model for Text Generation: A Survey \- IJCAI, accessed July 4, 2025, [https://www.ijcai.org/proceedings/2021/0612.pdf](https://www.ijcai.org/proceedings/2021/0612.pdf)  
4. Pre-trained Language Models for Text Generation: A Survey \- arXiv, accessed July 4, 2025, [https://arxiv.org/pdf/2201.05273](https://arxiv.org/pdf/2201.05273)  
5. \[2506.00413\] Accelerating Diffusion LLMs via Adaptive Parallel Decoding \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2506.00413](https://arxiv.org/abs/2506.00413)  
6. \[2503.00491\] Tutorial Proposal: Speculative Decoding for Efficient LLM Inference \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2503.00491](https://arxiv.org/abs/2503.00491)  
7. Unlocking Efficiency in Large Language Model Inference: A ..., accessed July 4, 2025, [https://arxiv.org/abs/2401.07851](https://arxiv.org/abs/2401.07851)  
8. \[2502.15470\] PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2502.15470](https://arxiv.org/abs/2502.15470)  
9. \[2502.11517\] Learning to Keep a Promise: Scaling Language Model Decoding Parallelism with Learned Asynchronous Decoding \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2502.11517](https://arxiv.org/abs/2502.11517)  
10. \[2405.18628\] Hardware-Aware Parallel Prompt Decoding for Memory-Efficient Acceleration of LLM Inference \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2405.18628](https://arxiv.org/abs/2405.18628)  
11. arxiv.org, accessed July 4, 2025, [https://arxiv.org/html/2409.00070v1](https://arxiv.org/html/2409.00070v1)  
12. arXiv:2409.00070v1 \[cs.CL\] 23 Aug 2024, accessed July 4, 2025, [https://arxiv.org/pdf/2409.00070?](https://arxiv.org/pdf/2409.00070)  
13. A Survey on Large Language Models for Automated Planning \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2502.12435v1](https://arxiv.org/html/2502.12435v1)  
14. On The Planning Abilities of OpenAI's o1 Models: Feasibility, Optimality, and Generalizability \- OpenReview, accessed July 4, 2025, [https://openreview.net/pdf?id=qgvQx30Z0R](https://openreview.net/pdf?id=qgvQx30Z0R)  
15. L2P: A Python Toolkit for Automated PDDL Model Generation with Large Language Models, accessed July 4, 2025, [https://plan-fm.github.io/Paper\_L2P.pdf](https://plan-fm.github.io/Paper_L2P.pdf)  
16. What is Byte-Level Language Models \- Activeloop, accessed July 4, 2025, [https://www.activeloop.ai/resources/glossary/byte-level-language-models/](https://www.activeloop.ai/resources/glossary/byte-level-language-models/)  
17. From Bytes to Ideas: Language Modeling with Autoregressive ... \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2506.14761](https://arxiv.org/abs/2506.14761)  
18. arXiv:2410.09303v2 \[cs.CL\] 11 Apr 2025, accessed July 4, 2025, [https://www.arxiv.org/pdf/2410.09303](https://www.arxiv.org/pdf/2410.09303)  
19. \[2506.07956\] Language Models over Canonical Byte-Pair Encodings \- arXiv, accessed July 4, 2025, [http://www.arxiv.org/abs/2506.07956](http://www.arxiv.org/abs/2506.07956)  
20. A Comparative Analysis of Byte-Level and Token-Level Transformer ..., accessed July 4, 2025, [https://gregrobison.medium.com/a-comparative-analysis-of-byte-level-and-token-level-transformer-models-in-natural-language-9fb4331b6acc](https://gregrobison.medium.com/a-comparative-analysis-of-byte-level-and-token-level-transformer-models-in-natural-language-9fb4331b6acc)  
21. L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models, accessed July 4, 2025, [https://arxiv.org/html/2505.17505v1](https://arxiv.org/html/2505.17505v1)  
22. Why would anyone let LLMs predict 4 tokens at once? \- YouTube, accessed July 4, 2025, [https://www.youtube.com/watch?v=4BhZZYg2\_J4](https://www.youtube.com/watch?v=4BhZZYg2_J4)  
23. \[2402.01528\] Decoding Speculative Decoding \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2402.01528](https://arxiv.org/abs/2402.01528)  
24. \[2505.17505\] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models \- arXiv, accessed July 4, 2025, [https://www.arxiv.org/abs/2505.17505](https://www.arxiv.org/abs/2505.17505)  
25. L-MTP: Leap Multi-Token Prediction Beyond Adjacent ... \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2505.17505](https://arxiv.org/abs/2505.17505)  
26. Multi-Token Prediction Needs Registers \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2505.10518](https://arxiv.org/abs/2505.10518)  
27. Reward-Guided Speculative Decoding for Efficient LLM ... \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2501.19324](https://arxiv.org/abs/2501.19324)  
28. Multi-Modal Generative AI: Multi-modal LLM, Diffusion and Beyond \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2409.14993v1](https://arxiv.org/html/2409.14993v1)  
29. Beyond Language Models: Byte Models are Digital World Simulators, accessed July 4, 2025, [https://arxiv.org/abs/2402.19155](https://arxiv.org/abs/2402.19155)  
30. Paper page \- Beyond Language Models: Byte Models are Digital World Simulators, accessed July 4, 2025, [https://huggingface.co/papers/2402.19155](https://huggingface.co/papers/2402.19155)  
31. Ten ways to learn new words as a language learner | British Council, accessed July 4, 2025, [https://www.britishcouncil.org/voices-magazine/ten-ways-learn-new-words-language-learner](https://www.britishcouncil.org/voices-magazine/ten-ways-learn-new-words-language-learner)  
32. Learning language in chunks \- Cambridge University Press & Assessment, accessed July 4, 2025, [https://www.cambridge.org/elt/blog/wp-content/uploads/2019/10/Learning-Language-in-Chunks.pdf](https://www.cambridge.org/elt/blog/wp-content/uploads/2019/10/Learning-Language-in-Chunks.pdf)  
33. arXiv:2410.13070v1 \[cs.CL\] 16 Oct 2024, accessed July 4, 2025, [https://arxiv.org/pdf/2410.13070](https://arxiv.org/pdf/2410.13070)  
34. Chunking in LLMs (Large Language Models) | by Elif Beyza Tok | Medium, accessed July 4, 2025, [https://medium.com/@elifbeyzatok/chunking-in-llms-large-language-models-450687c4378a](https://medium.com/@elifbeyzatok/chunking-in-llms-large-language-models-450687c4378a)  
35. Mastering Chunking Strategies for RAG: Best Practices & Code Examples \- Databricks Community, accessed July 4, 2025, [https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)  
36. Semantic Chunking for RAG. What is Chunking ? | by Plaban Nayak | The AI Forum, accessed July 4, 2025, [https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5](https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5)  
37. Chunking Strategies for LLM Applications \- Pinecone, accessed July 4, 2025, [https://www.pinecone.io/learn/chunking-strategies/](https://www.pinecone.io/learn/chunking-strategies/)  
38. 15 Chunking Techniques to Build Exceptional RAGs Systems \- Analytics Vidhya, accessed July 4, 2025, [https://www.analyticsvidhya.com/blog/2024/10/chunking-techniques-to-build-exceptional-rag-systems/](https://www.analyticsvidhya.com/blog/2024/10/chunking-techniques-to-build-exceptional-rag-systems/)  
39. Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2506.00773v1](https://arxiv.org/html/2506.00773v1)  
40. Fixed-size, Semantic and Recursive Chunking Strategies for LLMs \- Langformers Blog, accessed July 4, 2025, [https://blog.langformers.com/llm-chunking-strategies/](https://blog.langformers.com/llm-chunking-strategies/)  
41. Is Semantic Chunking Worth the Computational Cost? \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2410.13070v1](https://arxiv.org/html/2410.13070v1)  
42. S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2501.05485v1](https://arxiv.org/html/2501.05485v1)  
43. \[2501.05485\] S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2501.05485](https://arxiv.org/abs/2501.05485)  
44. \[2506.16035\] Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding \- arXiv, accessed July 4, 2025, [http://www.arxiv.org/abs/2506.16035](http://www.arxiv.org/abs/2506.16035)  
45. Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2506.16035](https://arxiv.org/html/2506.16035)  
46. Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding \- arXiv, accessed July 4, 2025, [https://arxiv.org/pdf/2506.16035](https://arxiv.org/pdf/2506.16035)  
47. Chunk-Distilled Language Modeling, accessed July 4, 2025, [https://arxiv.org/abs/2501.00343](https://arxiv.org/abs/2501.00343)  
48. arxiv.org, accessed July 4, 2025, [https://arxiv.org/html/2506.23643](https://arxiv.org/html/2506.23643)  
49. \[2506.23643\] Act-With-Think: Chunk Auto-Regressive Modeling for Generative Recommendation \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2506.23643](https://arxiv.org/abs/2506.23643)  
50. \[2503.09600\] MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2503.09600](https://arxiv.org/abs/2503.09600)  
51. MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System, accessed July 4, 2025, [https://arxiv.org/html/2503.09600v1](https://arxiv.org/html/2503.09600v1)  
52. arXiv:2503.09600v1 \[cs.CL\] 12 Mar 2025, accessed July 4, 2025, [https://arxiv.org/pdf/2503.09600](https://arxiv.org/pdf/2503.09600)  
53. arxiv.org, accessed July 4, 2025, [https://arxiv.org/abs/2409.13373](https://arxiv.org/abs/2409.13373)  
54. LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench \- OpenReview, accessed July 4, 2025, [https://openreview.net/pdf/d4306e0aa51bd78d93a99139b0aa391aee50a66f.pdf](https://openreview.net/pdf/d4306e0aa51bd78d93a99139b0aa391aee50a66f.pdf)  
55. Revealing the Barriers of Language Agents in Planning \- ACL Anthology, accessed July 4, 2025, [https://aclanthology.org/2025.naacl-long.93.pdf](https://aclanthology.org/2025.naacl-long.93.pdf)  
56. Explicit Planning Helps Language Models in Logical Reasoning \- OpenReview, accessed July 4, 2025, [https://openreview.net/forum?id=Jk6LA0NGOU¬eId=Nx2W4ShVJk](https://openreview.net/forum?id=Jk6LA0NGOU&noteId=Nx2W4ShVJk)  
57. Explicit Planning Helps Language Models in Logical Reasoning \- ACL Anthology, accessed July 4, 2025, [https://aclanthology.org/2023.emnlp-main.688/](https://aclanthology.org/2023.emnlp-main.688/)  
58. \[2303.15714\] Explicit Planning Helps Language Models in Logical Reasoning \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2303.15714](https://arxiv.org/abs/2303.15714)  
59. Explicit Planning Helps Language Models in Logical Reasoning \- ACL Anthology, accessed July 4, 2025, [https://aclanthology.org/anthology-files/pdf/emnlp/2023.emnlp-main.688.pdf](https://aclanthology.org/anthology-files/pdf/emnlp/2023.emnlp-main.688.pdf)  
60. Narrative Text Generation with a Latent Discrete Plan \- ACL Anthology, accessed July 4, 2025, [https://aclanthology.org/2020.findings-emnlp.325.pdf](https://aclanthology.org/2020.findings-emnlp.325.pdf)  
61. arXiv:2010.03272v1 \[cs.CL\] 7 Oct 2020, accessed July 4, 2025, [https://arxiv.org/pdf/2010.03272](https://arxiv.org/pdf/2010.03272)  
62. Planning with Logical Graph-based Language Model for Instruction Generation \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2308.13782v2](https://arxiv.org/html/2308.13782v2)  
63. Planning with Logical Graph-based Language Model for Instruction Generation \- arXiv, accessed July 4, 2025, [https://arxiv.org/pdf/2308.13782](https://arxiv.org/pdf/2308.13782)  
64. Data-to-text Generation with Variational Sequential Planning \- ACL Anthology, accessed July 4, 2025, [https://aclanthology.org/2022.tacl-1.40.pdf](https://aclanthology.org/2022.tacl-1.40.pdf)  
65. Generation of Hip-Hop Lyrics with Hierarchical Modeling and Conditional Templates \- ACL Anthology, accessed July 4, 2025, [https://aclanthology.org/W19-8638.pdf](https://aclanthology.org/W19-8638.pdf)  
66. On the Affinity, Rationality, and Diversity of Hierarchical Topic Modeling \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2401.14113v2](https://arxiv.org/html/2401.14113v2)  
67. Hierarchical Concept-Driven Language Model \- ResearchGate, accessed July 4, 2025, [https://www.researchgate.net/publication/351725917\_Hierarchical\_Concept-Driven\_Language\_Model](https://www.researchgate.net/publication/351725917_Hierarchical_Concept-Driven_Language_Model)  
68. arxiv.org, accessed July 4, 2025, [https://arxiv.org/abs/2401.11374](https://arxiv.org/abs/2401.11374)  
69. Language Models as Hierarchy Encoders \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2401.11374v1](https://arxiv.org/html/2401.11374v1)  
70. arXiv:2406.01506v3 \[cs.CL\] 18 Feb 2025, accessed July 4, 2025, [https://arxiv.org/pdf/2406.01506?](https://arxiv.org/pdf/2406.01506)  
71. The Geometry of Meaning: Perfect Spacetime Representations of Hierarchical Structures arXiv:2505.08795v1 \[cs.LG\] 7 May 2025, accessed July 4, 2025, [https://www.arxiv.org/pdf/2505.08795](https://www.arxiv.org/pdf/2505.08795)