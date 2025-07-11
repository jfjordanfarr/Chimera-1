

# **Architectural Blueprint for the Chimera-1 Semantic Core**

## **Introduction**

This report presents the complete architectural and training blueprint for the Semantic Core of the Chimera-1 model. The Semantic Core is the foundational component of the system, a unified, high-dimensional embedding space designed to co-represent textual and visual information. The primary objective is to engineer a semantic space that is simultaneously multimodal, specialized, and efficient. This presents a tripartite challenge: the space must seamlessly integrate vision and text to enable cross-modal reasoning; it must be deeply specialized in the complex domains of bioinformatics, legal text, and source code; and it must be representationally efficient to support advanced retrieval and generative capabilities over million-token contexts, a key design constraint for the Chimera-1 project.

The design of this Semantic Core is paramount, as it will serve as the shared conceptual vocabulary for all higher-order functions of the Chimera-1 model. The quality of this space—its structure, richness, and efficiency—will directly dictate the model's ultimate performance on tasks requiring nuanced understanding and generation.

This document provides a prescriptive and technically exhaustive strategy for creating and training this joint embedding space. The analysis proceeds in a logical sequence, beginning with the foundational decision of whether to build the space from scratch or adapt a pre-existing model. It then details the incorporation of advanced representation techniques essential for efficiency at scale, followed by a robust methodology for domain-specific adaptation. Finally, it specifies the optimal training objectives required to forge a powerful and coherent multimodal space. The report culminates in a final, synthesized blueprint and an actionable implementation roadmap, providing a clear path forward for the development of the Chimera-1 Semantic Core.

## **Section 1: Foundational Strategy — A Hybrid Approach to Joint Embedding**

The initial and most fundamental decision in designing the Semantic Core is the choice of a foundational model. This choice dictates the starting point for our semantic space, influencing everything from training cost and time to the final model's inherent capabilities. A purely from-scratch approach, while offering maximum theoretical flexibility, is overwhelmingly impractical and suboptimal in the current landscape of foundation models. Instead, a hybrid strategy that leverages the immense, generalized knowledge of a state-of-the-art pre-trained model and intelligently adapts it for our specific multimodal and domain-specific needs is the most pragmatic and powerful path forward.

### **1.1. Analysis: The Trade-off Between Training from Scratch and Fine-tuning**

The development of large-scale AI models has converged on a paradigm that heavily favors transfer learning over training from scratch. The rationale is supported by overwhelming evidence regarding computational cost, resource allocation, and the inheritance of learned knowledge.

Training a large language or multimodal model from the ground up is a computationally prohibitive endeavor, requiring vast quantities of data and GPU-hours that are often beyond the scope of even well-resourced projects.1 This "cold start" problem is not merely a matter of expense; it represents a significant strategic disadvantage. Foundation models pre-trained on web-scale datasets have already encoded a massive and diverse body of general semantic and syntactic knowledge about the world.2 Attempting to replicate this learning process from scratch is not only inefficient but also risks failing to capture the breadth of understanding that these base models provide. The industry standard, therefore, is to fine-tune or perform continual pre-training on existing foundation models, a method that is demonstrably faster, more cost-effective, and allows for the tailoring of a general-purpose AI to specific business or research needs.3

Multimodal Large Language Models (MLLMs) that are built upon pre-trained unimodal encoders—such as a text encoder and a vision encoder—inherit the powerful cognitive capabilities of these components.2 This circumvents the need to relearn fundamental concepts like language grammar or basic visual features. The modern and most effective approach involves selecting a powerful, permissively licensed open-source model and adapting it to the target use case.4 This strategy leverages the billions of dollars in research and computation already invested in these base models, allowing developers to focus their resources on the more targeted and valuable tasks of specialization and alignment.

### **1.2. Recommendation: A Staged Continual Pre-training and Alignment Approach**

Given the clear advantages of transfer learning, the recommended strategy for Chimera-1 is a staged process of continual pre-training and cross-modal alignment. This approach avoids the immense costs and risks of training from scratch while providing a structured path to creating a bespoke, high-performance semantic space.

The process begins by selecting a state-of-the-art pre-trained text embedding model to serve as the semantic "rootstock." The subsequent stage involves aligning the visual tokens, generated by the Chimera-1 Vision System, with this pre-existing text embedding space. This is a common and highly effective strategy in modern MLLM development.7 The alignment is typically achieved by training a lightweight projection module—often a simple Multi-Layer Perceptron (MLP)—that learns to map the image embeddings from the vision encoder into the latent space of the language model.9 This projection layer acts as a bridge, translating visual concepts into the "language" of the text embedding space. More complex architectures may employ cross-attention mechanisms to allow for deeper fusion between modalities.11

This methodology can be conceptualized not as a simple "injection" of visual tokens, but as a more deliberate "Semantic Grafting." The pre-trained text model provides a robust and well-structured semantic foundation, complete with nuanced relationships between concepts learned from vast data. The task then becomes to train the vision system (the vision encoder and the projection layer) to map its visual representations onto the *correct, existing* semantic coordinates within that space. This is significantly more efficient and stable than training a joint space from scratch, which can suffer from instability or "catastrophic forgetting".1 The success of prominent MLLMs like LLaVA, which projects visual features into the embedding space of a frozen Large Language Model (LLM), provides strong validation for this principle.10

To maximize the effectiveness of this grafting process, the initial training phase will freeze the parameters of the base text embedding model. Training will focus exclusively on the vision encoder and the new projection layer. This isolates the training objective to the cross-modal alignment task, preventing the text model's carefully calibrated semantic space from being perturbed or degraded. Once a strong alignment is achieved, the text model can be unfrozen for subsequent end-to-end fine-tuning.

### **1.3. Base Model Selection: mixedbread-ai/mxbai-embed-large-v1**

The choice of the foundational text embedding model is critical. It must be a top-performing, open-source model with a permissive license that allows for commercial use and modification. Based on a thorough review of the current state-of-the-art, the recommended base model for the Chimera-1 Semantic Core is mixedbread-ai/mxbai-embed-large-v1.

This model consistently ranks at the top of the Massive Text Embedding Benchmark (MTEB), a comprehensive benchmark for evaluating embedding models.13 It demonstrates state-of-the-art performance, outperforming other strong open-source models like

BAAI/bge-large-en-v1.5 and even proprietary offerings such as OpenAI's text-embedding-3-large.14 Its high performance is attributable to its training on a massive, high-quality dataset of over 700 million text pairs and 30 million high-quality triplets, using the advanced AnglE (Angle-optimized) loss function.16

Crucially for the Chimera-1 project, mxbai-embed-large-v1 is released under an Apache-2.0 license, which is a permissive license suitable for both research and commercial applications.16 Architecturally, it is a

BertModel (BERT-large) with 24 hidden layers, producing a 1024-dimensional embedding vector from a maximum context length of 512 tokens.18 While the initial context length is short, this BERT-style architecture is well-suited for context extension techniques that will be applied during domain adaptation.

The selection of mxbai-embed-large-v1 extends beyond its benchmark scores and license. Its architecture exhibits a powerful synergy with the other advanced techniques proposed for the Chimera-1 Semantic Core.

1. **Contrastive Learning Compatibility:** The model's underlying BertModel architecture is a dual-encoder by nature, which is the ideal structure for the contrastive learning objectives, specifically Triple Contrastive Learning, that will be used to train the final embedding space (detailed in Section 4).  
2. **Matryoshka Representation Learning (MRL) Support:** The model was explicitly designed with support for MRL, a technique for creating variable-sized embeddings that is a cornerstone of our efficiency strategy for handling million-token contexts (detailed in Section 2).14  
3. **Context Length Extensibility:** BERT-style architectures are the primary targets for proven context extension methods like LongLoRA, which will be employed during the domain adaptation phase to prepare the model for long-sequence tasks.20

This deliberate selection of a foundation model that is architecturally aligned with the entire proposed technology stack significantly de-risks the project. It ensures a seamless integration of components and provides a robust, high-performance starting point for building the Chimera-1 Semantic Core.

## **Section 2: Advanced Representation Structure for Efficiency and Retrieval**

To meet the demanding requirement of operating within a million-token context, the Chimera-1 Semantic Core cannot rely on a simple, monolithic embedding structure. The computational cost of performing dense vector search over such a vast space would be prohibitive. Therefore, the core's architecture must incorporate state-of-the-art representation techniques designed for efficiency, scalability, and enhanced retrieval performance. The proposed solution is a hybrid dense-sparse representation, leveraging Matryoshka Representation Learning for adaptive dense embeddings and SPLADE for precise lexical embeddings.

### **2.1. Matryoshka Representation Learning (MRL): The Core of Adaptive Efficiency**

Matryoshka Representation Learning (MRL) is a transformative technique for training a single embedding model to produce nested, adaptive-size representations.22 Inspired by Russian Matryoshka dolls, where smaller dolls are nested within larger ones, MRL trains a model such that any prefix of the full embedding vector is also a high-quality, lower-dimensional representation of the same concept.24

This is achieved by modifying the training objective. Instead of optimizing a single loss function for the full-dimension embedding (e.g., 1024-dim), MRL optimizes a weighted sum of losses for a predefined set of nested dimensions, or a nesting\_list (e.g., \`\`).25 This forces the model to encode the most critical, high-level semantic information in the earliest dimensions of the vector, with subsequent dimensions adding progressively finer-grained detail.22

The primary benefit of MRL is the ability to dynamically trade a small amount of performance for significant gains in computational efficiency. Developers can truncate a full-sized embedding to a much smaller dimension by simply taking a prefix of the vector, without needing to retrain or run a separate model.28 For example, OpenAI's

text-embedding-3-large model, which uses MRL, can be shortened from 3072 dimensions down to 256 while still outperforming their older, full-sized 1536-dimension ada-002 model on benchmarks.30 This flexibility is critical for resource-constrained applications and large-scale retrieval systems. MRL has been proven effective across modalities, including vision (ResNet, ViT), text (BERT), and multimodal (ALIGN) models, making it a robust choice for Chimera-1.22

The million-token context planned for Chimera-1 is not a monolithic block to be searched via brute force. It is a vast semantic landscape that necessitates a sophisticated, multi-stage retrieval strategy. MRL is the core enabling technology for what can be termed "Funnel Retrieval." This strategy addresses the computational challenge of searching long contexts by creating a multi-step process:

1. **Stage 1: Broad Sweep Retrieval.** For an initial pass over the entire million-token context, a highly truncated, low-dimensional MRL embedding (e.g., 128-dim) will be used. This allows for a massively parallel and computationally inexpensive Approximate Nearest Neighbor (ANN) search to quickly identify a large set of potentially relevant document chunks.  
2. **Stage 2: High-Fidelity Re-ranking.** The top-k candidates (e.g., the top 1,000 chunks) identified in the first stage are then re-ranked using a much larger, more accurate MRL embedding (e.g., the full 1024-dim vector). This re-ranking step is performed on a much smaller subset of the data, making it computationally feasible.

This "funnel" approach is the only practical way to perform dense vector retrieval at the scale of a million tokens. It elegantly sidesteps the prohibitive cost of a full-dimensional search across the entire context window, directly satisfying the efficiency constraints of the project.

### **2.2. Sparse Lexical Embeddings (SPLADE): A Parallel Path to Precision**

While dense embeddings excel at capturing semantic similarity, they can sometimes fail to preserve exact lexical information. For the highly technical domains of law, bioinformatics, and code, this is a critical limitation. These fields are replete with specific, non-negotiable keywords, function names, legal statutes, and gene identifiers (e.g., res ipsa loquitur, KRAS G12C, torch.nn.functional.relu) where the exact term is more important than a semantic synonym.

To address this, the Semantic Core must incorporate sparse vector representations. Sparse vectors, such as those generated by models like SPLADE (SParse Lexical AnD Expansion), are very high-dimensional (e.g., 30,000+ dimensions, corresponding to the model's vocabulary) but contain very few non-zero values.32 Each non-zero value represents the importance weight of a specific token in the document.34 This structure makes them highly efficient for storage and excellent for exact keyword matching, while also providing a degree of semantic expansion (e.g., matching "car" with "automobile").33

A key advantage of sparse vectors is their strong performance in zero-shot or out-of-domain retrieval scenarios.35 Because they are fundamentally based on term representations, they do not suffer from the same domain shift issues as dense vectors, which may struggle when encountering vocabulary not seen during fine-tuning. Furthermore, their structure is highly interpretable, as one can directly inspect which terms are being weighted for a given document or query.33

For Chimera-1's target domains, a hybrid search system that combines dense and sparse vectors is not merely an enhancement; it is a necessity for achieving high accuracy and reliability. The dense MRL embeddings will capture the broad semantic context and conceptual relationships, while the sparse SPLADE embeddings will ensure that critical, domain-specific keywords and identifiers are precisely matched. Therefore, the Semantic Core must be designed to produce both a dense and a sparse embedding for every piece of textual input.

### **2.3. Architectural Integration: A Dual Dense-Sparse Output Module and Hybrid Fusion**

To implement this dual-representation strategy, the Chimera-1 Semantic Core will be built upon the mixedbread-ai/mxbai-embed-large-v1 base model and augmented with a second output head. The model will have:

* **A Dense Head:** An MRL-enabled linear layer that produces the 1024-dimensional adaptive dense embedding.  
* **A Sparse Head:** A SPLADE-style head that produces a high-dimensional (e.g., \~30,522-dim, matching the BERT vocabulary size) sparse lexical embedding.

The retrieval process will then execute a hybrid search, running parallel queries against both the dense and sparse vector indexes and merging the results.36 The final, unified ranking of results will be determined by a fusion algorithm. The state-of-the-art method for this task is

**Reciprocal Rank Fusion (RRF)**.

RRF is a powerful and elegant fusion technique that combines multiple ranked lists without needing to normalize their disparate and often incompatible scores.38 It works by assigning a score to each document based on the reciprocal of its rank in each result list. The final score for a document is the sum of its reciprocal rank scores from all lists it appears in. The formula is:

Score(d)=r∈R∑​k+rankr​(d)1​  
where R is the set of ranked lists, rankr​(d) is the rank of document d in list r, and k is a constant that dampens the influence of lower-ranked items.41 This method naturally prioritizes documents that consistently appear at or near the top of multiple search results, making it robust and effective.38

The k parameter in the RRF formula is a crucial tuning knob that modulates the penalty for lower-ranked documents.41 A smaller

k (e.g., 5-15) gives significantly more weight to the top-ranked items, making the fusion highly sensitive to the best results. A larger k (e.g., 30-60) creates more uniform scores, giving more influence to documents that appear further down the lists.40 While a default value of

k=60 is common, this may not be optimal for all of Chimera-1's diverse use cases.39

For queries where lexical precision is paramount (e.g., searching for a specific function name in code or a legal term of art), the sparse search results are likely to be most relevant. In these cases, a smaller k value is preferable to heavily weight the top results from the SPLADE index. Conversely, for more exploratory, semantic queries (e.g., "find research papers related to cellular senescence"), the dense search results are more important, and a larger k value can provide a more balanced fusion that incorporates broader semantic matches. This suggests that the RRF k parameter should not be a fixed constant for Chimera-1. Instead, it should be a dynamic parameter, adjusted based on a simple pre-analysis of the query. A query classifier that detects the presence of code, legal jargon, or other high-precision terms can trigger the use of a smaller k, thereby tailoring the fusion strategy to the user's implicit intent and significantly improving the relevance of the final search results.

### **2.4. Compatibility with Quantization and Long-Context Retrieval**

The proposed hybrid dense-sparse architecture is fully compatible with the broader Chimera-1 design constraints. The chosen mixedbread-ai/mxbai-embed-large-v1 model already supports advanced quantization techniques, such as binary quantization, which can be applied to further reduce the memory footprint of the dense embeddings.14

Critically, this entire retrieval framework is designed to be compatible with the million-token context requirement by leveraging the **Late Chunking** method. Traditional retrieval methods pre-chunk long documents and embed each chunk in isolation, thereby losing the vital context that exists between chunks.44 Late chunking reverses this process. First, the entire long document is passed through the embedding model to generate context-aware token embeddings. Only after this step is the document divided into chunks, with the final embedding for each chunk being an aggregation (e.g., mean pooling) of its now context-aware token embeddings.45 This ensures that the vector representation of each chunk is informed by the full document context. Implementations of this technique are readily available and can be integrated into the Chimera-1 data processing pipeline.49 The retrieval pipeline will therefore first use Late Chunking to generate contextually-aware chunks from the million-token input, and then embed these chunks using the MRL+SPLADE model for the subsequent Funnel Retrieval process.

## **Section 3: Domain Specialization — Tailoring the Core for Expert Knowledge**

A general-purpose semantic space, no matter how powerful, will underperform in the highly specialized and nuanced domains of bioinformatics, law, and software engineering. To equip Chimera-1 with true expert-level understanding, its Semantic Core must be systematically adapted to these domains. This specialization will be achieved through a deliberate, two-phase process: first, a broad infusion of domain knowledge via Domain-Adaptive Pre-Training (DAPT), and second, a sharp optimization for retrieval tasks via Contrastive Fine-Tuning (CFT).

### **3.1. Curating High-Fidelity Corpora: Best Practices**

The foundation of any successful domain adaptation is the quality and breadth of the training data.53 The first step is therefore to assemble three distinct, large-scale, and high-quality corpora, one for each target domain. The curation process must adhere to best practices for dataset creation, including responsible sourcing to navigate copyright and licensing issues, thorough documentation, and rigorous cleaning.55

* **Legal Corpus:** This corpus should be diverse, encompassing a wide range of legal documents to capture the breadth of the domain. It will include legislation from various jurisdictions, extensive collections of court cases (case law), and a variety of legal contracts and agreements.57 Publicly available resources will be scraped and compiled to create a comprehensive dataset on the order of 12 GB or more, similar to the corpus used for LEGAL-BERT.57  
* **Bioinformatics Corpus:** This corpus will be assembled from large-scale, primarily unlabeled data from core life science fields. This includes the full text of research articles from repositories like PubMed Central, covering genomics, proteomics, molecular biology, and clinical studies.54 The goal is to capture the vocabulary, syntax, and conceptual relationships prevalent in scientific literature.  
* **Code Corpus:** This corpus will consist of a massive collection of source code and associated natural language text. It will be built from open-source repositories on platforms like GitHub, covering multiple programming languages (Python, Java, JavaScript, etc.) and their corresponding documentation, comments, and issue discussions. This mirrors the bimodal NL-PL pair data used to train powerful code models like CodeBERT.58

These three corpora will serve as the raw material for infusing the Chimera-1 Semantic Core with the necessary domain expertise.

### **3.2. Phase 1: Domain-Adaptive Pre-Training (DAPT) for Knowledge Infusion**

Domain-Adaptive Pre-Training (DAPT) is the process of taking a general-purpose pre-trained model and continuing its training on a large, domain-specific corpus.57 This technique, also known as continual pre-training, refines the model's internal representations to align with the target domain's vocabulary, concepts, and linguistic patterns.61 The training objective for DAPT is typically self-supervised, with Masked Language Modeling (MLM) being the most common and effective choice.62 In MLM, a percentage of tokens in the input text are randomly masked, and the model is trained to predict the original tokens based on the surrounding context. This process forces the model to learn the statistical and semantic regularities of the domain-specific language. This is the same methodology used to create successful domain-specific models like BioBERT and LEGAL-BERT.57

For Chimera-1, the mixedbread-ai/mxbai-embed-large-v1 base model will undergo DAPT using a mixture of the three curated corpora. This will enrich the model's foundational knowledge, ensuring it understands the specific terminology and conceptual frameworks of law, bioinformatics, and code.

Crucially, this DAPT phase is also the ideal stage to extend the model's effective context window to meet the million-token requirement. The base model's 512-token limit will be expanded using **LongLoRA**, an efficient fine-tuning method specifically designed for context extension.20 LongLoRA employs an innovative

**Shifted Sparse Attention (S²-Attn)** mechanism during training.64 S²-Attn approximates the computationally expensive full attention over a long sequence with a much more efficient sparse, local attention pattern, dramatically reducing memory and compute requirements while achieving comparable performance to full fine-tuning.21 By integrating LongLoRA into the DAPT stage, we can simultaneously infuse domain knowledge and prepare the model for long-context retrieval tasks.

### **3.3. Phase 2: Contrastive Fine-Tuning (CFT) for Retrieval Acuity**

While DAPT endows the model with domain knowledge, it does not explicitly optimize it for retrieval tasks like semantic search. Research has shown that DAPT can sometimes even degrade performance on prompting and retrieval-style tasks, suggesting a trade-off between raw knowledge and task-specific alignment.60 To address this, a second, distinct phase of fine-tuning is required: Contrastive Fine-Tuning (CFT).

CFT sharpens the model's ability to perform retrieval by training it to distinguish between relevant and irrelevant documents for a given query.67 This is a supervised process that requires a labeled dataset of (query, positive\_document, negative\_document) triplets.67 The model is then trained with a contrastive loss function, such as Triplet Loss, which adjusts the embedding space to minimize the distance between a query and its positive document while maximizing the distance between the query and its negative document.68

This two-phase DAPT-then-CFT approach is a core recommendation for specializing the Semantic Core. It decouples the process of knowledge acquisition from retrieval optimization, allowing the use of the most effective training objective for each goal. DAPT with MLM is for learning "what things are" within a domain. CFT with a contrastive loss is for learning "what is relevant to what" for a search query. This structured approach mitigates the risks of a single-stage process and ensures the final model is both knowledgeable and highly adept at retrieval. To ensure this fine-tuning stage is computationally efficient, it will be performed using Low-Rank Adaptation (LoRA), which significantly reduces the number of trainable parameters without sacrificing performance.68

### **3.4. Triplet Mining and Hard Negative Generation Strategies**

The efficacy of Contrastive Fine-Tuning is critically dependent on the quality of the training triplets, particularly the choice of negative examples.71 Using randomly sampled documents as negatives is ineffective, as they are typically too semantically distant from the query to provide a meaningful learning signal for the model.72 The model can easily distinguish them, and the training objective provides little gradient information.

To create a challenging and effective training signal, a technique called **Hard Negative Mining** is employed. This involves finding negative documents that are semantically similar to the query but are factually irrelevant, forcing the model to learn the subtle, fine-grained distinctions that separate correct from incorrect answers.68 This can be done "online" within a training batch or "offline" in a separate pre-processing step.71 Online methods include "batch hard" mining, which selects the most difficult positive and negative examples within each batch.71

For Chimera-1, a more robust offline hard negative mining pipeline is recommended. This pipeline will leverage a dedicated tool like negminer to systematically generate high-quality hard negatives for our domain-specific fine-tuning datasets.72 The process will be as follows:

1. **Candidate Retrieval:** For each query in our domain-specific triplet datasets, we will use the model from the DAPT stage to perform an initial ANN search and retrieve the top N candidate documents.  
2. **Re-ranking and Scoring:** These N candidates will then be re-ranked using a powerful cross-encoder model (e.g., BAAI/bge-reranker-v2-m3 76). A cross-encoder is more computationally expensive but more accurate than a bi-encoder, as it processes the query and document together, allowing for deeper attention-based scoring.  
3. **Negative Selection:** The cross-encoder will score each candidate for its relevance to the query. The highest-scoring documents that are known to be *not* the correct answer (i.e., not the positive document) will be selected as the hard negatives for that query's triplet.

This systematic process creates a highly challenging and informative training set for the CFT phase, ensuring that the model is trained to make the precise distinctions necessary for expert-level retrieval in its target domains.

## **Section 4: The Unified Training Objective — A Multi-Loss Optimization Framework**

The training objective, or loss function, is the mathematical formulation that guides the model's learning process. For a sophisticated multimodal system like Chimera-1, a single, simple loss function is insufficient. To achieve a semantic space that is robustly aligned both within and across modalities, and that captures both global relationships and fine-grained details, a unified training objective composed of multiple, complementary loss functions is required. The proposed framework is a weighted synthesis of Triple Contrastive Learning (TCL) and Joint Masked Vision-Language Modeling (MVLM).

### **4.1. The Primary Objective: Triple Contrastive Learning (TCL) for Structural Alignment**

The dominant paradigm for aligning modalities in a shared embedding space is contrastive learning.77 The most well-known example is CLIP, which learns to align images and their text captions by pulling the embeddings of matched pairs together and pushing mismatched pairs apart in the latent space.11 This is typically achieved with a contrastive loss like InfoNCE.77

However, standard cross-modal contrastive learning has a significant limitation: while it enforces alignment *between* modalities, it can neglect the semantic structure *within* each modality.78 This can lead to a phenomenon where the embedding space becomes well-aligned for image-text pairs but loses its internal coherence, meaning two semantically similar images might drift apart, or two similar sentences might no longer be close.

To overcome this, the primary training objective for the Chimera-1 Semantic Core will be **Triple Contrastive Learning (TCL)**.78 TCL enhances standard contrastive learning by introducing an additional

*intra-modal* contrastive objective. The total loss function therefore has three components:

1. **Cross-Modal Alignment (Image-Text):** The standard contrastive loss that pulls matching image-text pairs together.  
2. **Intra-Modal Alignment (Image-Image):** A contrastive loss that pulls embeddings of similar images (e.g., augmentations of the same image) together.  
3. **Intra-Modal Alignment (Text-Text):** A contrastive loss that pulls embeddings of similar texts (e.g., paraphrases or sentences from the same document) together.

By optimizing all three objectives simultaneously, TCL ensures that the resulting semantic space is not only aligned across modalities but also preserves the rich, internal semantic structure of both the visual and textual domains. This creates a more robust, coherent, and well-structured space, which will be the primary objective during both the initial vision-text alignment stage and the final Contrastive Fine-Tuning (CFT) stage.

### **4.2. The Auxiliary Objective: Joint Masked Vision-Language Modeling (MVLM) for Fine-Grained Understanding**

While contrastive learning is exceptionally effective for learning global semantic alignments, it can sometimes miss the fine-grained, token-level interactions between modalities.12 To capture these details, a generative, reconstruction-based objective is needed. Masked Language Modeling (MLM) and Masked Image Modeling (MIM) are powerful self-supervised techniques that train a model by hiding (masking) parts of the input and forcing the model to predict the missing content.11

**Joint Masked Vision-Language Modeling (MVLM)** extends this principle to the multimodal domain.79 In this framework, the model is trained on two complementary tasks:

1. **Masked Image Reconstruction:** A portion of the input image is masked, and the model must reconstruct the missing visual information (e.g., pixel values) using the full, unmasked text description as context.  
2. **Masked Text Reconstruction:** A portion of the input text is masked, and the model must predict the missing tokens using the full, unmasked image as context.

This cross-modal reconstruction task forces the model to learn a deep, fine-grained alignment between specific visual features and corresponding text tokens.79 It implicitly teaches the model to ground language in vision and vice-versa. A significant advantage of MVLM is its data efficiency; it has been shown to outperform other methods by a significant margin in low-data regimes, as it extracts a stronger learning signal from each image-text pair.79 The loss function for MVLM is a combination of a reconstruction loss for the image (e.g., L1 or L2 distance on pixels) and a standard cross-entropy loss for the masked text tokens.79

For Chimera-1, an MVLM loss will be incorporated as an *auxiliary* objective during the DAPT and CFT training stages. It will run in parallel with the primary training objectives (MLM during DAPT, TCL during CFT), providing a complementary learning signal that encourages the model to develop a more detailed, fine-grained understanding of the relationship between vision and language.

### **4.3. The Chimera-1 Semantic Core Loss: A Weighted Synthesis**

The final, unified training objective for the Semantic Core will be a dynamically weighted sum of the primary and auxiliary loss components. This multi-task learning approach allows the model to benefit from both discriminative (contrastive) and generative (reconstructive) signals, a strategy employed by many state-of-the-art models.12

For instance, during the crucial Contrastive Fine-Tuning (CFT) stage, the total loss will be formulated as:

Ltotal​=wTCL​⋅LTCL​+wMVLM​⋅LMVLM​  
Here, LTCL​ is the Triple Contrastive Loss, LMVLM​ is the Joint Masked Vision-Language Modeling loss, and wTCL​ and wMVLM​ are scalar weights that balance their relative importance.

However, a static weighting may not be optimal throughout the entire training process. The relative importance of learning global structure versus fine-grained details can change as the model converges. Early in training, establishing a robust, globally coherent semantic structure is the most critical task. As the model matures, refining the finer details of cross-modal interaction becomes more important.

Therefore, a **dynamic loss weighting schedule** will be implemented. The training will begin with a higher weight for the TCL loss (wTCL​\>wMVLM​) to prioritize the rapid establishment of a well-structured, globally aligned space. As training progresses and the TCL loss begins to plateau, the schedule will gradually increase the weight of the MVLM loss (wMVLM​), shifting the model's focus toward learning the more subtle, fine-grained cross-modal reconstructions. This curriculum-based approach to the loss function will guide the model through a more effective and efficient learning trajectory, ensuring that it first builds a solid foundation and then elaborates upon it with rich detail.

## **Section 5: Final Blueprint and Implementation Roadmap**

This section synthesizes the preceding analyses into a final, comprehensive blueprint for the Chimera-1 Semantic Core. It details the complete architecture, outlines the end-to-end staged training protocol, and provides specific guidance on hyperparameter tuning for the key components. This roadmap serves as an actionable guide for the implementation and development of this foundational system.

### **5.1. The Complete Semantic Core Architecture**

The proposed architecture is a hybrid, multimodal, and domain-specialized system designed for performance and efficiency.

* **Base Text Encoder:** The core of the text processing pipeline will be the mixedbread-ai/mxbai-embed-large-v1 model. This is a BERT-large architecture with 24 layers, a 1024-dimensional hidden size, and a vocabulary of approximately 30,522 tokens.18 It will be adapted for long-context and specialized domains.  
* **Vision Encoder:** The vision encoder will be the google/vit-base-patch16-224 Vision Transformer, as defined in the Chimera-1 Multimodal Sensory System Refined\_.md report. This model is pre-trained on ImageNet-21k and provides robust general-purpose visual features.81  
* **Projection Layer:** A lightweight 2-layer Multi-Layer Perceptron (MLP) will serve as the bridge between the vision and text modalities. It will project the output features from the ViT into the 1024-dimensional embedding space of the text encoder.  
* **Output Heads:** The model will feature two distinct output heads to generate the hybrid dense-sparse representation:  
  1. **Dense Head:** A linear layer enabled with Matryoshka Representation Learning (MRL). It will output a full 1024-dimensional dense vector that can be truncated to smaller, nested dimensions (e.g., 768, 512, 384, 256, 128\) for adaptive retrieval.  
  2. **Sparse Head:** A SPLADE-style head that outputs a high-dimensional sparse vector (dimension size matching the vocabulary, \~30k). This head will be trained to predict the importance weights of vocabulary tokens for lexical retrieval.

### **5.2. End-to-End Staged Training Protocol**

The training process is divided into three distinct stages to systematically build the desired capabilities into the Semantic Core.

* **Stage 1: Initial Cross-Modal Alignment**  
  * **Objective:** To teach the model to map visual concepts into the text embedding space.  
  * **Procedure:**  
    1. Freeze the parameters of the base mxbai-embed-large-v1 text encoder.  
    2. Train only the google/vit-base-patch16-224 vision encoder and the MLP projection layer.  
    3. Use a large-scale, general-purpose image-text dataset (e.g., a filtered subset of LAION or CC12M).  
  * **Loss Function:** The primary training objective will be the **Triple Contrastive Loss (LTCL​)** to establish a robust, structurally sound alignment between and within modalities.  
* **Stage 2: Domain-Adaptive Pre-Training (DAPT)**  
  * **Objective:** To infuse the model with expert knowledge and extend its context window.  
  * **Procedure:**  
    1. Unfreeze all model components (text encoder, vision encoder, projector).  
    2. Continue training on the curated mixture of Legal, Bioinformatics, and Code corpora.  
    3. Employ **LongLoRA** with its **Shifted Sparse Attention (S²-Attn)** mechanism to efficiently extend the model's context handling capabilities towards the million-token target.  
  * **Loss Function:** A weighted sum of a primary reconstruction loss and an auxiliary cross-modal loss: **LDAPT​=LMLM​+waux​⋅LMVLM​**. The primary Masked Language Modeling objective will drive the learning of domain vocabulary, while the auxiliary Joint Masked Vision-Language Modeling objective will continue to refine fine-grained alignments.  
* **Stage 3: Contrastive Fine-Tuning (CFT)**  
  * **Objective:** To sharpen the model's retrieval performance for domain-specific search tasks.  
  * **Procedure:**  
    1. Fine-tune the full model using **LoRA** for parameter efficiency.  
    2. Train on the domain-specific triplet datasets generated via the offline hard negative mining pipeline.  
  * **Loss Function:** A dynamically weighted sum of the contrastive and reconstructive losses: **LCFT​=wTCL​⋅LTCL​+wMVLM​⋅LMVLM​**. The loss weights will be scheduled to first prioritize global alignment (TCL) and then shift focus to fine-grained reconstruction (MVLM).

### **5.3. Hyperparameter Tuning Guidance**

Effective implementation will require careful tuning of key hyperparameters.

* **LoRA (r and alpha):**  
  * The r parameter defines the rank (and thus the number of trainable parameters) of the LoRA update matrices. The lora\_alpha parameter is a scaling factor for the LoRA updates. The effective learning rate of the LoRA update is proportional to lora\_alpha / r.82  
  * **Recommendation:** Start with a rank of r=16 or r=32 and set lora\_alpha \= r. This is a stable baseline.83 For more aggressive learning, a common heuristic is to set  
    lora\_alpha to double the rank (lora\_alpha \= 2\*r), but this should be paired with a lower base learning rate to maintain stability.84 For maximum performance, LoRA should be applied to all major linear layers in the transformer blocks, including  
    q\_proj, k\_proj, v\_proj, and o\_proj.70  
* **Reciprocal Rank Fusion (k):**  
  * The k parameter in the RRF formula controls the fusion of dense and sparse search results. It determines how steeply lower-ranked documents are penalized.  
  * **Recommendation:** Begin with the commonly used default of k=60.39 Implement a dynamic  
    k selection mechanism based on query analysis. For queries identified as requiring high lexical precision (e.g., containing code), test smaller k values in the range of 5-15 to heavily favor the top results from the sparse index.43 For general semantic queries, test larger  
    k values in the range of 30-50 to achieve a more balanced fusion of dense and sparse results.43  
* **Matryoshka Representation Learning (MRL):**  
  * The matryoshka\_dims parameter defines the nested dimensions to be trained.  
  * **Recommendation:** A suitable set of dimensions for the 1024-dim base model would be . The \`matryoshka\_weights\` parameter controls the contribution of each dimension's loss to the total loss. A uniform weighting of is a robust starting point.28 These weights can be adjusted later if specific smaller dimensions need to be prioritized for certain high-speed retrieval tasks.

### **5.4. Proposed Table: Chimera-1 Semantic Core Architectural Blueprint**

The following table summarizes the core architectural and training decisions for the Chimera-1 Semantic Core.

| Component | Recommendation | Rationale & Key Sources |
| :---- | :---- | :---- |
| **Foundational Model** | Fine-tune mixedbread-ai/mxbai-embed-large-v1 | State-of-the-art performance on MTEB, permissive Apache-2.0 license, and direct architectural synergy with MRL and BERT-style contrastive learning. 14 |
| **Embedding Structure** | Hybrid Dense (MRL) \+ Sparse (SPLADE) | MRL enables efficient "Funnel Retrieval" for million-token contexts. SPLADE provides essential lexical precision for technical domains like law and code. 22 |
| **Fusion Strategy** | Reciprocal Rank Fusion (RRF) with Dynamic k | State-of-the-art, score-normalization-free method for combining dense and sparse results. Dynamic k adapts fusion to query type (lexical vs. semantic). 38 |
| **Domain Adaptation** | Two-Phase: DAPT (MLM) \+ CFT (Contrastive) | Separates knowledge infusion from retrieval tuning, using the optimal objective for each task and mitigating performance trade-offs. 60 |
| **Training Objective** | Weighted sum of Triple Contrastive Loss (TCL) \+ Joint Masked V-L Modeling (MVLM) | TCL provides robust structural alignment (both intra- and inter-modal). MVLM adds fine-grained, token-level cross-modal understanding. 78 |

### **5.5. Concluding Remarks**

The architecture and training protocol detailed in this report constitute a state-of-the-art blueprint for the Chimera-1 Semantic Core. The design is robust, efficient, and meticulously tailored to the project's unique requirements for multimodality, domain specialization, and long-context operation.

By adopting a hybrid strategy built upon a premier open-source foundation model, the plan leverages vast, pre-existing knowledge while focusing resources on targeted adaptation. The integration of advanced representation techniques—namely Matryoshka Representation Learning and sparse embeddings—directly addresses the critical challenge of efficiency at the million-token scale. The two-phase specialization process ensures the model will possess both deep domain knowledge and sharp retrieval acuity. Finally, the sophisticated, multi-task training objective is designed to create a semantic space that is structurally sound, richly detailed, and coherently aligned across modalities.

The feasibility of this ambitious plan is underpinned by the availability of high-quality open-source models, libraries, and research for every recommended component. This blueprint provides a clear and comprehensive path to engineering a powerful and efficient semantic foundation for the Chimera-1 model.

#### **Works cited**

1. \[2503.14963\] Continual Multimodal Contrastive Learning \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2503.14963](https://arxiv.org/abs/2503.14963)  
2. Efficient Multimodal Large Language Models: A Survey \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2405.10739v1](https://arxiv.org/html/2405.10739v1)  
3. How to Train and Fine Tune a Multimodal Language Model \[+ Use ..., accessed July 4, 2025, [https://hatchworks.com/blog/gen-ai/train-and-fine-tune-multimodal-model/](https://hatchworks.com/blog/gen-ai/train-and-fine-tune-multimodal-model/)  
4. Multimodal AI: A Guide to Open-Source Vision Language Models \- BentoML, accessed July 4, 2025, [https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)  
5. The best open source large language model | Baseten Blog, accessed July 4, 2025, [https://www.baseten.co/blog/the-best-open-source-large-language-model/](https://www.baseten.co/blog/the-best-open-source-large-language-model/)  
6. Four approaches to creating a specialized LLM \- Stack Overflow, accessed July 4, 2025, [https://stackoverflow.blog/2024/12/05/four-approaches-to-creating-a-specialized-llm/](https://stackoverflow.blog/2024/12/05/four-approaches-to-creating-a-specialized-llm/)  
7. Aligning Vision Language Models with Contrastive Learning \- Amazon Science, accessed July 4, 2025, [https://assets.amazon.science/36/5c/19734bdf4fdb8da3cc809590c05d/aligning-vision-language-models-with-contrastive-learning.pdf](https://assets.amazon.science/36/5c/19734bdf4fdb8da3cc809590c05d/aligning-vision-language-models-with-contrastive-learning.pdf)  
8. Aligning vision language models with contrastive learning \- Amazon Science, accessed July 4, 2025, [https://www.amazon.science/publications/aligning-vision-language-models-with-contrastive-learning](https://www.amazon.science/publications/aligning-vision-language-models-with-contrastive-learning)  
9. Training-free Deep Concept Injection Enables Language Models for ..., accessed July 4, 2025, [https://aclanthology.org/2024.emnlp-main.1249/](https://aclanthology.org/2024.emnlp-main.1249/)  
10. TokenPacker: Efficient Visual Projector for Multimodal LLM \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2407.02392v1](https://arxiv.org/html/2407.02392v1)  
11. Guide to Vision-Language Models (VLMs) \- Encord, accessed July 4, 2025, [https://encord.com/blog/vision-language-models-guide/](https://encord.com/blog/vision-language-models-guide/)  
12. Connecting Text with Vision: How Multimodal Models Align ..., accessed July 4, 2025, [https://medium.com/@hexiangnan/connecting-text-with-vision-how-multimodal-models-align-modalities-in-the-embedding-space-1d97abbda472](https://medium.com/@hexiangnan/connecting-text-with-vision-how-multimodal-models-align-modalities-in-the-embedding-space-1d97abbda472)  
13. Top embedding models on the MTEB leaderboard | Modal Blog, accessed July 4, 2025, [https://modal.com/blog/mteb-leaderboard-article](https://modal.com/blog/mteb-leaderboard-article)  
14. Mxbai Embed Large V1 · Models \- Dataloop, accessed July 4, 2025, [https://dataloop.ai/library/model/mixedbread-ai\_mxbai-embed-large-v1/](https://dataloop.ai/library/model/mixedbread-ai_mxbai-embed-large-v1/)  
15. Embedding Models \- Mixedbread, accessed July 4, 2025, [https://www.mixedbread.com/docs/embeddings/models](https://www.mixedbread.com/docs/embeddings/models)  
16. mxbai-embed-large-v1 model | Clarifai \- The World's AI, accessed July 4, 2025, [https://clarifai.com/mixedbread-ai/embed/models/mxbai-embed-large-v1](https://clarifai.com/mixedbread-ai/embed/models/mxbai-embed-large-v1)  
17. What are the licensing considerations for embedding models? \- Milvus, accessed July 4, 2025, [https://milvus.io/ai-quick-reference/what-are-the-licensing-considerations-for-embedding-models](https://milvus.io/ai-quick-reference/what-are-the-licensing-considerations-for-embedding-models)  
18. config.json · mixedbread-ai/mxbai-embed-large-v1 at main \- Hugging Face, accessed July 4, 2025, [https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/blob/main/config.json](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/blob/main/config.json)  
19. Finetuned mixedbread ai deepset mxbai embed de large v1 · Models \- Dataloop, accessed July 4, 2025, [https://dataloop.ai/library/model/fareedkhan\_finetuned\_mixedbread\_ai\_deepset\_mxbai\_embed\_de\_large\_v1/](https://dataloop.ai/library/model/fareedkhan_finetuned_mixedbread_ai_deepset_mxbai_embed_de_large_v1/)  
20. LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2309.12307v2](https://arxiv.org/html/2309.12307v2)  
21. \[2309.12307\] LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2309.12307](https://arxiv.org/abs/2309.12307)  
22. Matryoshka Representation Learning Explained: The Method Behind OpenAI's Efficient Text Embeddings | by Zilliz | Medium, accessed July 4, 2025, [https://medium.com/@zilliz\_learn/matryoshka-representation-learning-explained-the-method-behind-openais-efficient-text-embeddings-a600dfe85ff8](https://medium.com/@zilliz_learn/matryoshka-representation-learning-explained-the-method-behind-openais-efficient-text-embeddings-a600dfe85ff8)  
23. Matryoshka Representation Learning \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2205.13147v4](https://arxiv.org/html/2205.13147v4)  
24. Matryoshka Representation Learning \- Emergent Mind, accessed July 4, 2025, [https://www.emergentmind.com/topics/matryoshka-representation-learning](https://www.emergentmind.com/topics/matryoshka-representation-learning)  
25. RAIVNLab/MRL: Code repository for the paper \- "Matryoshka Representation Learning" \- GitHub, accessed July 4, 2025, [https://github.com/RAIVNLab/MRL](https://github.com/RAIVNLab/MRL)  
26. Papers Explained 96: Matryoshka Representation Learning | by Ritvik Rastogi \- Medium, accessed July 4, 2025, [https://ritvik19.medium.com/papers-explained-matryoshka-representation-learning-e7a139f6ad27](https://ritvik19.medium.com/papers-explained-matryoshka-representation-learning-e7a139f6ad27)  
27. Matryoshka Representation Learning \- Thalles' blog, accessed July 4, 2025, [https://sthalles.github.io/matryoshka-representation-learning/](https://sthalles.github.io/matryoshka-representation-learning/)  
28. Matryoshka Representation Learning with CLIP for Multimodal Retrieval and Ranking, accessed July 4, 2025, [https://www.marqo.ai/blog/matryoshka-representation-learning-with-clip-for-multimodal-retrieval-and-ranking](https://www.marqo.ai/blog/matryoshka-representation-learning-with-clip-for-multimodal-retrieval-and-ranking)  
29. Exploring the potential of OpenAI Matryoshka embeddings with Vespa, accessed July 4, 2025, [https://blog.vespa.ai/matryoshka-embeddings-in-vespa/](https://blog.vespa.ai/matryoshka-embeddings-in-vespa/)  
30. New embedding models and API updates | OpenAI, accessed July 4, 2025, [https://openai.com/index/new-embedding-models-and-api-updates/](https://openai.com/index/new-embedding-models-and-api-updates/)  
31. Matryoshka Representation Learning, accessed July 4, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf)  
32. Exploring Sparse and Dense Embeddings: A Guide for Effective Information Retrieval with Milvus \- YouTube, accessed July 4, 2025, [https://www.youtube.com/watch?v=6\_Tjdu6IZdA](https://www.youtube.com/watch?v=6_Tjdu6IZdA)  
33. Training and Finetuning Sparse Embedding Models with Sentence ..., accessed July 4, 2025, [https://huggingface.co/blog/train-sparse-encoder](https://huggingface.co/blog/train-sparse-encoder)  
34. Embedding Overview Milvus v2.4.x documentation, accessed July 4, 2025, [https://milvus.io/docs/v2.4.x/embeddings.md](https://milvus.io/docs/v2.4.x/embeddings.md)  
35. Understanding sparse vector embeddings with trained ML models ..., accessed July 4, 2025, [https://www.elastic.co/search-labs/blog/sparse-vector-embedding](https://www.elastic.co/search-labs/blog/sparse-vector-embedding)  
36. Multi-Vector Hybrid Search | Milvus Documentation, accessed July 4, 2025, [https://milvus.io/docs/multi-vector-search.md](https://milvus.io/docs/multi-vector-search.md)  
37. Hybrid Search: Combining Dense and Sparse Vectors for Superior Search Results, accessed July 4, 2025, [https://dev.to/skitsanosdesign/hybrid-search-combining-dense-and-sparse-vectors-for-superior-search-results-4dod](https://dev.to/skitsanosdesign/hybrid-search-combining-dense-and-sparse-vectors-for-superior-search-results-4dod)  
38. Hybrid Search Explained | Weaviate, accessed July 4, 2025, [https://weaviate.io/blog/hybrid-search-explained](https://weaviate.io/blog/hybrid-search-explained)  
39. Relevance scoring in hybrid search using Reciprocal Rank Fusion (RRF) \- Learn Microsoft, accessed July 4, 2025, [https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)  
40. Introducing reciprocal rank fusion for hybrid search \- OpenSearch, accessed July 4, 2025, [https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)  
41. Reciprocal rank fusion | Elasticsearch Guide \[8.18\] | Elastic, accessed July 4, 2025, [https://www.elastic.co/guide/en/elasticsearch/reference/8.18/rrf.html](https://www.elastic.co/guide/en/elasticsearch/reference/8.18/rrf.html)  
42. RAG — VII (Reranking with RRF). Reciprocal Rank Fusion (RRF) is a rank… | by DhanushKumar | Medium, accessed July 4, 2025, [https://medium.com/@danushidk507/rag-vii-reranking-with-rrf-d8a13dba96de](https://medium.com/@danushidk507/rag-vii-reranking-with-rrf-d8a13dba96de)  
43. drittich/reciprocal-rank-fusion \- GitHub, accessed July 4, 2025, [https://github.com/drittich/reciprocal-rank-fusion](https://github.com/drittich/reciprocal-rank-fusion)  
44. Unlocking Better Text Retrieval with Late Chunking: A Revolutionary Approach for RAG Applications | by Bluetick Consultants Inc. | May, 2025, accessed July 4, 2025, [https://bluetickconsultants.medium.com/unlocking-better-text-retrieval-with-late-chunking-a-revolutionary-approach-for-rag-applications-2366ea508719](https://bluetickconsultants.medium.com/unlocking-better-text-retrieval-with-late-chunking-a-revolutionary-approach-for-rag-applications-2366ea508719)  
45. Late Chunking: Contextual Chunk Embeddings Using Long-Context ..., accessed July 4, 2025, [https://openreview.net/forum?id=74QmBTV0Zf](https://openreview.net/forum?id=74QmBTV0Zf)  
46. Late Chunking In Long Context Embedding Models \- Towards AI, accessed July 4, 2025, [https://towardsai.net/p/machine-learning/late-chunking-in-long-context-embedding-models](https://towardsai.net/p/machine-learning/late-chunking-in-long-context-embedding-models)  
47. Paper page \- Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models \- Hugging Face, accessed July 4, 2025, [https://huggingface.co/papers/2409.04701](https://huggingface.co/papers/2409.04701)  
48. Late Chunking: Embedding First Chunk Later — Long-Context Retrieval in RAG Applications | by BavalpreetSinghh | Stackademic, accessed July 4, 2025, [https://blog.stackademic.com/late-chunking-embedding-first-chunk-later-long-context-retrieval-in-rag-applications-3a292f6443bb](https://blog.stackademic.com/late-chunking-embedding-first-chunk-later-long-context-retrieval-in-rag-applications-3a292f6443bb)  
49. Late Chunking: Improving RAG Performance with Context-Aware Embeddings, accessed July 4, 2025, [https://www.pondhouse-data.com/blog/advanced-rag-late-chunking](https://www.pondhouse-data.com/blog/advanced-rag-late-chunking)  
50. Practical introduction to Late Chunking or Chunked Pooling \- LanceDB Blog, accessed July 4, 2025, [https://blog.lancedb.com/late-chunking-aka-chunked-pooling-2/](https://blog.lancedb.com/late-chunking-aka-chunked-pooling-2/)  
51. recipes/weaviate-features/services-research/late\_chunking\_berlin.ipynb at main \- GitHub, accessed July 4, 2025, [https://github.com/weaviate/recipes/blob/main/weaviate-features/services-research/late\_chunking\_berlin.ipynb](https://github.com/weaviate/recipes/blob/main/weaviate-features/services-research/late_chunking_berlin.ipynb)  
52. Easy Late-Chunking With Chonkie | Towards AI, accessed July 4, 2025, [https://towardsai.net/p/machine-learning/easy-late-chunking-with-chonkie](https://towardsai.net/p/machine-learning/easy-late-chunking-with-chonkie)  
53. NYU Tandon engineers create first AI model specialized for chip design language, earning top journal honor, accessed July 4, 2025, [https://engineering.nyu.edu/news/nyu-tandon-engineers-create-first-ai-model-specialized-chip-design-language-earning-top](https://engineering.nyu.edu/news/nyu-tandon-engineers-create-first-ai-model-specialized-chip-design-language-earning-top)  
54. Large Language Models for Bioinformatics \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2501.06271v1](https://arxiv.org/html/2501.06271v1)  
55. Towards Best Practices for Open Datasets for LLM Training \- Mozilla ..., accessed July 4, 2025, [https://www.mozillafoundation.org/en/research/library/towards-best-practices-for-open-datasets-for-llm-training/](https://www.mozillafoundation.org/en/research/library/towards-best-practices-for-open-datasets-for-llm-training/)  
56. \[2501.08365\] Towards Best Practices for Open Datasets for LLM Training \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2501.08365](https://arxiv.org/abs/2501.08365)  
57. LEGAL-BERT: The Muppets straight out of Law ... \- ACL Anthology, accessed July 4, 2025, [https://aclanthology.org/2020.findings-emnlp.261.pdf](https://aclanthology.org/2020.findings-emnlp.261.pdf)  
58. microsoft/CodeBERT \- GitHub, accessed July 4, 2025, [https://github.com/microsoft/CodeBERT](https://github.com/microsoft/CodeBERT)  
59. CodeBERT: A Pre-Trained Model for Programming and Natural Languages \- arXiv, accessed July 4, 2025, [https://arxiv.org/abs/2002.08155](https://arxiv.org/abs/2002.08155)  
60. arXiv:2504.13603v1 \[cs.CL\] 18 Apr 2025, accessed July 4, 2025, [https://arxiv.org/pdf/2504.13603](https://arxiv.org/pdf/2504.13603)  
61. Domain-Adaptive Pre-Training: Tailoring LLMs for Specialized Applications | GTC 25 2025 | NVIDIA On-Demand, accessed July 4, 2025, [https://www.nvidia.com/en-us/on-demand/session/gtc25-dlit71175/](https://www.nvidia.com/en-us/on-demand/session/gtc25-dlit71175/)  
62. From Large Language Models to Large Multimodal Models: A Literature Review \- MDPI, accessed July 4, 2025, [https://www.mdpi.com/2076-3417/14/12/5068](https://www.mdpi.com/2076-3417/14/12/5068)  
63. BERT (language model) \- Wikipedia, accessed July 4, 2025, [https://en.wikipedia.org/wiki/BERT\_(language\_model)](https://en.wikipedia.org/wiki/BERT_\(language_model\))  
64. longlora: efficient fine-tuning of long- context large language models \- arXiv, accessed July 4, 2025, [https://arxiv.org/pdf/2309.12307](https://arxiv.org/pdf/2309.12307)  
65. LongLoRA Explained: Efficient Fine-Tuning of Long Context LLMs | by Sheli Kohan, accessed July 4, 2025, [https://ai.plainenglish.io/longlora-how-to-extend-llms-context-sizes-through-fine-tuning-9f27894d1c06](https://ai.plainenglish.io/longlora-how-to-extend-llms-context-sizes-through-fine-tuning-9f27894d1c06)  
66. LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models \- OpenReview, accessed July 4, 2025, [https://openreview.net/forum?id=6PmJoRfdaK](https://openreview.net/forum?id=6PmJoRfdaK)  
67. How do I fine-tune embeddings for domain-specific search? \- Milvus, accessed July 4, 2025, [https://milvus.io/ai-quick-reference/how-do-i-finetune-embeddings-for-domainspecific-search](https://milvus.io/ai-quick-reference/how-do-i-finetune-embeddings-for-domainspecific-search)  
68. Fine-tuning Embeddings for Domain-Specific NLP \- Prem AI Blog, accessed July 4, 2025, [https://blog.premai.io/fine-tuning-embeddings-for-domain-specific-nlp/](https://blog.premai.io/fine-tuning-embeddings-for-domain-specific-nlp/)  
69. A beginners guide to fine tuning LLM using LoRA \- zabirauf || Zohaib, accessed July 4, 2025, [https://zohaib.me/a-beginners-guide-to-fine-tuning-llm-using-lora/](https://zohaib.me/a-beginners-guide-to-fine-tuning-llm-using-lora/)  
70. LoRA Hyperparameters Guide | Unsloth Documentation, accessed July 4, 2025, [https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide](https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide)  
71. Triplet Loss: Intro, Implementation, Use Cases \- V7 Labs, accessed July 4, 2025, [https://www.v7labs.com/blog/triplet-loss](https://www.v7labs.com/blog/triplet-loss)  
72. nixiesearch/negminer: A hard negative mining tool for embedding model training \- GitHub, accessed July 4, 2025, [https://github.com/nixiesearch/negminer](https://github.com/nixiesearch/negminer)  
73. What is hard negative mining and how does it improve embeddings? \- Zilliz, accessed July 4, 2025, [https://zilliz.com/ai-faq/what-is-hard-negative-mining-and-how-does-it-improve-embeddings](https://zilliz.com/ai-faq/what-is-hard-negative-mining-and-how-does-it-improve-embeddings)  
74. How Does Triplet Loss and Online Triplet Mining Work? \- Sanjaya's Blog, accessed July 4, 2025, [https://sanjayasubedi.com.np/deeplearning/online-triplet-mining/](https://sanjayasubedi.com.np/deeplearning/online-triplet-mining/)  
75. Triplet Loss — Deep Learning \- FR, accessed July 4, 2025, [https://perso.esiee.fr/\~chierchg/deep-learning/tutorials/metric/metric-2.html](https://perso.esiee.fr/~chierchg/deep-learning/tutorials/metric/metric-2.html)  
76. The best open-source embedding models | Baseten Blog, accessed July 4, 2025, [https://www.baseten.co/blog/the-best-open-source-embedding-models/](https://www.baseten.co/blog/the-best-open-source-embedding-models/)  
77. Full Guide to Contrastive Learning | Encord, accessed July 4, 2025, [https://encord.com/blog/guide-to-contrastive-learning/](https://encord.com/blog/guide-to-contrastive-learning/)  
78. Better joint representations of image and text \- Amazon Science, accessed July 4, 2025, [https://www.amazon.science/blog/better-joint-representations-of-image-and-text](https://www.amazon.science/blog/better-joint-representations-of-image-and-text)  
79. Masked Vision and Language Modeling for Multi-modal Representation Learning, accessed July 4, 2025, [https://openreview.net/forum?id=ZhuXksSJYWn](https://openreview.net/forum?id=ZhuXksSJYWn)  
80. An Introduction to Vision-Language Modeling \- arXiv, accessed July 4, 2025, [https://arxiv.org/html/2405.17247v1](https://arxiv.org/html/2405.17247v1)  
81. google/vit-base-patch16-224 \- Hugging Face, accessed July 4, 2025, [https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)  
82. Is LoRA Rank \= Precision, Alpha \= Strength relative to rank? : r/LocalLLaMA \- Reddit, accessed July 4, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1bjzc75/is\_lora\_rank\_precision\_alpha\_strength\_relative\_to/](https://www.reddit.com/r/LocalLLaMA/comments/1bjzc75/is_lora_rank_precision_alpha_strength_relative_to/)  
83. Understanding LoRA Adapters Rank and Alpha Parameters \- Datawizz.ai, accessed July 4, 2025, [https://datawizz.ai/blog/understanding-lora-adapters-rank-and-alpha-parameters](https://datawizz.ai/blog/understanding-lora-adapters-rank-and-alpha-parameters)  
84. What is LoRA alpha? · Issue \#762 \- GitHub, accessed July 4, 2025, [https://github.com/TimDettmers/bitsandbytes/issues/762](https://github.com/TimDettmers/bitsandbytes/issues/762)  
85. Eternal question: what rank (r) and alpha to use in QLoRA? : r/LocalLLaMA \- Reddit, accessed July 4, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal\_question\_what\_rank\_r\_and\_alpha\_to\_use\_in/](https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal_question_what_rank_r_and_alpha_to_use_in/)  
86. LoRA \- Hugging Face, accessed July 4, 2025, [https://huggingface.co/docs/peft/main/conceptual\_guides/lora](https://huggingface.co/docs/peft/main/conceptual_guides/lora)