

# **ARC-EVOLUTION: A Framework for Compositional Expression and Dynamic Adaptation in a Genomic Language System**

## **Introduction: From Static Transcriptome to Dynamic Genome**

The ARC-TRANSCRIPTOME research program successfully established a foundational atlas of recurring, static units of human expression. This comprehensive catalog serves as a "reference genome"—a stable, well-defined map of the most common and functionally significant expressive patterns within a given linguistic corpus. It provides the necessary baseline for understanding the building blocks of language.

However, a static atlas, while essential, is insufficient for achieving true linguistic intelligence. Language is not a fixed collection of artifacts but a living, evolving system characterized by compositionality, innovation, and constant adaptation.1 To fully empower the Chimera-1 agent and validate the genomic analogy that has guided this project, the system must move beyond a static picture to a dynamic, generative model. The analogy of the genome as a book or manual has been a powerful heuristic in biology, inspiring new ways to think about mutation, selection, and recombination.2 While recognizing the limitations of this metaphor—language is not literally DNA—its conceptual power to structure inquiry into complex systems remains unparalleled.2

This proposal, **ARC-EVOLUTION**, outlines the next-generation experimental framework designed to bring this linguistic genome to life. It confronts the dynamic, compositional, and evolutionary nature of language head-on. The research is organized into four interconnected thrusts that model the core processes of a living genetic system: (1) a model of **transcriptional splicing** for compositional expression; (2) a protocol for **dynamic transcriptome adaptation** to model linguistic evolution; (3) a framework for **reverse transcription** to enable model self-improvement; and (4) an experimental design to analyze the linguistic "dark matter" or **"junk DNA"** that constitutes the long tail of human creativity. Together, these components provide a comprehensive roadmap for advancing from a static map of expression to a dynamic model of expressive life.

## **I. A Model for Transcriptional Splicing: A Neuro-Symbolic Architecture for Compositional Expression**

To capture the combinatorial richness of language, this section details an architecture for generating a vast space of nuanced expressions by composing core semantic units with an array of modifiers. This process is analogous to the biological mechanism of alternative splicing, where different exons from a single primary gene transcript are combined to produce a multitude of proteins.

### **1.1 The Semantic Gene: Predicate-Argument Structure (PAS) as the Core Semantic Locus**

The fundamental, meaning-encoding "gene" of a linguistic expression is proposed to be its **Predicate-Argument Structure (PAS)**. Rooted in linguistic theory, PAS provides a formal representation of the core semantic relationship between a predicate (an action or state) and its arguments (the entities involved in that event).4 A sentence like "The developer debugged the code" can be reduced to a stable PAS representation such as

DEBUG(AGENT: developer, PATIENT: code). This structure captures the essential meaning, abstracting away from the specific syntactic realization.

The utility of PAS as a semantic backbone is well-established in Natural Language Processing (NLP), where it is foundational to tasks like Semantic Role Labeling (SRL), question answering, and text summarization that require a deep understanding of who did what to whom.4 By leveraging established theoretical frameworks like Generative Grammar and Lexical Functional Grammar to formalize these representations, and employing state-of-the-art NLP techniques for their extraction from text, we can reliably identify these core semantic "genes" within any given corpus.4

### **1.2 Modifier Exons: A Catalog of Expressive Overlays**

While PAS forms the semantic core, the richness, nuance, and pragmatic force of human language arise from "modifier exons"—conventionalized constructions that are layered on top of the core meaning. This framework will explicitly model and catalog several critical classes of these modifiers:

* **Idiomatic Expressions:** These are non-compositional phrases such as "kick the bucket" or "spill the beans," whose meanings are not derivable from their constituent parts.7 Mastery of idioms is essential for human-like fluency and presents a significant challenge for purely literal models, as their interpretation requires cultural knowledge and contextual awareness.9 The system will employ specialized deep learning models, such as fine-tuned BERT or Recurrent Convolutional Neural Networks (RCNNs), as dedicated detectors for these idiomatic constructions.7  
* **Rhetorical Structures:** These are patterns that organize discourse and signal logical relationships between spans of text, such as CONTRAST, ELABORATION, or EVIDENCE.11 Drawing from Rhetorical Structure Theory (RST), the system will define a taxonomy of these relations. Identifying these structures, which are often signaled by discourse markers like "but" or "because," is crucial for understanding authorial intent and generating coherent, persuasive, and logically structured text.11  
* **Stylistic and Pragmatic Constructions:** This category includes a wide range of other conventionalized patterns that convey politeness, formality, register, and other pragmatic functions. Examples include formal modes of address, colloquialisms, and specific phrasal templates that are characteristic of particular genres or social contexts.

### **1.3 Architectural Framework: A Hybrid Construction Grammar (CxG) Approach**

To enable the dynamic "splicing" of semantic genes with modifier exons, this research proposes a **hybrid neuro-symbolic architecture**.13 This design choice is deliberate, seeking to combine the transparent, structured reasoning of symbolic AI with the pattern-recognition power of neural networks, thereby avoiding the brittleness of pure symbolic systems and the "black-box" opacity of pure neural approaches.13

* **Symbolic Layer (The Constructicon):** The heart of the symbolic component is a **Constructicon**, a structured knowledge base inspired by Construction Grammar (CxG).16 In CxG, the traditional distinction between the lexicon (words) and grammar (rules) is dissolved. Instead, all linguistic knowledge is captured in the form of  
  constructions—form-meaning pairings of varying complexity and abstraction, from concrete words to abstract syntactic frames.16 The system's Constructicon will house a comprehensive inventory of constructions, including:  
  * Core PAS "genes" (e.g., the Transitive-Action construction).  
  * Modifier "exons" (e.g., the kick-the-bucket idiom construction, the Contrast-Marker rhetorical construction).  
* **Neural Layer (The Splicing Engine):** The Chimera-1 Large Language Model (LLM) will serve as the neural processing engine. Its role is elevated beyond simply generating text from a flat probability distribution. It will function as a **probabilistic parser and realizer** that operates over the symbolic Constructicon. Given a communicative goal, the LLM will learn to:  
  1. Select a core PAS "gene" construction that matches the intended core meaning.  
  2. Contextually select one or more appropriate "modifier exon" constructions from the Constructicon.  
  3. Unify the formal and semantic constraints of these selected constructions and generate a final surface-form utterance that correctly realizes the combined, nuanced meaning. This process, where constructions can freely combine so long as their constraints do not conflict, is a central tenet of advanced computational CxG frameworks like Fluid Construction Grammar (FCG).16

The interaction between the neural and symbolic layers can be technically implemented using a Graph Neural Network (GNN).19 The Constructicon is represented as a graph of available linguistic components, and the LLM learns to perform a structured traversal of this graph to assemble a valid and contextually appropriate compositional expression. This architecture directly addresses the necessary and sufficient conditions for compositional generalization:

**structural alignment** (the model's computational graph mirrors the compositional structure of language) and **compressed, unambiguous representation** (the constructions provide minimal, well-defined units of meaning).19

This architectural choice unifies the lexicon and grammar into a single, cohesive generative system. The traditional view often treats idioms or rhetorical devices as special cases requiring separate modules. However, Construction Grammar reveals that the distinction between lexicon and grammar is a false dichotomy; both are simply form-meaning pairings at different levels of abstraction.18 This framework, therefore, treats all expressive tools—from the most concrete idiom to the most abstract syntactic frame—as first-class citizens within a unified Constructicon. This transforms the genomic analogy from a mere metaphor into a deep structural principle. The Constructicon

*is* the linguistic genome: the complete, heritable set of expressive possibilities. The LLM is the transcriptional machinery, dynamically reading and splicing these possibilities into coherent expression. This approach is not only more theoretically elegant but also more interpretable and extensible than a purely end-to-end neural model, as it builds a model that reasons over a formal system of composition.

| Framework | Core Unit | Handling of Composition | Strengths | Weaknesses | Alignment with Genomic Analogy |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Predicate-Argument Structure (PAS)** | Predicate-argument relation | Defines core semantic roles. | Excellent for representing core event structure; stable and well-defined. | Lacks a native mechanism for incorporating non-compositional modifiers like idioms or style. | Models the "gene" (core meaning) but not the "exons" (modifiers) or splicing. |
| **Rhetorical Structure Theory (RST)** | Nucleus-satellite relations | Defines discourse-level relations between text spans. | Strong for analyzing and generating coherent, multi-sentence discourse. | Less focused on sentence-internal compositionality and lexical nuance. | Models high-level organization (like gene regulation) but not the composition of a single transcript. |
| **Construction Grammar (CxG)** | Construction (form-meaning pair) | Unifies all linguistic elements as constructions that can be combined. | Natively unifies lexical and grammatical phenomena; highly scalable and extensible. | Can be complex to formalize and implement computationally. | **Excellent.** The Constructicon is the "genome," and the LLM performs "splicing" to combine constructions. |
| *Table 1: Comparison of Compositional Frameworks for the ARC-EVOLUTION System. This table justifies the selection of a Construction Grammar (CxG) based architecture by systematically comparing it against other plausible frameworks, demonstrating its superior alignment with the project's goals.* |  |  |  |  |  |

## **II. A Protocol for Dynamic Transcriptome Adaptation: A Model of Linguistic Evolution**

To reflect the living, changing nature of language, the Reference Transcriptome cannot remain static. This section outlines a two-stage protocol for the system to evolve over time by detecting and canonizing novel linguistic expressions, mirroring the processes of variation and selection in natural language change.

### **2.1 Stage 1: Neologism and Novel Construction Detection**

The first stage acts as the "mutation" or "variation" engine of the system, tasked with identifying emerging, non-canonical expressions from large, streaming text corpora.20 This requires an unsupervised novelty detection pipeline capable of flagging linguistic innovations for further analysis.

The methodology involves several steps:

1. **Embedding and Representation:** All incoming expressions are first converted into high-dimensional vector representations using a state-of-the-art embedding model, such as OpenAI's text-embedding-ada-002, which captures rich semantic meaning.21  
2. **Anomaly Scoring:** A multi-pronged approach is used to score each expression for its degree of novelty or anomaly.22 Rather than relying on a single signal, the system will integrate several techniques:  
   * **Density-Based Methods:** Techniques like the Minimum Covariance Determinant (MCD) are used to build a model of "normal" expressions based on the current transcriptome. Outliers that fall outside the central data cloud are flagged as potential novelties.21  
   * **Likelihood-Based Methods:** Simple low-likelihood scores from an LLM can be noisy indicators of novelty. This system will implement a more sophisticated metric like **"oddballness,"** which measures how "strange" or unexpected a given token or phrase is within its context. This has been shown to be a more robust signal for true grammatical and semantic anomalies.23  
   * **Semantic Drift Detection:** To capture more than just new words, the system will incorporate metrics that specifically track **vocabulary drift** (new terms), **structural drift** (new syntactic patterns), and **semantic drift** (new meanings for existing words) over time.25

The output of this stage is a ranked list of candidate neologisms and novel constructions, which are then passed to the second stage for evaluation of their "memetic fitness."

### **2.2 Stage 2: Canonization via a Computational Model of Norm Adoption**

The second stage acts as the "selection" mechanism, determining when a candidate neologism has achieved sufficient stability and utility to be formally integrated into the Reference Transcriptome. This process is guided by computational models of language change and sociolinguistics, which treat linguistic evolution as a process of variant adoption and norm formation within a population of speakers.1

To quantify the "memetic fitness" of a neologism, a composite metric called the Canonization Score (CS) is defined:

CS=w1​⋅log(Frequency)+w2​⋅Dispersion+w3​⋅Utility

Where:

* **Frequency:** The raw count of the expression over a given time window. This is the most basic measure of adoption.  
* **Dispersion:** A measure of how widely the expression is used across different authors, topics, or social contexts within the corpus. An expression used by many different groups is more likely to be a genuine emerging norm than an idiosyncratic quirk of a single author or community.  
* **Utility:** A functional measure of the expression's value. This can be proxied by its ability to improve performance on downstream NLP tasks (e.g., does recognizing this new idiom improve sentiment analysis accuracy?) or by its information-theoretic value (e.g., does it allow for more compressed or effective communication of a complex idea?).

The full protocol operates as follows: Candidate neologisms from Stage 1 are tracked over time. Their Canonization Score is updated with each new batch of corpus data. When an expression's CS crosses a pre-defined, empirically validated threshold, it is considered to have become a stable norm. At this point, it undergoes **lexicalization**—the process of becoming an established part of the language's lexicon—and is formally added to the Reference Transcriptome and the CxG Constructicon as a new canonical unit.29

This two-stage protocol treats the Reference Transcriptome not as a static database to be manually curated, but as a dynamic, evolving cultural artifact. The process of language change is not merely about frequency; it is a complex interplay of cognitive factors, social dynamics, and communicative function.27 By incorporating

Dispersion and Utility into the Canonization Score, the system moves beyond a simple statistical model to a computational model of *cultural evolution*. An expression survives and becomes "canon" not just because it is used often, but because it is useful and it successfully spreads through the linguistic community. This directly confronts and models the evolutionary nature of language.

| Stage | Description | Key Mechanisms | Metrics / Triggers |
| :---- | :---- | :---- | :---- |
| **0\. Canonical State** | Expression is a stable, recognized part of the language. | Stored in Reference Transcriptome and Constructicon. | N/A |
| **1\. Emergence (Variation)** | A novel expression appears in the input stream. | Unsupervised Novelty/Anomaly Detection. | High "oddballness" score; outlier status in density model; detected semantic drift. |
| **2\. Propagation (Selection)** | The expression is tracked for usage patterns and fitness. | Canonization Score calculation over time. | Monitored for changes in Frequency, Dispersion, and Utility. |
| **3\. Canonization (Fixation)** | The expression becomes a stable norm and is integrated. | Formal addition to the Reference Transcriptome/Constructicon. | Canonization Score exceeds the established threshold. |
| *Table 2: The Lifecycle of a Linguistic Norm in ARC-EVOLUTION. This table provides a step-by-step summary of the dynamic adaptation protocol, from the emergence of a novel expression to its potential canonization as a stable linguistic norm.* |  |  |  |

## **III. A Framework for "Reverse Transcription": Safe and Targeted Model Self-Improvement**

This section details a closed-loop system for the Chimera-1 agent to identify its own high-value generated expressions and safely integrate them back into its core model weights. This process is analogous to a retrovirus using the enzyme reverse transcriptase to write its genetic information into the host's DNA, thereby becoming a permanent part of the genome. For this process to be beneficial rather than pathological, it must be highly controlled.

### **3.1 Identifying High-Value Generated Transcripts: The "LLM-as-a-Judge"**

The first step is to enable Chimera-1 to self-assess the quality of its own linguistic productions. To achieve this, the framework will implement a **Self-Rewarding Language Model** architecture.30 In this paradigm, the agent is fine-tuned not only to follow instructions but also to evaluate its own performance, effectively becoming its own reward model.30

After generating a response, the model will be fed a second, evaluative prompt that instructs it to score its own output against a multi-point rubric.30 This rubric will be carefully designed to capture dimensions of expressive value relevant to the project's goals, including:

* **Task Success:** Did the expression successfully and accurately achieve the specified communicative goal?  
* **Novelty & Creativity:** Is the expression non-trivial, original, and elegant? This can be scored using the computational metrics developed in Sections II and IV.  
* **Efficiency & Conciseness:** Does the expression convey the intended meaning in a clear and compact manner?  
* **External Feedback:** Where available, does the expression correlate with positive user interaction signals, such as explicit acceptance, low rates of clarification requests, or successful task completion?

Expressions that receive a high self-reward score are flagged as high-value "transcripts" and become candidates for "reverse transcription."

### **3.2 The Reverse Transcriptase: Stabilized Model Editing**

The core challenge of self-improvement is creating a safe and reliable mechanism for writing a novel, high-value expressive pattern back into the LLM's parameters without causing unintended side effects or degrading its general capabilities.31

The core technology for this process will be a state-of-the-art "locate-and-edit" model editing technique. The leading candidates are **ROME (Rank-One Model Editing)** and **MEMIT (Mass-Editing Memory in a Transformer)**.32 These methods function by first using causal tracing to identify the specific Multi-Layer Perceptron (MLP) modules within the transformer architecture responsible for storing a particular factual association. They then apply a targeted, low-rank update to the weights of these modules to modify or insert the new knowledge.34

However, a naive application of model editing is fraught with peril. Recent research has demonstrated that making sequential edits to an LLM can lead to **gradual and catastrophic forgetting**, a phenomenon where the model's performance on unrelated tasks suddenly collapses after a certain number of edits.32 This is a critical failure mode that must be rigorously prevented.

To ensure the stability of the reverse transcription process, the following safeguards will be implemented:

1. **Use of Stabilized Implementations:** The system will specifically use **r-ROME**, a revised and more stable implementation of the ROME algorithm. Research has shown that r-ROME corrects irregularities in the original implementation that were responsible for "disabling edits" that caused model collapse, enabling large-scale sequential editing without this failure mode.38  
2. **Explicit Regularization:** The editing process will be regularized using the **RECT (RElative Change in weighT)** method.31 RECT constrains the complexity of the weight update by analyzing the relative change each parameter undergoes. It identifies the most "principal" components of the edit and preserves them, while zeroing out "minor" changes that are more likely to contribute to overfitting on the new fact. This has been shown to mitigate negative side effects and preserve the model's general abilities.31

### **3.3 The Full Reverse Transcription Protocol**

The complete, safeguarded protocol for model self-improvement is as follows:

1. Chimera-1 generates an expression in response to a prompt.  
2. The internal "LLM-as-a-Judge" module evaluates the generated expression and assigns it a reward score based on the predefined rubric.  
3. If the score exceeds a high threshold, the expression is identified as a high-value "transcript" worthy of integration.  
4. The novel expressive pattern (e.g., a new compositional construction, a useful turn of phrase) is formalized into a target representation.  
5. The r-ROME/MEMIT editor, regularized with RECT, is invoked to compute the minimal, stable weight update required to integrate this new pattern into the model's parameters.  
6. **Verification Step:** Before the change is permanently committed, the newly edited model is run against a comprehensive suite of benchmark tasks to ensure there has been no degradation in its general performance or previously edited knowledge.  
7. If the verification suite passes, the edit is committed to the model's baseline weights. If it fails, the edit is discarded, and the failure case is logged for analysis to further refine the editing and verification process.

This protocol reframes model self-improvement from a risky, open-ended process of unchecked mutation into a controlled, **homeostatic** one. The primary goal is not just to add new knowledge but to do so while actively preserving the functional integrity and health of the entire system. The "LLM-as-a-Judge" acts as the selection pressure, ensuring only high-quality "mutations" are considered. The combination of r-ROME and RECT acts as the high-fidelity DNA repair and replication machinery, ensuring the integration process itself is stable and does not corrupt the rest of the genome. The final verification step is the ultimate safeguard against pathology. This transforms "reverse transcription" from a potentially destructive force into a safe, robust, and reliable mechanism for adaptation and growth.

## **IV. An Experimental Design for "Junk DNA": Characterizing the Long Tail of Creativity**

This section proposes a formal experiment to analyze the "dark matter" of language—the vast space of unique, non-recurring, but valid and creative expressions that lie beyond the canonical transcriptome. In the genomic analogy, this corresponds to "junk DNA," which was once thought to be useless but is now understood to play crucial roles in regulation and providing the raw material for evolutionary innovation.

### **4.1 Defining the Object of Study: The Creative Long Tail**

The "creative long tail" is defined as the set of linguistic expressions that are novel, meaningful, and potentially valuable, but which do not appear with sufficient frequency to be canonized into the Reference Transcriptome.39 This is the domain of poetic metaphor, witty turns of phrase, novel scientific ideas, and imaginative descriptions. The ability to generate and comprehend expressions in this space is a hallmark of deep linguistic intelligence and a key differentiator between mere pattern replication and true creativity.

### **4.2 Experimental Protocol: A Blind, Multi-Metric Comparative Study**

To rigorously compare the creative text generation capabilities of the Chimera-1 agent against human experts, a **blind comparison study** will be conducted. This methodology is the gold standard for evaluating subjective qualities like creativity, as it minimizes bias.40

* **Participants:** Two groups of human experts will be recruited: (1) a cohort of generators (e.g., creative writers, senior researchers in a specific domain) and (2) a panel of expert judges to evaluate the generated texts.  
* **Task:** The generators (both human and Chimera-1) will be given a series of creative prompts designed to elicit responses from the long tail. Examples include abstract prompts ("Describe the color of jealousy") or constrained prompts ("Write a one-sentence story using the words 'algorithm,' 'shadow,' and 'nostalgia'").  
* **Anonymization and Standardization:** All generated outputs will be standardized into a uniform format and fully anonymized. The expert judges will have no knowledge of the origin (human or AI) of any given text, ensuring a fair, "apples-to-apples" comparison.40

### **4.3 A Multi-Faceted Creativity Assessment Framework**

A simplistic, single score for "creativity" is insufficient. To build a rich, multi-dimensional "creativity fingerprint" for both human and AI generators, the experiment will employ a combination of subjective human ratings and objective computational metrics.

**A) Subjective Human-Rated Metrics:** The expert judges will rate each generated text on a 1-10 scale for several dimensions, based on established practices in creativity assessment 40:

* **Novelty:** How original, surprising, and unprecedented is the expression?  
* **Feasibility/Meaningfulness:** How coherent, understandable, and non-nonsensical is the expression? This is a crucial control to filter out creative-sounding gibberish.  
* **Excitement/Aesthetics:** How engaging, thought-provoking, or beautiful is the expression? This captures the affective impact of creativity.  
* **Overall Quality:** An aggregate score of the expression's creative merit.

**B) Objective Computational Metrics:** In parallel, all texts will be programmatically scored using a battery of computational measures designed to quantify different facets of creativity:

* **The DeepCreativity Measure:** This comprehensive framework, based on Margaret Boden's theory of creativity, will provide three distinct scores 41:  
  * **Value:** How well does the text conform to the stylistic patterns of a high-quality reference corpus? (Measured using the discriminator of a Generative Adversarial Network).  
  * **Novelty:** How much does the text's style deviate from the established norm? (Measured using a style classifier).  
  * **Surprise:** How much would the text force a predictive language model to update its internal weights? (Measured by quantifying the model's influence).  
* **Information-Theoretic Measures:** To quantify the "surprisal" of lexical choices, the system will calculate metrics such as the Kullback-Leibler (KL) divergence between the context distributions of words and their compositions.39

### **4.4 Hypotheses and Analysis**

The experiment is designed to test several key hypotheses about the nature of human versus machine creativity:

* **H1 (Novelty-Feasibility Trade-off):** It is hypothesized that Chimera-1 will score significantly higher on the Novelty metric than human experts, but will score lower on Feasibility/Meaningfulness. This would reflect a known pattern where AI-generated ideas can be more "out-of-the-box" but also less grounded in reality or common sense.40  
* **H2 (Divergent Creative Fingerprints):** It is hypothesized that human and AI creativity will exhibit different computational profiles. For instance, the highest-rated human texts may show a strong correlation with the "Surprise" and "Value" components of the DeepCreativity measure, while the highest-rated AI texts may be better characterized by raw "Novelty" and information-theoretic surprisal.  
* **H3 (Characterizing the Long Tail):** By analyzing the linguistic and semantic features of the highest-rated creative outputs from both groups, the study will build a descriptive model of the "junk DNA" of language, identifying the distinct generative priors and creative strategies employed by human and machine intelligence.

This experimental design moves the investigation beyond a simple Turing test or a horse race between human and AI. The goal is not just to ask, "Is it creative?" but to ask, "**How is it creative?**" By running a blind human evaluation in parallel with a battery of objective computational metrics, the study can triangulate the nature of creativity itself. Correlating the subjective ratings from human judges with the objective scores from the computational framework will allow for the construction of a quantitative, predictive model of linguistic creativity. This will enable a characterization of the "Junk DNA" not as a random void, but as a space with its own discoverable structure and principles, and a deeper understanding of how humans and AI navigate that space differently.

| Metric | Type | Description | Source/Method |
| :---- | :---- | :---- | :---- |
| **Novelty** | Subjective | How original and unprecedented is the expression? | Rated by human expert judges 40 |
| **Feasibility / Meaningfulness** | Subjective | How coherent and understandable is the expression? | Rated by human expert judges 40 |
| **Excitement / Aesthetics** | Subjective | How engaging, surprising, or beautiful is the expression? | Rated by human expert judges 40 |
| **Overall Quality** | Subjective | An aggregate score of the expression's creative merit. | Rated by human expert judges 40 |
| **Value** | Objective | Conformity to high-quality stylistic patterns. | DeepCreativity (GAN Discriminator) 41 |
| **Novelty (Computational)** | Objective | Deviation from normative stylistic patterns. | DeepCreativity (Style Classifier) 41 |
| **Surprise** | Objective | The expression's influence on a predictive model. | DeepCreativity (Generative Model Influence) 41 |
| **Lexical Surprisal** | Objective | The unexpectedness of word choices in context. | Information Theory (KL-Divergence) 39 |
| *Table 3: The ARC-EVOLUTION Creativity Assessment Matrix. This table provides a comprehensive, at-a-glance view of the rigorous, multi-faceted evaluation protocol for the creativity experiment, combining subjective human judgment with objective computational analysis.* |  |  |  |

## **Synthesis and Future Directions**

The four research programs detailed in this proposal—Transcriptional Splicing, Dynamic Adaptation, Reverse Transcription, and Junk DNA Analysis—are not independent initiatives. They are deeply interconnected components of a single, cohesive system designed to model a living linguistic genome. The compositional expressions generated by the **Splicing** model (I) are the very objects that are evaluated for quality by the **Reverse Transcription** judge (III). Novel expressions that emerge and prove their fitness through the **Adaptation** protocol (II) are formally added to the Constructicon, expanding the generative repertoire of the Splicing model. Finally, the **Creativity** experiment (IV) provides crucial insights into the nature of high-value, long-tail expressions, which can be used to refine the reward models and generative priors for all other components, creating a virtuous cycle of improvement.

With the implementation of ARC-EVOLUTION, the genomic analogy is fully realized. The system will possess a heritable genome (the Constructicon), a mechanism for dynamic and compositional expression (splicing), a process for adapting to a changing environment (evolution), a safe method for incorporating beneficial mutations back into the germline (reverse transcription), and a vast reservoir of latent potential for innovation (the creative long tail).

The successful completion of this research will mark a critical step forward in the development of truly intelligent, adaptable, and creative linguistic agents. It moves beyond static, corpus-based models to a dynamic system that learns, evolves, and improves through its own expressive experience, laying the groundwork for the next generation of artificial intelligence.

#### **Works cited**

1. COMPUTATIONAL MODELS OF LANGUAGE EVOLUTION Ergashova Zarnigor Khasanjon kizi Teacher of English for specific purposes (ESP) Koka \- Oriens.uz, accessed July 9, 2025, [https://oriens.uz/media/journalarticles/108\_Ergashova\_Zarnigor\_Khasanjon\_kizi\_708-714.pdf](https://oriens.uz/media/journalarticles/108_Ergashova_Zarnigor_Khasanjon_kizi_708-714.pdf)  
2. We Call DNA a Language. Is It? \- proto.life, accessed July 9, 2025, [https://proto.life/2020/12/we-call-dna-a-language-is-it/](https://proto.life/2020/12/we-call-dna-a-language-is-it/)  
3. On the Analogy Between DNA and Language \- The BioLogos Forum, accessed July 9, 2025, [https://discourse.biologos.org/t/on-the-analogy-between-dna-and-language/50980](https://discourse.biologos.org/t/on-the-analogy-between-dna-and-language/50980)  
4. Unlocking Predicate Argument Structure \- Number Analytics, accessed July 9, 2025, [https://www.numberanalytics.com/blog/predicate-argument-structure-linguistic-theory](https://www.numberanalytics.com/blog/predicate-argument-structure-linguistic-theory)  
5. U4 NLP Notes \- Predicate Argument Structure \- Scribd, accessed July 9, 2025, [https://www.scribd.com/document/763618013/u4-nlp-notes](https://www.scribd.com/document/763618013/u4-nlp-notes)  
6. (PDF) Extraction of Predicate-Argument Structures from Texts \- ResearchGate, accessed July 9, 2025, [https://www.researchgate.net/publication/2823743\_Extraction\_of\_Predicate-Argument\_Structures\_from\_Texts](https://www.researchgate.net/publication/2823743_Extraction_of_Predicate-Argument_Structures_from_Texts)  
7. arXiv:2501.14528v3 \[cs.CL\] 20 Apr 2025, accessed July 9, 2025, [https://arxiv.org/pdf/2501.14528?](https://arxiv.org/pdf/2501.14528)  
8. Mastering Idiom Detection in Computational Linguistics \- Number Analytics, accessed July 9, 2025, [https://www.numberanalytics.com/blog/mastering-idiom-detection-computational-linguistics](https://www.numberanalytics.com/blog/mastering-idiom-detection-computational-linguistics)  
9. Can NLP models understand idioms or metaphors? \- Zilliz Vector Database, accessed July 9, 2025, [https://zilliz.com/ai-faq/can-nlp-models-understand-idioms-or-metaphors](https://zilliz.com/ai-faq/can-nlp-models-understand-idioms-or-metaphors)  
10. arXiv:2501.14528v3 \[cs.CL\] 20 Apr 2025, accessed July 9, 2025, [https://arxiv.org/pdf/2501.14528](https://arxiv.org/pdf/2501.14528)  
11. Rhetorical structure theory \- Wikipedia, accessed July 9, 2025, [https://en.wikipedia.org/wiki/Rhetorical\_structure\_theory](https://en.wikipedia.org/wiki/Rhetorical_structure_theory)  
12. Using Automatically Labelled Examples to Classify Rhetorical ..., accessed July 9, 2025, [https://homepages.inf.ed.ac.uk/alex/pubs/jnle.rhetorical.html](https://homepages.inf.ed.ac.uk/alex/pubs/jnle.rhetorical.html)  
13. Hybrid Symbolic-Deep Learning Models for Logical ... \- ijrpr, accessed July 9, 2025, [https://ijrpr.com/uploads/V6ISSUE5/IJRPR45045.pdf](https://ijrpr.com/uploads/V6ISSUE5/IJRPR45045.pdf)  
14. Hybrid Symbolic-Neural Architectures for Explainable Artificial Intelligence in Decision-Critical Domains \- ResearchGate, accessed July 9, 2025, [https://www.researchgate.net/publication/391715731\_Hybrid\_Symbolic-Neural\_Architectures\_for\_Explainable\_Artificial\_Intelligence\_in\_Decision-Critical\_Domains](https://www.researchgate.net/publication/391715731_Hybrid_Symbolic-Neural_Architectures_for_Explainable_Artificial_Intelligence_in_Decision-Critical_Domains)  
15. Neuro-symbolic AI \- Wikipedia, accessed July 9, 2025, [https://en.wikipedia.org/wiki/Neuro-symbolic\_AI](https://en.wikipedia.org/wiki/Neuro-symbolic_AI)  
16. Evolutionary & Hybrid AI \- Vrije Universiteit Brussel, accessed July 9, 2025, [https://ehai.ai.vub.ac.be/computational-construction-grammar.html](https://ehai.ai.vub.ac.be/computational-construction-grammar.html)  
17. Construction Grammar and Language Models (Chapter 22\) \- Cambridge University Press, accessed July 9, 2025, [https://www.cambridge.org/core/books/cambridge-handbook-of-construction-grammar/construction-grammar-and-language-models/131443D7CBBC87CB6A4D2A85CC531851](https://www.cambridge.org/core/books/cambridge-handbook-of-construction-grammar/construction-grammar-and-language-models/131443D7CBBC87CB6A4D2A85CC531851)  
18. Explaining pretrained language models' understanding of linguistic structures using construction grammar \- Frontiers, accessed July 9, 2025, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1225791/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1225791/full)  
19. A Theoretical Analysis of Compositional Generalization in Neural ..., accessed July 9, 2025, [https://www.arxiv.org/pdf/2505.02627](https://www.arxiv.org/pdf/2505.02627)  
20. The process of Novelty Detection in a text stream. | Download ..., accessed July 9, 2025, [https://www.researchgate.net/figure/The-process-of-Novelty-Detection-in-a-text-stream\_fig1\_259584402](https://www.researchgate.net/figure/The-process-of-Novelty-Detection-in-a-text-stream_fig1_259584402)  
21. Textual Novelty Detection | Towards Data Science, accessed July 9, 2025, [https://towardsdatascience.com/textual-novelty-detection-ce81d2e689bf/](https://towardsdatascience.com/textual-novelty-detection-ce81d2e689bf/)  
22. Anomaly detection \- Wikipedia, accessed July 9, 2025, [https://en.wikipedia.org/wiki/Anomaly\_detection](https://en.wikipedia.org/wiki/Anomaly_detection)  
23. Oddballness: universal anomaly detection with language models \- ACL Anthology, accessed July 9, 2025, [https://aclanthology.org/2025.coling-main.183/](https://aclanthology.org/2025.coling-main.183/)  
24. Oddballness: universal anomaly detection with language models ..., accessed July 9, 2025, [https://aclanthology.org/2025.coling-main.183](https://aclanthology.org/2025.coling-main.183)  
25. Characterizing and Measuring Linguistic Dataset Drift \- ACL Anthology, accessed July 9, 2025, [https://aclanthology.org/2023.acl-long.498/](https://aclanthology.org/2023.acl-long.498/)  
26. Lexical Semantic Change through Large Language Models: a Survey \- ResearchGate, accessed July 9, 2025, [https://www.researchgate.net/publication/381333580\_Lexical\_Semantic\_Change\_through\_Large\_Language\_Models\_a\_Survey](https://www.researchgate.net/publication/381333580_Lexical_Semantic_Change_through_Large_Language_Models_a_Survey)  
27. Computational Approaches to the Study of Language Change, accessed July 9, 2025, [https://langev.com/pdf/baker08languageChange.pdf](https://langev.com/pdf/baker08languageChange.pdf)  
28. Computational Sociolinguistics: A Survey \- MIT Press Direct, accessed July 9, 2025, [https://direct.mit.edu/coli/article/42/3/537/1536/Computational-Sociolinguistics-A-Survey](https://direct.mit.edu/coli/article/42/3/537/1536/Computational-Sociolinguistics-A-Survey)  
29. Lexicalization. Lexicalisation Decoded\! | by Riaz Laghari | Medium, accessed July 9, 2025, [https://medium.com/@riazleghari/lexicalization-5cc4c5eadd20](https://medium.com/@riazleghari/lexicalization-5cc4c5eadd20)  
30. Meta's “Self-Rewarding Language Models” paper explained | by ..., accessed July 9, 2025, [https://medium.com/@smitshah00/metas-self-rewarding-language-models-paper-explained-38b5c6ee9dd3](https://medium.com/@smitshah00/metas-self-rewarding-language-models-paper-explained-38b5c6ee9dd3)  
31. Model Editing Harms LLMs \- Jia-Chen Gu, accessed July 9, 2025, [https://jasonforjoy.github.io/Model-Editing-Hurt/](https://jasonforjoy.github.io/Model-Editing-Hurt/)  
32. Model Editing at Scale leads to Gradual and Catastrophic Forgetting \- arXiv, accessed July 9, 2025, [https://arxiv.org/html/2401.07453v3](https://arxiv.org/html/2401.07453v3)  
33. Mass-Editing Memory in a Transformer \- OpenReview, accessed July 9, 2025, [https://openreview.net/forum?id=MkbcAHIYgyS](https://openreview.net/forum?id=MkbcAHIYgyS)  
34. Mass Editing Memory in a Transformer, accessed July 9, 2025, [https://memit.baulab.info/](https://memit.baulab.info/)  
35. Locating and Editing Factual Associations in GPT, accessed July 9, 2025, [https://rome.baulab.info/](https://rome.baulab.info/)  
36. Model Editing at Scale leads to Gradual and Catastrophic Forgetting \- ACL Anthology, accessed July 9, 2025, [https://aclanthology.org/2024.findings-acl.902/](https://aclanthology.org/2024.findings-acl.902/)  
37. The Fall of ROME: Understanding the Collapse of LLMs in Model Editing \- ACL Anthology, accessed July 9, 2025, [https://aclanthology.org/2024.findings-emnlp.236.pdf](https://aclanthology.org/2024.findings-emnlp.236.pdf)  
38. Rebuilding ROME : Resolving Model Collapse during Sequential Model Editing \- arXiv, accessed July 9, 2025, [https://arxiv.org/html/2403.07175v2](https://arxiv.org/html/2403.07175v2)  
39. Understanding and Quantifying Creativity in Lexical Composition, accessed July 9, 2025, [https://homes.cs.washington.edu/\~yejin/Papers/emnlp13\_creativity.pdf](https://homes.cs.washington.edu/~yejin/Papers/emnlp13_creativity.pdf)  
40. The Illusion of Machine Creativity: Recombination versus ... \- Medium, accessed July 9, 2025, [https://medium.com/@adnanmasood/the-illusion-of-machine-creativity-recombination-versus-understanding-in-the-age-of-ai-15ace83e055a](https://medium.com/@adnanmasood/the-illusion-of-machine-creativity-recombination-versus-understanding-in-the-age-of-ai-15ace83e055a)  
41. DeepCreativity: Measuring Creativity with Deep ... \- Mirco Musolesi, accessed July 9, 2025, [https://www.mircomusolesi.org/papers/ia22\_deepcreativity.pdf](https://www.mircomusolesi.org/papers/ia22_deepcreativity.pdf)