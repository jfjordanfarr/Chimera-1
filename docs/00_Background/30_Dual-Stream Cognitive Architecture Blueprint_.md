

# **ARC-SPLICER V2: A Dual-Stream Cognitive Architecture for Linguistic Regulation**

## **1\. Introduction: The Regulatory Genome as a Model for Language Processing**

The history of science is marked by periodic re-evaluations of concepts once dismissed as unimportant. A prime example from molecular biology is the evolving understanding of introns, segments of pre-mRNA that are removed, or "spliced out," during the maturation of a messenger RNA (mRNA) molecule. Initially characterized as "junk DNA," these non-coding sequences were thought to be evolutionary remnants with no significant function.1 However, decades of research have overturned this view, revealing that introns are, in fact, critical components of the gene regulatory network. They contain enhancers and silencers, influence alternative splicing to create protein diversity, and are integral to the efficiency of transcription and translation.3 This intellectual journey—from dismissing non-coding regions as noise to recognizing them as a sophisticated control layer—offers a powerful and instructive parallel for the field of Natural Language Processing (NLP).

This report posits that NLP is at a similar inflection point. Current models, while highly proficient in syntactic and semantic analysis, often treat pragmatic, affective, and other contextual elements of language as secondary phenomena—a form of "linguistic junk" to be filtered in pursuit of core propositional content. This focus on literal meaning creates a significant gap in capability, as models struggle to interpret nuance, implied intent, and the rich tapestry of meta-information that governs human communication.7 Just as genomics required a new paradigm to understand the regulatory genome, NLP requires a new architecture to process the regulatory components of language. This report introduces ARC-SPLICER V2, a bio-inspired cognitive architecture designed to address this fundamental challenge.

### **1.1 Revisiting the "Junk DNA" Fallacy in Linguistics: The Functional Role of Non-Semantic Text**

The analogy between genetic and linguistic "junk" is not merely superficial. The initial dismissal of introns stemmed from a protein-centric view of the genome; if a DNA sequence did not code for a protein, its utility was questioned. Similarly, traditional NLP has been built on a foundation of semantics-centric processing; if a word or phrase does not contribute directly to the propositional, truth-conditional meaning of a sentence, its importance is often minimized. This approach overlooks the vast landscape of language use that extends beyond literal definitions.8

Research in pragmatics, the study of how context contributes to meaning, has long established that what is *not* explicitly stated is often as important as what is. Phenomena such as implicature, presupposition, and speech acts are central to human understanding.7 Yet, NLP models have historically struggled with these concepts, leading to a call for pragmatics to "take center stage" in the evaluation of modern language systems.10 The failure to account for these elements is not a minor limitation; it is a systemic deficiency that prevents models from achieving true human-like communicative competence. Just as introns play a crucial role in regulating gene expression at virtually every stage of mRNA maturation—from transcription initiation to nuclear export and stability 11—linguistic meta-information regulates the interpretation, intent, and impact of semantic content.

The narrative of scientific correction in genomics provides a compelling justification for a fundamental re-evaluation of our approach in NLP. The discovery of intron function was not an incremental improvement but a paradigm shift that revealed a hidden layer of complexity and control within the genome. This project is motivated by the conviction that a similar hidden layer exists within language and that building architectures to explicitly model it is the next logical step in the evolution of artificial intelligence.

### **1.2 A New Paradigm: Drawing Parallels Between Genetic and Linguistic Regulation**

The ARC-SPLICER V2 architecture is founded on a direct and operationalizable analogy between specific genetic regulatory mechanisms and their linguistic counterparts. This bio-inspired framework provides a principled basis for designing a system that can disentangle and leverage the different functional layers of text.

* **Enhancers and Silencers:** In genetics, introns harbor regulatory elements known as enhancers and silencers, which increase or decrease the rate of gene transcription, respectively.3 This mechanism has a clear parallel in language. Linguistic "boosters" or "intensifiers" (e.g.,  
  *absolutely*, *certainly*, *no doubt*) function as enhancers, strengthening the illocutionary force of an utterance. Conversely, "hedges" (e.g., *perhaps*, *maybe*, *I think*, *it seems*) function as silencers, weakening the speaker's commitment to the proposition and signaling uncertainty.13 An architecture that can identify and quantify these elements can better gauge the true confidence and intent behind a statement.  
* **Alternative Splicing:** This process allows a single gene to generate multiple distinct mRNA transcripts by selectively including or excluding certain exons. This is a primary source of proteomic diversity in higher eukaryotes, enabling the creation of over 90,000 different proteins from approximately 25,000 human genes.4 The linguistic analog is the way pragmatic context allows a single semantic statement to serve multiple functions. The utterance, "It's cold in here," has a consistent semantic core (the temperature of the room is low). However, depending on the "splicing" provided by the context—the speaker, the listener, the setting, the tone of voice—it can be interpreted as a simple observation, a polite request to close a window, a complaint about the building's heating, or an excuse to move closer to someone. The "exonic" semantic content is stable, but the "intronic" pragmatic information dictates its final functional form.  
* **Co-transcriptional Processing:** A pivotal insight from modern genetics is that splicing is not necessarily a post-processing step that occurs after a gene has been fully transcribed. Much of it happens co-transcriptionally, as the pre-mRNA strand is being synthesized by RNA polymerase II.16 For gigantic genes with massive introns, this co-transcriptional splicing is essential to prevent the nascent RNA from becoming entangled and halting transcription altogether.16 This suggests that an effective linguistic architecture should not treat the separation of semantic and pragmatic information as a sequential filtering task (i.e., Text \-\> Pragmatic Filter \-\> Semantic Processor). Instead, it points toward a more integrated, parallel processing model where both streams of information are derived simultaneously from the input, reflecting their intertwined nature in human cognition.

### **1.3 Introducing ARC-SPLICER V2: A Bio-Inspired Architecture for Separating and Leveraging Semantic and Pragmatic Information**

Based on this new paradigm, the mission of ARC-SPLICER V2 is to redesign the system's front-end processing to perform "linguistic splicing." The architecture will ingest raw text and, through a unified co-processing mechanism, separate it into two functionally distinct and synchronized data streams:

1. **The Exonic Stream:** This stream consists of the core propositional and semantic content of the input. It is conceptually "purified" of the pragmatic and affective modulators that can introduce ambiguity for purely logical reasoning systems. This stream is destined for the GATformer, the system's primary semantic and logical reasoning engine.  
2. **The Intronic Stream:** This stream consists of a structured representation of the pragmatic, affective, discourse, and other meta-informational cues present in the input. It does not contain semantic content itself but rather information *about* the semantic content. This stream is destined for a new ARC-REGULATION module, which will use this information to control and modulate the system's final behavior and output generation.

By explicitly separating "what is said" (Exonic) from "how it is said and why" (Intronic), ARC-SPLICER V2 aims to create a more robust, nuanced, and contextually aware AI system, marking a significant step away from the limitations of purely semantics-focused models.

## **2\. The ARC-SPLICER V2 Cognitive Architecture**

The design of ARC-SPLICER V2 is a direct implementation of the biological principles outlined in the preceding section. It departs from traditional pipeline-based NLP systems in favor of a co-processing model that simultaneously derives semantic content and regulatory meta-information from a single, shared representation of the input text. This section provides a detailed blueprint of the architecture, its core components, and the flow of information between them.

### **2.1 The Dual-Stream Principle: A Co-Processing Model**

The architectural design is fundamentally guided by the biological insight of co-transcriptional splicing.16 A simplistic approach might involve a pipeline where a "pragmatics filter" first removes intronic elements before passing the remaining text to a semantic model. Such a design is inefficient and conceptually flawed, as it assumes that pragmatic and semantic information can be cleanly separated without mutual influence.

Instead, ARC-SPLICER V2 employs a co-processing model. A single, powerful encoder processes the raw input text once, generating a rich, contextualized representation of the entire utterance. From this shared representation, two specialized downstream components, or "heads," work in parallel to derive the Exonic and Intronic streams. This approach is not only more computationally efficient, as it avoids redundant encoding, but it is also more faithful to the biological model where regulatory decisions are made in concert with the primary process of transcription.

The input to the ARC-SPLICER V2 system is a raw text sequence. The output is a pair of synchronized data structures:

1. A sequence of contextual token embeddings, which constitutes the **Exonic Stream**. This stream has the same length as the input sequence but is "purified," with regulatory elements neutralized.  
2. A single, fixed-dimension dense vector, the **Intronic Vector**, which represents the aggregated regulatory signals from the entire utterance and constitutes the **Intronic Stream**.

### **2.2 The Intronic/Exonic Classifier (IEC): A Multi-Task, Multi-Head Splicing Engine**

The heart of the ARC-SPLICER V2 architecture is the Intronic/Exonic Classifier (IEC). This module is responsible for performing the core "splicing" function. It is designed as a multi-task learning (MTL) system built upon a state-of-the-art Transformer-based encoder, such as RoBERTa or a similar model, which are known for their strong performance in a wide range of NLU tasks.19

The key mechanism enabling the separation of streams is the strategic use of the Multi-Head Attention mechanism inherent to Transformers.20 The "Attention is All You Need" paradigm demonstrated that a model can learn complex dependencies by allowing it to jointly attend to information from different representation subspaces at different positions.21 Subsequent research has shown that different attention heads can specialize in capturing distinct types of linguistic features, such as syntactic dependencies or semantic relationships.22 The IEC design leverages this property by explicitly training subsets of attention heads to specialize in identifying either "intronic" or "exonic" features within the text. This approach is inspired by models like the Multi-Head Attention Labeller (MHAL), which uses a multi-head mechanism to wire together word-level (sequence labeling) and sentence-level (classification) prediction tasks, enabling a fluid transfer of knowledge between compositional levels.24

To achieve this specialization, the IEC is structured with a shared Transformer backbone and two distinct, task-specific heads that are trained jointly:

1. **A Sequence Tagging Head:** This is a token-level classifier that operates on the final hidden state of each input token from the Transformer backbone. Its task is to assign a specific pragmatic or affective label to each token from a predefined "Linguistic Intron Taxonomy" (detailed in Section 3.1). For tokens that are part of the core semantic content, it assigns a null or "Outside" label. This head is directly responsible for identifying the location and type of all intronic elements in the text.  
2. **A Regulatory Vector Generation Head:** This head is responsible for creating the Intronic Stream. It takes as input only the final hidden states of the tokens that the Sequence Tagging Head has identified as "intronic." It then uses an aggregation mechanism (e.g., an attention-based pooling layer) to combine these representations into a single, fixed-dimension dense vector. This process is an application of the Multi-Task Label Embedding concept, where the goal is to learn a rich, semantic vector representation for the pragmatic "labels" present in the text, rather than treating them as simple one-hot categories.26 This resulting vector is the Intronic Vector.

This multi-task architecture ensures that the model learns not just to *identify* regulatory words but to understand their contribution to the overall pragmatic context of the utterance, encoding this understanding into a structured vector. This design choice enables a powerful form of emergent compositionality. A standard classifier can only map an input to a predefined class. The IEC, by producing a vector, can represent novel combinations of pragmatic markers. For example, if the model has seen "maybe" (signaling uncertainty) and "frankly" (signaling stance) separately, it learns to activate distinct dimensions in the regulatory vector for each. When it encounters the novel combination "frankly, maybe...", it can activate both dimensions simultaneously, creating a new vector that represents a "frankly uncertain" stance. The ARC-REGULATION module can then interpret this novel vector to produce a uniquely nuanced response, achieving a form of zero-shot pragmatic generalization.

### **2.3 The Exonic Stream: Routing Purified Semantic Embeddings to the GATformer**

The Exonic Stream is not generated as a new, shorter string of text. Instead, it is the complete sequence of final-layer hidden states from the IEC's shared Transformer backbone, passed on to the GATformer for downstream reasoning. However, this stream is "purified" based on the output of the IEC's Sequence Tagging Head.

The hidden state vectors corresponding to tokens that were tagged as intronic are algorithmically neutralized. This can be achieved through several methods, such as:

* **Masking:** Replacing the intronic token's embedding with a zero vector or a special \`\` token embedding.  
* **Down-weighting:** Scaling the magnitude of the intronic token's embedding by a factor close to zero.  
* **Averaging:** Replacing the intronic token's embedding with an average of its non-intronic neighbors.

This process effectively erases the direct contribution of the pragmatic markers from the sequence of embeddings that the GATformer receives. The hypothesis is that this purification will reduce ambiguity and noise, allowing the GATformer to operate more effectively on the core logical and semantic structure of the utterance, free from the complexities of pragmatic modulation.

### **2.4 The Intronic Stream: The Structured Regulatory Vector**

As described, the Intronic Stream is the dense vector produced by the IEC's Regulatory Vector Generation Head. This is a critical output of the system. The vector is not an undifferentiated "bag of pragmatics"; it is a structured representation where specific dimensions or subspaces are trained to correspond directly to the categories in the Linguistic Intron Taxonomy.

For instance, the vector might have dedicated dimensions for:

* **Epistemic Modality:** A value representing the degree of certainty or uncertainty.  
* **Deontic Modality:** A value representing the degree of obligation or permission.  
* **Affective Valence:** A value from \-1 (negative) to \+1 (positive) representing sentiment.  
* **Politeness Register:** A value indicating the level of formality or deference.

This structured vector serves as a rich, quantitative, and machine-readable summary of the entire pragmatic and affective context of the original utterance. It distills the complex, scattered linguistic cues into a single, actionable control signal.

### **2.5 The ARC-REGULATION Module: A Control System for Cognitive Modulation**

The final component in this new architecture is the ARC-REGULATION module. This module receives the Intronic Vector as its sole input. Its purpose is to act as a control system, translating the abstract features of the Intronic Vector into concrete modifications of the system's final output generation process. This concept is heavily influenced by research in Affective Computing, which seeks to build systems that can perceive, interpret, and respond to human emotions and affects, moving from purely logic-driven interaction to affect-sensitive communication.28

The ARC-REGULATION module itself can be implemented with varying levels of complexity, from a simple lookup table or rule-based system to a small, dedicated neural network. Its function is to map the state of the Intronic Vector to a set of control parameters. Examples of these control signals include:

* **Confidence Scaling:** A high value in an "uncertainty" dimension of the vector could trigger a command to lower the confidence score reported with the final answer.  
* **Template Selection:** A high value in a "politeness" dimension could select a more deferential and formal response template, while a high "negative affect" value could trigger an empathetic template.  
* **Hedge/Booster Insertion:** The module could instruct the text generator to prepend the output with a hedge ("It seems likely that...") or a booster ("It is certain that...") based on the vector's epistemic values.  
* **Behavioral Gating:** A strong "indirect question" signal could cause the system to bypass a direct answer and instead generate a clarifying question to resolve the ambiguity.

In essence, the ARC-REGULATION module uses the "how" and "why" from the Intronic Stream to intelligently shape the "what" that is ultimately communicated, enabling a far more sophisticated and socially aware interaction model.

## **3\. Methodology for Stream Separation and Regulation**

The successful implementation of the ARC-SPLICER V2 architecture hinges on a robust and principled methodology for training its core component, the Intronic/Exonic Classifier (IEC). This requires the development of a formal linguistic framework for what constitutes an "intron," a rigorous protocol for creating a high-quality annotated dataset based on this framework, and a sophisticated multi-task training regime to teach the model the complex task of linguistic splicing.

### **3.1 A Proposed Taxonomy of Linguistic Introns**

To move from the abstract concept of "linguistic introns" to a computationally tractable problem, a formal classification scheme is required. This taxonomy serves as the foundation for both the data annotation process and the design of the model's output layers. It must be comprehensive enough to capture a wide range of regulatory phenomena yet discrete enough to be reliably annotated. The proposed taxonomy synthesizes concepts from the fields of pragmatics, affective computing, and computational linguistics, with a particular focus on established schemes for classifying uncertainty and speech acts.7

The taxonomy is hierarchical, organizing specific linguistic markers into functional categories. This structure provides a clear specification for the sequence tagging task and defines the semantic meaning of the dimensions within the final Intronic Vector. The table below outlines the proposed taxonomy, which will form the basis of the annotation guidelines.

**Table 1: A Taxonomy of Linguistic Introns and their Regulatory Functions**

| Category (L1) | Sub-Category (L2) | Tag (for Annotation) | Linguistic Examples | Regulatory Function (Mapping to ARC-REGULATION) |
| :---- | :---- | :---- | :---- | :---- |
| **Epistemic** | Hedging / Uncertainty | EPI-HEDGE | *maybe, perhaps, might, could, I think, seems, appears, possibly, suggest* | DECREASE\_CONFIDENCE, TRIGGER\_CAUTIOUS\_TEMPLATE |
|  | Boosting / Certainty | EPI-BOOST | *definitely, certainly, absolutely, no doubt, clearly, obviously* | INCREASE\_CONFIDENCE, TRIGGER\_ASSERTIVE\_TEMPLATE |
|  | Doxastic Belief | EPI-DOXASTIC | *I believe, we assume, they claim, according to, reportedly* | ATTRIBUTE\_SOURCE, LOWER\_OBJECTIVE\_WEIGHT |
|  | Investigative | EPI-INVEST | *we examined, the study investigates, we analyze, research shows* | SET\_CONTEXT\_SCIENTIFIC, INCREASE\_EVIDENCE\_WEIGHT |
| **Deontic** | Obligation / Necessity | DEO-NECESSITY | *must, have to, need to, required, necessary* | FLAG\_CONSTRAINT, PRIORITIZE\_ACTION\_ITEM |
|  | Permission / Ability | DEO-PERMISSION | *can, may, able to, permitted, allowed* | FLAG\_CAPABILITY |
|  | Volition / Intent | DEO-INTENT | *will, want to, plan to, intend to, going to* | FLAG\_FUTURE\_COMMITMENT |
| **Affective** | Positive Sentiment | AFF-POS | *great, wonderful, excellent, love, amazing, fortunately* | SET\_SENTIMENT\_VALENCE(+), TRIGGER\_POSITIVE\_RESPONSE |
|  | Negative Sentiment | AFF-NEG | *terrible, awful, hate, poor, unfortunately, problem* | SET\_SENTIMENT\_VALENCE(-), TRIGGER\_EMPATHETIC\_RESPONSE |
|  | Emotional State | AFF-EMOTION | *happy, sad, angry, surprised, frustrated* | DETECT\_USER\_EMOTION, ADAPT\_INTERACTION\_STYLE |
| **Discourse** | Connectors (Contrast) | DISC-CONTRAST | *but, however, although, on the other hand, yet* | APPLY\_LOGICAL\_CONNECTOR(CONTRAST) |
|  | Connectors (Causation) | DISC-CAUSE | *because, since, therefore, as a result, so* | APPLY\_LOGICAL\_CONNECTOR(CAUSE) |
|  | Stance / Framing | DISC-STANCE | *honestly, frankly, personally, in my opinion* | FLAG\_SUBJECTIVE\_FRAME |
|  | Fillers / Pauses | DISC-FILLER | *well, um, uh, you know, like* | DETECT\_HESITATION, LOWER\_SPEECH\_FORMALITY |
| **Social** | Politeness / Deference | SOC-POLITE | *please, thank you, sorry to bother, if you don't mind, sir/ma'am* | ADOPT\_POLITE\_REGISTER, INCREASE\_DEFERENCE |
|  | Indirect Acts | SOC-INDIRECT | *(e.g., "It's cold in here" as a request)* | INFER\_IMPLICIT\_INTENT, TRIGGER\_CLARIFICATION |
|  | Conditionals | SOC-CONDITIONAL | *if, when, in case, provided that* | EVALUATE\_CONDITIONAL\_BRANCH |

### **3.2 Annotation Protocol and Corpus Curation**

A high-quality, manually annotated corpus is the single most critical prerequisite for training a reliable IEC model. The process for creating this corpus will be systematic and rigorous, adhering to established best practices in corpus linguistics and data annotation for machine learning.33 The methodology employed in creating specialized datasets like PragMaIMS, which focuses on pragmatic markers in political discourse, serves as a strong model for this effort.36

1. **Data Sourcing:** To ensure the model generalizes well, the source text will be sampled from a diverse range of domains and genres. This will include conversational data (e.g., dialogue transcripts), user-generated content (e.g., product reviews, social media posts), and formal text (e.g., news commentary, scientific articles). This diversity prevents the model from overfitting to the stylistic conventions of a single domain.  
2. **Annotation Guidelines:** A comprehensive annotation manual will be developed based on the taxonomy in Table 1\. This manual will provide clear definitions for each tag, positive and negative examples, and decision trees for ambiguous cases. A consistent and unambiguous set of guidelines is essential for achieving high inter-annotator agreement.  
3. **Annotation Tooling:** The annotation will be performed using a dedicated text annotation tool. Open-source options like Brat or Doccano are well-suited for this type of sequence labeling task, as they provide a user-friendly interface for assigning labels to spans of text.33  
4. **Quality Control and Inter-Annotator Agreement (IAA):** To ensure the reliability of the final "gold-standard" dataset, a multi-annotator process will be implemented. A subset of the data will be annotated independently by at least two trained annotators. The level of agreement will be measured using a standard statistical metric, such as Cohen's Kappa or Fleiss' Kappa. A target IAA score of \>0.80 will be set, indicating substantial agreement. All disagreements will be adjudicated by a senior linguist or researcher to resolve conflicts and refine the annotation guidelines. This iterative process of annotation, agreement measurement, and adjudication is crucial for producing a dataset of the quality needed for supervised model training.

### **3.3 Training the Intronic/Exonic Classifier (IEC)**

The IEC will be trained within a multi-task learning (MTL) framework, which has been shown to improve model generalization by allowing a model to leverage shared representations across related tasks.19

* **Model Framework:** The core of the IEC will be a pre-trained Transformer-based encoder (e.g., bert-base-uncased or roberta-base). On top of this shared backbone, the two task-specific heads described in Section 2.2 will be added: the Sequence Tagging Head and the Regulatory Vector Generation Head.  
* Composite Loss Function: The model will be trained by minimizing a composite loss function, Ltotal​, which is a weighted sum of the losses from the two individual tasks.  
  Ltotal​=α⋅Ltagging​+β⋅Lvector​  
  * Ltagging​: This will be a standard token-level Cross-Entropy Loss, calculated over the output of the Sequence Tagging Head against the ground-truth labels from the annotated corpus. To handle class imbalance (as some intronic tags will be much rarer than others), a weighted version of the cross-entropy loss or a more advanced technique like Focal Loss may be employed.  
  * Lvector​: The loss for the Regulatory Vector Generation Head is more complex. The ground-truth "target vector" for a given sentence will be constructed from its token-level annotations. For example, a simple approach would be to create a multi-hot vector where each dimension corresponds to a tag in the taxonomy and is set to 1 if that tag appears in the sentence. The loss would then be a Binary Cross-Entropy or similar loss between the predicted vector and this target. A more sophisticated approach would use a Mean Squared Error (MSE) or Cosine Similarity loss to encourage the generated vector to be close to a pre-defined target embedding for the set of pragmatic features present.  
  * α and β: These are hyperparameters that weight the contribution of each task's loss to the total loss. Their values will be tuned to balance the learning between the two tasks.  
* **Training Strategy:** The training will proceed via fine-tuning the pre-trained Transformer model on the newly curated, annotated corpus. Techniques such as gradual unfreezing will be explored, where the lower layers of the Transformer are initially frozen (to preserve their general language understanding) while the top layers and the task-specific heads are trained. The lower layers can then be unfrozen for end-to-end fine-tuning at a lower learning rate.37 This strategy often leads to more stable training and better final performance.

## **4\. Validation Framework and Performance Metrics**

Evaluating the success of ARC-SPLICER V2 requires a more sophisticated framework than standard NLP model evaluation. While traditional metrics like accuracy and F1-score are necessary for assessing sub-components, they are insufficient to capture the holistic goal of the architecture: to perform a clean and meaningful separation of linguistic streams that leads to improved downstream performance. Therefore, the validation framework is composed of two main pillars: a novel composite metric designed to measure the quality of the splicing process itself, and a series of downstream, task-based evaluations to measure the end-to-end impact of the architecture.

### **4.1 Defining "Separation Fidelity": A Novel Composite Metric**

To quantify the performance of the Intronic/Exonic Classifier (IEC), a new composite metric, **Separation Fidelity** (Φ), is proposed. This metric is designed to provide a single, interpretable score that reflects not only the accuracy of the intronic tagger but also the functional integrity of the two resulting streams. A high Φ score would indicate that the IEC is not just correctly identifying markers but is successfully isolating regulatory information into the Intronic Stream while preserving core meaning in the Exonic Stream.

The Separation Fidelity score is defined as a weighted average of three component scores:  
$$ \\Phi \= w\_1 \\cdot \\text{Purity}{\\text{Exonic}} \+ w\_2 \\cdot \\text{Richness}{\\text{Intronic}} \+ w\_3 \\cdot \\text{Accuracy}\_{\\text{Tagging}} $$  
where w1​,w2​,w3​ are weights that sum to 1, allowing for the tuning of the metric's focus.

* **Component 1: Accuracy\_Tagging:** This is the most straightforward component. It measures the performance of the IEC's Sequence Tagging Head. It will be calculated as the token-level macro F1-score on a held-out test set from the annotated corpus. This ensures that the model is correctly identifying and classifying the linguistic introns according to the defined taxonomy. This is a necessary baseline for the quality of the separation.  
* **Component 2: Purity\_Exonic:** This component evaluates how well the Exonic Stream maintains the essential semantic and propositional content of the original utterance after the intronic elements have been neutralized. A high purity score means that the splicing process did not inadvertently destroy or distort the core meaning.  
  * Methodology: The evaluation will use a standard NLU benchmark task that relies heavily on semantic understanding, such as Natural Language Inference (NLI) or Question Answering (QA). First, a high-performance, pre-trained model for this task will be run on the original, unaltered text of the benchmark's test set to establish a baseline performance score (e.g., NLI accuracy). Second, the IEC will process the same test set to generate the "purified" Exonic Streams (i.e., the sequences of embeddings with intronic tokens masked). The NLU task will then be run again, this time using these purified embeddings as input. The Purity\_Exonic score will be calculated as a function of the performance degradation, where a smaller drop in performance yields a higher purity score. For example:  
    $$ \\text{Purity}{\\text{Exonic}} \= \\frac{\\text{Performance}{\\text{Purified}}}{\\text{Performance}\_{\\text{Baseline}}} $$  
* **Component 3: Richness\_Intronic:** This component evaluates the quality and utility of the Intronic Stream. It measures how well the generated Intronic Vector captures the high-level pragmatic and affective properties of the original utterance. A high richness score indicates that the vector is a meaningful and predictive representation of the regulatory context.  
  * **Methodology:** This evaluation requires a separate, held-out dataset that has been annotated by human raters for sentence-level pragmatic attributes (e.g., a politeness score from 1-5, a sentiment rating from \-1 to \+1, a binary label for sarcasm). A simple probe model, such as a linear regression or logistic regression classifier, will be trained to predict these human ratings using only the IEC-generated Intronic Vector as input features. The Richness\_Intronic score will be the performance of this probe model, measured by an appropriate metric like Pearson correlation for continuous ratings or F1-score for categorical labels. A high correlation or F1-score demonstrates that the Intronic Vector contains rich, decodable information about the pragmatic nature of the text.

### **4.2 Downstream Task-Based Evaluation**

The ultimate validation of the ARC-SPLICER V2 architecture is its ability to improve performance on complex, real-world tasks where understanding pragmatic nuance is critical. An A/B testing framework will be established to compare the full ARC-SPLICER V2 system against a baseline model.

* **Model A (Baseline):** A standard architecture where the GATformer processes the raw text directly, without the benefit of the Splicer or Regulation modules.  
* **Model B (ARC-SPLICER V2):** The full, proposed dual-stream architecture, including the IEC, the GATformer processing the Exonic Stream, and the ARC-REGULATION module using the Intronic Stream to modulate the final output.

The two models will be evaluated on a suite of carefully selected downstream tasks:

* **Persuasion and Negotiation Dialogue:** In a simulated negotiation task, the models will engage with a human or another AI agent to reach an agreement. Success will be measured by task-specific outcomes (e.g., achieving a favorable deal, number of turns to agreement). The hypothesis is that Model B, by understanding and responding to hedges, politeness cues, and expressions of intent from its counterpart, will be a more effective negotiator.  
* **Fact-Checking with Biased Sources:** The models will be tasked with assessing the veracity of claims presented in text snippets from sources with known biases. The texts will be rich in doxastic markers like "sources claim," "it is believed," and "many people are saying." Performance will be measured by the accuracy of the fact-checking classification. The hypothesis is that Model B will leverage the Intronic Stream to identify these markers of subjective belief and appropriately lower its confidence in the factuality of the associated claims, leading to higher accuracy than the baseline, which might interpret the claims more literally.  
* **Empathetic Customer Support:** The models will act as customer support chatbots interacting with users expressing various emotions, from satisfaction to frustration. Performance will be evaluated using both automated metrics (e.g., problem resolution rate) and human ratings for qualities like empathy, appropriateness, and helpfulness. The hypothesis is that Model B, using its Intronic Stream to detect affective cues like negative sentiment and frustration, will trigger more appropriate and empathetic responses via the ARC-REGULATION module, leading to significantly higher user satisfaction scores.

Across all these tasks, the primary hypothesis is that Model B (ARC-SPLICER V2) will demonstrate statistically significant improvements over Model A (Baseline), providing conclusive evidence for the value of the dual-stream, bio-inspired approach to language processing.

## **5\. End-to-End Regulatory Data Flow: A Concrete Walkthrough**

To make the abstract architectural concepts tangible, this section provides a step-by-step walkthrough of how a single, pragmatically complex utterance is processed by the ARC-SPLICER V2 system. This narrative traces the data from raw input to modulated final output, illustrating the state of the Exonic and Intronic streams at each stage.

**Input Text:** *"Well, I guess it might be possible, but honestly, are you absolutely sure that's a good idea? It seems a bit risky."*

This utterance is chosen for its density of linguistic introns. It contains discourse fillers, hedges, stance markers, boosters, and affective language, all wrapped around a core set of semantic propositions.

### **Step 1: Ingestion by the Intronic/Exonic Classifier (IEC)**

The raw text string is received by the IEC. It is first tokenized into a sequence of tokens according to the model's vocabulary (e.g., using a WordPiece or BPE tokenizer 39). This token sequence is then converted into input embeddings and fed into the IEC's shared Transformer backbone. The Transformer processes the entire sequence, generating a final-layer contextual hidden state vector for each token.

### **Step 2: Parallel Splicing within the IEC**

The sequence of hidden states is passed simultaneously to the IEC's two specialized heads.

* **Sequence Tagging Head Output:** The tagging head performs a token-level classification, assigning a label from the Linguistic Intron Taxonomy (Table 1\) to each token. The output is a sequence of labels:  
  * Well \-\> DISC-FILLER  
  * I \-\> O  
  * guess \-\> EPI-HEDGE  
  * it \-\> O  
  * might \-\> EPI-HEDGE  
  * be \-\> O  
  * possible \-\> O  
  * , \-\> O  
  * but \-\> DISC-CONTRAST  
  * honestly \-\> DISC-STANCE  
  * , \-\> O  
  * are \-\> O  
  * you \-\> O  
  * absolutely \-\> EPI-BOOST  
  * sure \-\> O  
  * that's \-\> O  
  * a \-\> O  
  * good \-\> O  
  * idea \-\> O  
  * ? \-\> O  
  * It \-\> O  
  * seems \-\> EPI-HEDGE  
  * a \-\> O  
  * bit \-\> AFF-NEG (as a mitigator for "risky")  
  * risky \-\> O  
  * . \-\> O  
* **Regulatory Vector Generation Head Output:** This head identifies the hidden states corresponding to the tokens tagged as intronic (Well, guess, might, but, honestly, absolutely, seems, bit). It applies an attention-pooling mechanism to these specific hidden states, aggregating their information into a single, fixed-dimension **Intronic Vector**. For visualization, this vector's key activated dimensions might look like this:  
  * epistemic\_uncertainty: 0.85 (High, from guess, might, seems)  
  * epistemic\_certainty: 0.90 (High, from absolutely sure as a boosted query)  
  * discourse\_contrast: 0.95 (High, from but)  
  * subjective\_stance: 0.70 (Moderate, from honestly, I guess)  
  * negative\_affect: 0.65 (Moderate, from a bit risky and the overall hesitant tone)  
  * politeness: 0.20 (Low)

### **Step 3: Stream Separation and Routing**

The system now has two distinct, synchronized streams of information.

* **Exonic Stream (Routed to GATformer):** This is the full sequence of contextual hidden states from the IEC's backbone. However, the embeddings for the tokens tagged as intronic are now masked (conceptually replaced with null values). The GATformer receives a "purified" sequence that represents the core propositions: \[I, it, be, possible, ,, are, you, sure, that's, a, good, idea,?, It, risky,.\]  
* **Intronic Stream (Routed to ARC-REGULATION):** The generated Intronic Vector \[uncertainty: 0.85, certainty\_query: 0.90, contrast: 0.95,...\] is sent to the ARC-REGULATION module.

### **Step 4: GATformer Semantic Processing**

The GATformer processes the purified Exonic Stream. Free from the pragmatic modulators, it can focus on the logical structure. It identifies the key semantic propositions:

1. Proposition 1: (it is possible)  
2. Proposition 2: (is it a good idea)  
3. Proposition 3: (it is risky)  
   The GATformer notes the semantic tension between Proposition 1/2 and Proposition 3\.

### **Step 5: ARC-REGULATION Module Interpretation**

Simultaneously, the ARC-REGULATION module receives the Intronic Vector. It interprets the high activation values:

* The high epistemic\_uncertainty value indicates the speaker's strong doubt.  
* The high certainty\_query value indicates the speaker is directly questioning the certainty of the listener.  
* The high discourse\_contrast value signals that the second clause (...a good idea?) stands in opposition to the first (...possible).  
* The negative\_affect value confirms a negative emotional valence associated with the concept.

### **Step 6: Modulated Output Generation**

The control signals from the ARC-REGULATION module are sent to the system's final output generator. A simple, literal response based only on the GATformer's output might be, "Yes, it is possible but risky." However, the regulatory signals override this simple generation. The generator is now constrained by the following directives:

1. **Acknowledge speaker uncertainty:** Do not give a simple "yes" or "no." Reflect the doubt expressed.  
2. **Address the core question about certainty:** The user is explicitly asking for a confidence judgment.  
3. **Address the expressed risk:** The negative affect associated with "risky" must be acknowledged.  
4. **Adopt a neutral-to-cautious stance:** The combination of hedging and stance markers indicates a non-committal position is appropriate.

### **Final System Output**

The output generator synthesizes these constraints to produce a nuanced, pragmatically appropriate response:

**Final Output:** *"You've raised an important point about the risk. While it is technically possible, the high degree of uncertainty you've expressed is valid. Based on the available information, I cannot state with high confidence that it is a 'good idea' without a more thorough analysis of the risks involved."*

This final output demonstrates the power of the dual-stream architecture. It is semantically grounded, logically coherent, and, crucially, pragmatically and affectively aligned with the user's complex input.

## **6\. Amended Research Plan and Future Trajectories**

The development of ARC-SPLICER V2 is not an end-point but the beginning of a new research trajectory focused on building more sophisticated, bio-inspired language processing systems. The successful implementation of this architecture will validate the core hypothesis that linguistic introns are regulatory and will open up several avenues for future investigation. This section outlines the immediate research plan and maps out longer-term future directions.

### **6.1 Summary of Core Research Questions and Milestones**

The research program is structured around three core questions, each with a corresponding deliverable milestone that marks a critical phase of the project.

* Q1: Can a rich, hierarchical taxonomy of linguistic introns be reliably and consistently annotated across diverse text corpora?  
  The entire project rests on the ability to create a high-quality, gold-standard dataset. This initial phase will focus on the development of the intron taxonomy, the creation of detailed annotation guidelines, and the execution of the annotation and adjudication process.  
  * **Milestone 1: Gold-Standard Annotated Corpus.** Delivery of a multi-genre corpus of at least 10,000 sentences, annotated according to the Linguistic Intron Taxonomy with an inter-annotator agreement (Cohen's Kappa) of \>0.80.  
* Q2: Can a multi-task, multi-head Transformer architecture effectively learn to separate text into semantic and regulatory streams with high fidelity?  
  This phase involves the technical implementation and training of the Intronic/Exonic Classifier (IEC). The primary goal is to validate that the model can learn the complex splicing task and produce high-quality, functionally distinct output streams.  
  * **Milestone 2: IEC v1.0.** A trained and validated IEC model that achieves a target threshold on the composite Separation Fidelity (Φ) metric, demonstrating high accuracy in tagging, high purity in the exonic stream, and high richness in the intronic stream.  
* Q3: Does the dual-stream architecture lead to measurable and statistically significant improvements in downstream tasks that require pragmatic and affective reasoning?  
  This is the ultimate validation phase, where the end-to-end ARC-SPLICER V2 system is benchmarked against a baseline model on complex, real-world tasks.  
  * **Milestone 3: Publication of A/B Test Results.** Completion of the downstream task-based evaluation and the publication of a research paper detailing the statistically significant performance gains of the ARC-SPLICER V2 architecture in areas like negotiation, fact-checking, and empathetic dialogue.

### **6.2 Future Directions: Towards a Fully Dynamic and Evolved System**

Beyond the immediate research plan, the ARC-SPLICER V2 framework enables exploration into several cutting-edge areas of AI research.

* **Dynamic and "Co-transcriptional" Splicing:** The current design generates a single Intronic Vector for an entire utterance. A more advanced model could update this vector dynamically as a sentence or turn is being processed. This would more closely mimic the biological process of co-transcriptional splicing, especially in gigantic genes where introns are removed sequentially to manage the length of the nascent transcript.16 Such a system could make real-time regulatory adjustments during a long conversational turn, for example, by detecting a shift in tone midway through and adapting its planned response before the user has even finished speaking.  
* **Cross-Lingual and Cross-Cultural Pragmatics:** The initial Intron Taxonomy will be developed for English. A major future research effort would be to expand this framework to other languages and cultures. This would involve a significant annotation effort but would allow for the investigation of universal versus language-specific pragmatic strategies.40 The architecture itself is language-agnostic, making it an ideal testbed for exploring the fundamental principles of pragmatic regulation across the world's languages.  
* **Intron Exonization and the Emergence of Pragmatic Competence:** In genetics, "exonization" is an evolutionary process where a portion of an intron can become a new exon, thus acquiring a protein-coding function.14 An intriguing long-term research question is whether an analogous process can occur in large language models. By training a massive, unsupervised version of the ARC-SPLICER architecture on a vast corpus, it may be possible to observe whether certain linguistic patterns that begin as purely semantic (exonic) gradually acquire regulatory (intronic) functions over the course of training. For example, a phrase that is initially just a descriptor might evolve to become a consistent marker of a particular stance or sentiment. Studying this phenomenon could provide profound insights into how pragmatic competence emerges from statistical learning, representing a landmark study in the evolution of artificial communicative intelligence.

#### **Works cited**

1. Introns and 'junk' DNA : r/biology \- Reddit, accessed July 9, 2025, [https://www.reddit.com/r/biology/comments/1b1lxze/introns\_and\_junk\_dna/](https://www.reddit.com/r/biology/comments/1b1lxze/introns_and_junk_dna/)  
2. The Functional Benefits of Introns in Genomes \- PMC, accessed July 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4742320/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4742320/)  
3. www.numberanalytics.com, accessed July 9, 2025, [https://www.numberanalytics.com/blog/introns-in-gene-regulation\#:\~:text=A%3A%20Introns%20play%20a%20crucial,affecting%20transcription%20and%20translation%20efficiency.](https://www.numberanalytics.com/blog/introns-in-gene-regulation#:~:text=A%3A%20Introns%20play%20a%20crucial,affecting%20transcription%20and%20translation%20efficiency.)  
4. The Role of Introns in Gene Regulation \- Number Analytics, accessed July 9, 2025, [https://www.numberanalytics.com/blog/introns-in-gene-regulation](https://www.numberanalytics.com/blog/introns-in-gene-regulation)  
5. Introns in gene evolution \- PubMed, accessed July 9, 2025, [https://pubmed.ncbi.nlm.nih.gov/12868603/](https://pubmed.ncbi.nlm.nih.gov/12868603/)  
6. The function of introns \- PubMed, accessed July 9, 2025, [https://pubmed.ncbi.nlm.nih.gov/22518112/](https://pubmed.ncbi.nlm.nih.gov/22518112/)  
7. arXiv:2502.12378v1 \[cs.CL\] 17 Feb 2025, accessed July 9, 2025, [https://arxiv.org/pdf/2502.12378](https://arxiv.org/pdf/2502.12378)  
8. Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges \- arXiv, accessed July 9, 2025, [https://arxiv.org/html/2502.12378v1](https://arxiv.org/html/2502.12378v1)  
9. Pragmatic Analysis in NLP: A Comprehensive Guide \- Number Analytics, accessed July 9, 2025, [https://www.numberanalytics.com/blog/pragmatic-analysis-nlp-guide](https://www.numberanalytics.com/blog/pragmatic-analysis-nlp-guide)  
10. A Pragmatics-Centered Evaluation Framework for Natural Language Understanding \- ACL Anthology, accessed July 9, 2025, [https://aclanthology.org/2022.lrec-1.255.pdf](https://aclanthology.org/2022.lrec-1.255.pdf)  
11. The Function of Introns \- PMC, accessed July 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3325483/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3325483/)  
12. Introns as Gene Regulators: A Brick on the Accelerator \- PMC, accessed July 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6374622/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6374622/)  
13. PeterZhizhin/BERTUncertaintyDetection: Hedge detection ... \- GitHub, accessed July 9, 2025, [https://github.com/PeterZhizhin/BERTUncertaintyDetection](https://github.com/PeterZhizhin/BERTUncertaintyDetection)  
14. Mechanism of alternative splicing and its regulation \- PMC, accessed July 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4360811/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4360811/)  
15. Alternative splicing \- Wikipedia, accessed July 9, 2025, [https://en.wikipedia.org/wiki/Alternative\_splicing](https://en.wikipedia.org/wiki/Alternative_splicing)  
16. Co-transcriptional splicing facilitates transcription of gigantic genes ..., accessed July 9, 2025, [https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1011241](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1011241)  
17. What is the canonical order of capping, splicing, and polyadenylation? : r/molecularbiology, accessed July 9, 2025, [https://www.reddit.com/r/molecularbiology/comments/193h0p1/what\_is\_the\_canonical\_order\_of\_capping\_splicing/](https://www.reddit.com/r/molecularbiology/comments/193h0p1/what_is_the_canonical_order_of_capping_splicing/)  
18. Pre-mRNA splicing and its co-transcriptional connections \- PMC, accessed July 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10524715/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10524715/)  
19. hellohaptik/multi-task-NLP: multi\_task\_NLP is a utility toolkit ... \- GitHub, accessed July 9, 2025, [https://github.com/hellohaptik/multi-task-NLP](https://github.com/hellohaptik/multi-task-NLP)  
20. Week 6 \- Lab 3 (Multi-Head Attention) \- YouTube, accessed July 9, 2025, [https://www.youtube.com/watch?v=PZE6Ev-pEXk](https://www.youtube.com/watch?v=PZE6Ev-pEXk)  
21. Multi-task BERT Classification \- Stanford University, accessed July 9, 2025, [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/final-reports/final-report-169683991.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/final-reports/final-report-169683991.pdf)  
22. Exploring Multi-Head Attention: Why More Heads Are Better Than One | by Hassaan Idrees, accessed July 9, 2025, [https://medium.com/@hassaanidrees7/exploring-multi-head-attention-why-more-heads-are-better-than-one-006a5823372b](https://medium.com/@hassaanidrees7/exploring-multi-head-attention-why-more-heads-are-better-than-one-006a5823372b)  
23. Multi-Scale Self-Attention for Text Classification, accessed July 9, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/6290/6146](https://ojs.aaai.org/index.php/AAAI/article/view/6290/6146)  
24. MirunaPislar/multi-head-attention-labeller: Joint text ... \- GitHub, accessed July 9, 2025, [https://github.com/MirunaPislar/multi-head-attention-labeller](https://github.com/MirunaPislar/multi-head-attention-labeller)  
25. Seeing Both the Forest and the Trees: Multi-head Attention for Joint Classification on Different Compositional Levels \- ACL Anthology, accessed July 9, 2025, [https://aclanthology.org/2020.coling-main.335.pdf](https://aclanthology.org/2020.coling-main.335.pdf)  
26. Multi-Task Label Embedding for Text Classification \- ACL Anthology, accessed July 9, 2025, [https://aclanthology.org/D18-1484/](https://aclanthology.org/D18-1484/)  
27. \[1710.07210\] Multi-Task Label Embedding for Text Classification \- arXiv, accessed July 9, 2025, [https://arxiv.org/abs/1710.07210](https://arxiv.org/abs/1710.07210)  
28. When LLMs Team Up: The Emergence of Collaborative Affective Computing \- arXiv, accessed July 9, 2025, [https://arxiv.org/html/2506.01698v1](https://arxiv.org/html/2506.01698v1)  
29. Emotions in the Loop: A Survey of Affective Computing for Emotional Support \- arXiv, accessed July 9, 2025, [https://arxiv.org/html/2505.01542v1](https://arxiv.org/html/2505.01542v1)  
30. Affective Computing in the Era of Large Language Models: A Survey from the NLP Perspective \- arXiv, accessed July 9, 2025, [https://arxiv.org/html/2408.04638v1](https://arxiv.org/html/2408.04638v1)  
31. Recent Trends of Multimodal Affective Computing: A Survey from an NLP Perspective \- arXiv, accessed July 9, 2025, [https://arxiv.org/pdf/2409.07388](https://arxiv.org/pdf/2409.07388)  
32. Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges \- arXiv, accessed July 9, 2025, [https://arxiv.org/html/2502.12378v3](https://arxiv.org/html/2502.12378v3)  
33. A Comprehensive Analysis of Text Annotation | with 2024 Trend Insights \- BasicAI, accessed July 9, 2025, [https://www.basic.ai/blog-post/what-is-text-annotation-in-machine-learning-nlp](https://www.basic.ai/blog-post/what-is-text-annotation-in-machine-learning-nlp)  
34. Complete Guide to Text Annotation in 2025 \- Labellerr, accessed July 9, 2025, [https://www.labellerr.com/blog/the-ultimate-guide-to-text-annotation-techniques-tools-and-best-practices-2/](https://www.labellerr.com/blog/the-ultimate-guide-to-text-annotation-techniques-tools-and-best-practices-2/)  
35. 1\. The Basics \- Natural Language Annotation for Machine Learning \[Book\] \- O'Reilly Media, accessed July 9, 2025, [https://www.oreilly.com/library/view/natural-language-annotation/9781449332693/ch01.html](https://www.oreilly.com/library/view/natural-language-annotation/9781449332693/ch01.html)  
36. PragMaBERT: Analyzing Pragmatic Markers in Political Speech, accessed July 9, 2025, [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/final-reports/256878985.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/final-reports/256878985.pdf)  
37. Building a Multi-Task NLP Model for Real-World Applications | by Tilak Mudgal | Medium, accessed July 9, 2025, [https://medium.com/@tilak559/building-a-multi-task-nlp-model-for-real-world-applications-95d6b5dd1d17](https://medium.com/@tilak559/building-a-multi-task-nlp-model-for-real-world-applications-95d6b5dd1d17)  
38. Multitask Text Classification Using DistilBERT | by Kuldeep Singh \- Medium, accessed July 9, 2025, [https://medium.com/@kuldeepsingh\_92974/multitask-text-classification-using-distilbert-085177145816](https://medium.com/@kuldeepsingh_92974/multitask-text-classification-using-distilbert-085177145816)  
39. Seq2seq and Attention \- Lena Voita, accessed July 9, 2025, [https://lena-voita.github.io/nlp\_course/seq2seq\_and\_attention.html](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)  
40. Natural Language Processing for Dialects of a Language: A Survey \- arXiv, accessed July 9, 2025, [https://arxiv.org/pdf/2401.05632](https://arxiv.org/pdf/2401.05632)  
41. What Is The Point of Introns? : r/askscience \- Reddit, accessed July 9, 2025, [https://www.reddit.com/r/askscience/comments/mz4da/what\_is\_the\_point\_of\_introns/](https://www.reddit.com/r/askscience/comments/mz4da/what_is_the_point_of_introns/)