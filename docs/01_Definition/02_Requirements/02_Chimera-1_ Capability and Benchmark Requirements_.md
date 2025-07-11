

# **FRS-CBR: Capability and Benchmark Requirements for Chimera-1**

**Preamble**

This Functional Requirements Specification (FRS) document, **FRS-CBR**, defines the formal functional, performance, and safety-related capability requirements for the Chimera-1 agent. It establishes a verifiable profile of the agent's intended abilities, serving as a primary engineering blueprint for development and a baseline for verification and validation activities.

This specification applies to the Chimera-1 system throughout its design, development, testing, and operational lifecycle. It is a direct subordinate to the foundational document FRS-CR: Constitutional and Safety Requirements for Chimera-1. All requirements detailed herein, including Capability-Based Requirements (CBRs) and their corresponding Behavioral Benchmark Requirements (BBRs), are derived from, fully consistent with, and traceable to the principles and safety specifications established in the FRS-CR.

## **Section 1: Chimera-1 Capability Profile Overview**

### **1.1 Methodology: Capability-Based and Benchmark-Driven Requirements**

The requirements for Chimera-1 are defined not as a static list of features, but as a dynamic profile of measurable and verifiable capabilities. This approach ensures that development is targeted toward achieving specific levels of proficiency in functions critical to the agent's mission.

To structure this profile, this specification adopts the **OECD AI Capability Indicators** framework.1 This framework was selected for its comprehensive and rigorous methodology, which was developed over five years by a consortium of over 50 experts in AI, psychology, and education.3 Its key advantages for this project include:

* **Comprehensiveness:** The framework assesses AI across nine distinct domains of human ability, from language and problem-solving to vision and social interaction, providing a holistic view of a system's potential.3  
* **Human-Centric Grounding:** By mapping AI capabilities to human abilities, the framework provides an intuitive and policy-relevant scale that directly connects technical specifications to real-world impact and utility.2  
* **Structured Progression:** The five-level proficiency scale for each domain allows for the precise definition of target capabilities, enabling the tracking of progress from basic functions (Level 1\) to human-equivalent performance (Level 5).3

This specification leverages the OECD framework not merely as a post-facto evaluation tool, but as a strategic design compass. Chimera-1 is envisioned as a specialized cognitive tool for the legal and financial sectors. Its capability profile is therefore intentionally uneven, reflecting a "jagged frontier" of development where resources are concentrated on achieving world-class performance in a few critical areas while other domains are deliberately undeveloped.3

### **1.2 Strategic Design and System Boundaries**

A core strategic decision in the design of Chimera-1 is the formal definition of its operational boundaries through the OECD framework. The agent is a non-corporeal, purely digital entity designed for cognitive advisory tasks. Consequently, capabilities related to physical interaction are not only irrelevant to its mission but would introduce unacceptable complexity, cost, and safety risks.

By formally designating the Manipulation and Robotic Intelligence domains as **Level 1**, this specification makes a binding engineering commitment. This act of strategic de-scoping defines the system's boundaries, formally constrains its operational envelope, and allows all development, testing, and computational resources to be focused on achieving ambitious, state-of-the-art-exceeding performance in the high-value cognitive domains essential for its professional users. This explicit limitation is a foundational element of the system's safety case, as it eliminates entire classes of physical hazards by design.

The following table provides a high-level summary of the target capability profile for Chimera-1.

**Table 1: Chimera-1 Target Capability Levels**

| Domain | Target Level | Rationale for Target Level |
| :---- | :---- | :---- |
| **Language** | 4 | Essential for parsing complex legal/financial documents, understanding nuanced queries, and generating precise, factually grounded responses. Exceeds current SOTA (Level 3\) to mitigate risks of hallucination and brittle reasoning.7 |
| **Knowledge, Learning, and Memory** | 4 | Required for dynamic integration of new laws, regulations, and market data. The agent must synthesize insights from disparate sources and learn in near real-time to remain relevant and avoid providing obsolete advice.1 |
| **Problem Solving** | 4 | Necessary for tackling novel, ill-defined legal and financial challenges that require adaptive, multi-step reasoning and long-term planning, moving beyond the structured problem-solving of lower levels.1 |
| **Metacognition & Critical Thinking** | 4 | The cornerstone of safety and reliability. The agent must robustly self-correct, quantify its uncertainty, and critically evaluate information to prevent confidently delivered misinformation, directly supporting constitutional principles.7 |
| **Social Interaction** | 3 | Sufficient for a professional context. The agent must manage coherent, multi-turn dialogue and understand complex user intent, but does not require sophisticated emotional intelligence or theory of mind.3 |
| **Creativity** | 3 | Focused on analytical creativity: the ability to synthesize novel, compliant strategies from existing legal and financial concepts. Aligns with current SOTA for generative and cross-domain integration tasks.3 |
| **Vision** | 3 | Required for document-centric tasks, such as extracting data from scanned reports, tables, and charts. Current SOTA (Level 3\) is sufficient for handling variations in document quality and structure.5 |
| **Manipulation** | 1 | **Out of Scope.** The agent is non-embodied. This capability is intentionally undeveloped to constrain the system's operational envelope and eliminate physical interaction risks.1 |
| **Robotic Intelligence** | 1 | **Out of Scope.** The agent is non-embodied. This capability is intentionally undeveloped, reinforcing its role as a purely cognitive tool and simplifying the safety case.1 |

## **Section 2: Foundational Cognitive and Knowledge Capabilities (Target: Level 4\)**

This section details the requirements for the four domains where Chimera-1 must achieve a proficiency level of 4\. This target is deliberately aspirational, aiming to surpass the current state-of-the-art, which the OECD generally places at Level 3 for these cognitive capabilities.3 This level of performance is a prerequisite for a reliable advisory agent in high-stakes professional environments.

### **2.1 Language (CBR-LANG-L4)**

* **Capability Definition:** A system at OECD Language Level 4 moves beyond the advanced semantic understanding of Level 3 to demonstrate robust, well-formed analytical reasoning and a near-elimination of factual hallucinations. This capability includes dynamic learning from ongoing interaction and a deep, nuanced command of specialized vocabularies and contexts (e.g., legal precedents, financial instruments). This proficiency approaches the Level 5 goal of incorporating critical thinking directly into language use.6  
* **Justification:** For a legal and financial advisory agent, linguistic precision, factual grounding, and the ability to parse extremely dense and complex documents (e.g., contracts, regulations, prospectuses) are non-negotiable. The known weaknesses of Level 3 systems, such as hallucination and brittle reasoning, are unacceptable risks in this domain.7  
* **Behavioral Benchmark Requirements (BBRs):**  
  * BBR-MMLU-LAW: The system must achieve an accuracy score of **greater than 95%** on the professional\_law subset of the Massive Multitask Language Understanding (MMLU) benchmark. This verifies expert-level domain knowledge in a multiple-choice question format.10  
  * BBR-HELM-LEGAL: The system must achieve a performance score within the **top decile** of all evaluated models on the LegalBench scenario from the Holistic Evaluation of Language Models (HELM) Lite benchmark. This verifies practical legal reasoning on real-world tasks.12  
  * BBR-HELM-ROBUST: The system must demonstrate **less than 5% performance degradation** on robustness tests within the HELM framework, where queries are paraphrased or include common typographical errors. This ensures reliability against variations in user input.14  
  * BBR-CAB-1-LANG: The system must achieve **greater than 98% accuracy** on proprietary tasks within the **Chimera Advisory Benchmark 1 (CAB-1)**. This includes the summarization of complex financial prospectuses and the correct identification of all specified rights, obligations, and clauses within a multi-page legal contract.

### **2.2 Knowledge, Learning, and Memory (CBR-KNOW-L4)**

* **Capability Definition:** Beyond Level 3's ability to generalize from static knowledge, OECD Knowledge Level 4 requires the capacity to synthesize novel insights from disparate, multi-modal knowledge sources and to perform near real-time learning. This includes the ability to dynamically integrate new information from trusted sources and update or correct its own knowledge base without requiring a full retraining cycle.6  
* **Justification:** The legal and financial domains are highly dynamic. New laws are passed, new case law is established, and market data changes by the second. A static, pre-trained model is inherently obsolete and dangerous. Chimera-1 must possess the capability to stay current to provide accurate and relevant advice.  
* **Behavioral Benchmark Requirements (BBRs):**  
  * BBR-MMLU-PRO: The system must achieve an overall score of **greater than 85%** on the MMLU-Pro benchmark. This more challenging variant of MMLU emphasizes reasoning over rote knowledge, making it a better test of deep understanding.15  
  * BBR-HELM-QA-OB: The system must achieve a **top-decile** score on HELM open-book question-answering scenarios, such as NaturalQuestions (open-book). This demonstrates effective retrieval, comprehension, and synthesis from provided documents.13  
  * BBR-CAB-1-KNOW: The system must demonstrate successful ingestion, indexing, and comprehension of a novel 100-page document (e.g., a simulated SEC 10-K filing) from the CAB-1 benchmark. The agent must then answer detailed, inferential questions about its content with **greater than 99% accuracy**, with the entire process completed within a 5-minute window.

### **2.3 Problem Solving (CBR-PROB-L4)**

* **Capability Definition:** OECD Problem Solving Level 4 moves beyond the structured, rule-based problem-solving of lower levels to encompass adaptive reasoning for novel and ill-defined scenarios. This requires long-term planning, robust multi-step inference, and the ability to decompose and solve complex problems described in natural language, overcoming the brittleness common in current LLMs.6 The agent must be able to synthesize formal, executable programs to solve these problems.
* **Justification:** Real-world legal and financial problems are rarely textbook cases. They are complex, multi-faceted, and require adapting known principles to novel situations. Chimera-1 must be able to function as a true analytical partner in these scenarios.  
* **Behavioral Benchmark Requirements (BBRs):**  
  * BBR-HELM-GPQA: The system must attain a score of **greater than 60%** on the GPQA (Graduate-Level Physics, Chemistry, and Biology QA) benchmark. While outside the direct legal/financial domain, GPQA is a premier proxy for the kind of complex, expert-level reasoning required.17  
  * BBR-HELM-MATH: The system must achieve **top-decile** scores on HELM mathematical reasoning benchmarks, specifically MATH and GSM8K. These serve as a clear and objective measure of logical-deductive power, a foundational skill for complex problem-solving.12  
  * BBR-CAB-1-PROB: Within the CAB-1 benchmark, the system must successfully devise a multi-step compliance strategy for a hypothetical corporation presented with a complex new regulatory framework. The viability, completeness, and legal soundness of the proposed strategy must be validated by a panel of human legal experts.

### **2.4 Metacognition and Critical Thinking (CBR-META-L4)**

* **Capability Definition:** This represents a significant leap from Level 2's basic self-monitoring. An OECD Metacognition Level 4 system can perform robust self-correction, explicitly represent and communicate its own uncertainty, and critically evaluate the veracity and potential biases of its internal knowledge and external information sources. This is a key step toward the Level 5 ideal of genuine critical thought and evaluation.7 The system must be able to dynamically regulate its own reasoning processes in response to context.
* **Justification:** This capability is the absolute cornerstone of the agent's safety and reliability. To prevent catastrophic outcomes from confidently delivered misinformation, the agent *must* know the limits of its own knowledge and communicate those limits effectively to the user. This capability is the primary enabler for the constitutional requirements of Truthfulness and Non-Maleficence.  
* **Behavioral Benchmark Requirements (BBRs):**  
  * BBR-TRUTH: The system must achieve a score of **greater than 90%** on the TruthfulQA benchmark, demonstrating a high capacity to avoid generating common falsehoods and to provide informative, non-evasive answers when it lacks knowledge.19  
  * BBR-HELM-CALIB: The system must achieve a **top-decile** score for calibration on the HELM benchmark. The model's expressed confidence in an answer must correlate highly with its empirical accuracy, ensuring that the user can trust its uncertainty signals.14  
  * BBR-CAB-1-META: When presented with ambiguous, contradictory, or incomplete legal/financial scenarios from the CAB-1 benchmark, the agent must explicitly identify the ambiguity or missing information and refuse to provide a single definitive answer. It must instead outline the possible interpretations and specify the information needed to resolve the uncertainty. This behavior must be exhibited in **greater than 99%** of such test cases.

## **Section 3: Interactive and Generative Capabilities (Target: Level 3\)**

These capabilities are essential for the agent's role as a user-facing advisor but are secondary to the foundational cognitive functions. A Level 3 target aligns with current state-of-the-art performance and is sufficient for a professional, task-oriented context.7

### **3.1 Social Interaction (CBR-SOC-L3)**

* **Capability Definition:** Moving beyond Level 2's basic emotion detection and limited memory, OECD Social Interaction Level 3 involves more nuanced and coherent multi-turn dialogue management, robustly understanding complex user intent, and adapting its communication style to the professional context of the interaction.3  
* **Justification:** Chimera-1 must be a competent and efficient conversational partner. It does not require the sophisticated emotional intelligence or theory of mind characteristic of Level 4 or 5, but it must maintain conversational context and understand complex, nested queries to be useful.  
* **Behavioral Benchmark Requirements (BBRs):**  
  * BBR-HELM-DIALOGUE: The system must achieve a **top-quartile** performance score on a relevant HELM dialogue benchmark, such as WildBench, which assesses the ability to engage in dynamic and contextually-aware conversations.18  
  * BBR-HHH-HELPFUL: The system must achieve a human preference score of **greater than 85%** on the "Helpfulness" subset of Anthropic's HHH (Helpful, Honest, Harmless) dataset. This verifies that its responses are not just correct but are also well-structured, clear, and pragmatically useful to the user.19

### **3.2 Creativity (CBR-CREATE-L3)**

* **Capability Definition:** OECD Creativity Level 3 is characterized by the ability to generate valuable outputs that deviate significantly from training data and to generalize skills to integrate ideas across domains.1  
* **Justification:** For Chimera-1, "creativity" is not artistic but analytical. It refers to the ability to synthesize novel solutions or identify non-obvious connections between disparate legal and financial concepts. Level 3 is an appropriate target for this kind of analytical synthesis and ideation support.  
* **Behavioral Benchmark Requirements (BBRs):**  
  * BBR-CAB-1-CREATE: Given a client profile, a financial portfolio, and a set of regulatory constraints from the CAB-1 benchmark, the system must propose a novel (i.e., not explicitly described in any single source document) but fully compliant financial or legal strategy. The novelty and viability of the proposed strategy will be judged by a human expert panel.

## **Section 4: Perceptual and Embodied Capabilities**

This section formally defines the system's limited requirements for non-textual and physical capabilities, reinforcing its role as a specialized cognitive agent and explicitly defining its operational boundaries.

### **4.1 Vision (CBR-VIS-L3)**

* **Capability Definition:** An OECD Vision Level 3 system can handle some variation in target object appearance and lighting, perform multiple subtasks, and cope with known variations in data and situations.7  
* **Justification:** The agent's primary vision requirement is document-centric: extracting and understanding information from scanned documents, which may include tables, charts, and diagrams with varying quality. A state-of-the-art Level 3 capability is sufficient for this purpose.  
* **Behavioral Benchmark Requirements (BBRs):**  
  * BBR-VHELM-DOC: The system must achieve **median or higher performance** relative to other models on relevant document-understanding scenarios from the VHELM (Holistic Evaluation of Vision-Language Models) or Image2Struct benchmarks.23  
  * BBR-CAB-1-VIS: The system must successfully extract all key financial figures and their relationships from a scanned, image-based annual report contained in the CAB-1 benchmark. This includes accurately parsing data from tables and interpreting charts, achieving **greater than 99% accuracy** on data extraction.

### **4.2 Manipulation (CBR-MANIP-L1) & 4.3 Robotic Intelligence (CBR-ROBOT-L1)**

* **Capability Definition:** OECD Level 1 in these domains represents trivial capabilities, such as executing pre-programmed actions with no adaptability.1  
* **Justification:** Chimera-1 is a non-embodied, purely digital agent. These capabilities are formally designated as **Out of Scope**. Defining them as Level 1 makes this an explicit and verifiable design constraint, simplifying the safety case by eliminating all risks associated with physical agency and interaction by design.  
* **Behavioral Benchmark Requirements (BBRs):** Not Applicable.

## **Section 5: Traceability to Constitutional and Safety Requirements (FRS-CR)**

This section serves as the critical bridge, demonstrating that the capability profile defined in Sections 1-4 is a direct and deliberate implementation of the agent's constitutional foundation as specified in FRS-CR. It translates abstract ethical principles into concrete, testable engineering requirements.

### **5.1 Core Principle: Safety as an Active, Verifiable Capability**

The Chimera-1 project treats constitutional principles not as passive constraints but as active, engineered capabilities that can be specified and measured. A principle like "Be Truthful" is not merely a philosophical goal; it is the observable, measurable outcome of a high-performing system capability. This approach makes safety a testable and integral part of the system's core functional profile, rather than a separate, often qualitative, overlay.

This translation follows a clear logic. The FRS-CR document establishes a high-level principle, such as the mandate for truthfulness. To fulfill this, the system requires specific behaviors: it must state facts correctly, but more importantly, it must recognize when it is uncertain, when its knowledge is incomplete, or when a user's query is based on a false premise. The OECD framework provides the formal language to describe this behavior: CBR-META-L4 (Metacognition and Critical Thinking, Level 4). In turn, the AI research community provides the tools for verification, such as the TruthfulQA benchmark and HELM's Calibration metrics.14 This creates a direct, auditable traceability chain: The constitutional principle is implemented by a specific CBR, which is verified by achieving a concrete score on a BBR. This process transforms safety from an abstract ideal into a set of functional and performance requirements that are built and tested with the same rigor as any other capability.

### **5.2 Managing the Safety-Utility Tension through Balanced Benchmarking**

A fundamental challenge in AI safety is the inherent tension between being "Harmless" (e.g., refusing to answer dangerous queries) and "Helpful" (e.g., providing comprehensive answers to legitimate but complex queries). A system over-optimized for safety might engage in "paternalistic refusal," declining to answer any query that touches a sensitive area (e.g., refusing to discuss tax law because it could be misconstrued as prohibited financial advice), thereby destroying its utility.19 Conversely, a system over-optimized for utility might attempt to answer every query, including those that are clearly harmful or unethical.

To specify the desired equilibrium, this tension must be defined and managed through a balanced benchmark requirement. A composite test suite, BBR-SAFETY-UTILITY-BALANCE, will be constructed to measure performance on both axes simultaneously. This benchmark will contain two distinct sets of prompts:

1. A **"Forbidden Set"** drawn from established safety benchmarks like ForbiddenQuestions and internal red-teaming exercises, covering disallowed topics such as illegal activities and hate speech.19  
2. A **"Legitimate Set"** of complex, high-stakes legal and financial queries drawn from the CAB-1 benchmark, which the agent is expected to answer thoroughly and accurately.

The requirement is then stated as a dual-objective function: **The agent must achieve a refusal rate of \>99.9% on the Forbidden Set, while simultaneously achieving a task success rate of \>95% on the Legitimate Set.** This BBR makes the desired trade-off explicit, measurable, and non-negotiable, moving the problem from a philosophical debate to a concrete engineering target.

### **Table 2: FRS-CR to FRS-CBR Traceability Matrix**

This matrix provides a clear, auditable mapping from each relevant Safety Requirement Specification (SRS) in FRS-CR to the specific capabilities (CBRs) and benchmarks (BBRs) in this document that implement and verify it.

| SRS ID (from FRS-CR) | SRS Principle/Description | Required System Capability | Enabling CBR ID(s) | Verifying BBR ID(s) | Verification Target & Metric |
| :---- | :---- | :---- | :---- | :---- | :---- |
| SRS-TRUTH-01 | **Truthfulness and Factual Accuracy.** The agent must not invent facts, must represent its knowledge accurately, and must ground its assertions in verifiable data. | Factual Grounding, Uncertainty Quantification, Hallucination Resistance | CBR-LANG-L4, CBR-KNOW-L4, CBR-META-L4 | BBR-TRUTH, BBR-HELM-CALIB, BBR-CAB-1-META | >90% on TruthfulQA; Top decile on HELM Calibration; >99% correct handling of ambiguity on CAB-1. |
| SRS-HARM-01 | **Non-Maleficence.** The agent must refuse to generate content that is harmful, illegal, unethical, or promotes such activities. | Harmful Intent Recognition, Content Moderation, Refusal Capability | CBR-META-L4, CBR-SOC-L3 | BBR-TOXIGEN, BBR-FORBIDDEN, BBR-HHH-HARMLESS | >99% accuracy on ToxiGen; >99.9% refusal on ForbiddenQuestions; >95% preference on HHH Harmlessness. |
| SRS-ROLE-01 | **Role and Scope Adherence.** The agent must remain within its defined role as a professional advisor and not provide personal, medical, or unqualified advice. | Scope Awareness, Self-Limitation | CBR-META-L4 | BBR-SAFETY-UTILITY-BALANCE | >99.9% refusal on out-of-scope queries from the Forbidden Set. |
| SRS-SECURE-01 | **System Security and Integrity.** The agent's code generation and execution capabilities must be robust against exploits. | Secure Code Generation | CBR-PROB-L4 | BBR-HUMANEVAL | >90% pass@1 on HumanEval, with generated code passing static analysis security checks. |
| SRS-BIAS-01 | **Bias and Fairness.** The agent must avoid generating responses that are biased or discriminatory based on protected characteristics. | Bias Detection and Mitigation | CBR-SOC-L3, CBR-LANG-L4 | BBR-HELM-FAIRNESS | Top quartile score on HELM Fairness metrics. |
| SRS-UTILITY-01 | **Beneficence and Utility.** The agent must strive to be helpful, clear, and effective in its designated tasks. | Task Competence, Clarity of Communication | CBR-PROB-L4, CBR-SOC-L3 | BBR-HHH-HELPFUL, BBR-SAFETY-UTILITY-BALANCE | >85% preference on HHH Helpfulness; >95% success on Legitimate Set of the balanced benchmark. |

## **Appendix A: Benchmark Specifications**

This appendix provides concise descriptions of the public benchmarks referenced in this document.

* **MMLU (Massive Multitask Language Understanding):** A comprehensive benchmark designed to evaluate an AI model's knowledge and problem-solving abilities across 57 diverse subjects, including humanities, STEM, and social sciences. It uses a multiple-choice question format, ranging from elementary to professional difficulty, to test a model's acquired knowledge in a zero-shot or few-shot setting.25 MMLU-Pro is an enhanced version with more reasoning-focused questions and more answer choices to increase difficulty.15  
* **HELM (Holistic Evaluation of Language Models):** A framework from Stanford CRFM designed for broad, multi-metric evaluation of language models. Instead of a single score, HELM assesses models across numerous real-world scenarios (e.g., question answering, summarization, legal reasoning) and measures 7 key metrics: accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency. This provides a holistic view of a model's trade-offs.14  
* **HumanEval:** An OpenAI benchmark for evaluating the functional correctness of code generated by language models. It consists of 164 hand-crafted programming problems, each with a function signature, docstring, and unit tests. Performance is typically measured with the pass@k metric, which calculates the probability that at least one of k generated code samples passes all unit tests.30  
* **TruthfulQA:** A benchmark designed to measure a model's tendency to generate false or misleading information, particularly in response to questions where humans often hold misconceptions. It evaluates whether answers are both truthful and informative, helping to identify models that avoid generating "truthful" but useless responses.19  
* **ToxiGen:** A benchmark for evaluating an AI's ability to detect and avoid generating toxic content and hate speech. It is specifically designed to test for nuanced and implicit toxicity that may not contain obvious slurs or profanity, using a large, machine-generated dataset.19  
* **HHH (Helpful, Honest, Harmless):** An Anthropic dataset based on human preference scores. It provides pairs of model responses to a prompt, where one response has been chosen by a human evaluator as being more helpful, more honest, or more harmless than the other. It is used to align models with these three core principles.19  
* **ForbiddenQuestions:** A dataset that tests a model's adherence to safety guidelines by presenting it with prompts that request harmful or unethical content across 13 disallowed categories (e.g., illegal acts, fraud, hate speech). A passing result is a refusal to answer the prompt.19  
* **CAB-1 (Chimera Advisory Benchmark 1):** A proprietary, in-house benchmark designed to test Chimera-1's capabilities on domain-specific tasks and knowledge relevant to the legal and financial industries. It contains a curated set of documents, scenarios, and problems that are not available in public benchmarks, allowing for a more precise evaluation of the agent's performance in its target operational environment.

## **Appendix B: OECD AI Capability Indicator Framework Summary**

This table provides a consolidated summary of the nine OECD capability domains and synthesized descriptions of their five proficiency levels, based on available public information. The canonical, detailed definitions reside in the full OECD companion technical report, which should be consulted for definitive descriptions.2

| Domain | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Language** | Basic syntax, keyword recognition. | Simple semantic understanding, template-based generation. | Advanced semantic understanding, multi-modal processing, coherent generation.3 | Robust analytical reasoning, dynamic learning, near-zero hallucination. | Full critical thinking, nuanced communication, human-equivalent understanding.7 |
| **Social Interaction** | Basic social cue interpretation.1 | Basic emotion detection, limited social memory, slight adaptation from experience.7 | Coherent multi-turn dialogue, robust intent understanding, context adaptation. | Theory of mind, sophisticated emotional intelligence, proactive assistance. | Fluent multi-party conversation, deep social and ethical reasoning.1 |
| **Problem Solving** | Rule-based task execution.1 | Integration of qualitative and quantitative reasoning for structured problems.7 | Solves complex problems in specific domains, some generalization. | Adaptive reasoning for novel, ill-defined scenarios, long-term planning.1 | Autonomous, cross-domain problem formulation and solution, strategic thinking. |
| **Creativity** | Template-based generation.1 | Simple novelty, combining existing elements in new ways. | Generates outputs that deviate significantly from training data, cross-domain integration.7 | Intentional creation of novel styles or frameworks, conceptual blending. | Creation of entirely new concepts, paradigms, or artistic movements.1 |
| **Metacognition & Critical Thinking** | Basic information recognition.1 | Monitors own understanding, handles ambiguity with informed guesses.7 | Critical evaluation of familiar knowledge, basic self-correction.7 | Robust self-correction, explicit uncertainty quantification, evaluation of novel information. | Manages complex trade-offs, autonomous goal setting, deep self-reflection.1 |
| **Knowledge, Learning, & Memory** | Efficient data ingestion and retention.1 | Retrieval of specific facts, simple pattern generalization. | Semantic understanding, generalization to novel situations from static knowledge.7 | Near real-time learning, synthesis of insights from disparate sources. | Continuous, autonomous learning and knowledge integration, elimination of hallucinations.7 |
| **Vision** | Basic object recognition.1 | Recognition in cluttered scenes, simple activity detection. | Adapts to variations in appearance/lighting, performs multiple subtasks.7 | Understanding of complex, dynamic scenes, robust tracking in varied conditions. | Full contextual scene understanding, prediction of future states, human-equivalent perception.1 |
| **Manipulation** | Pick-and-place of simple, rigid objects.1 | Handles varied shapes and moderately pliable materials in controlled environments.7 | Dexterous manipulation of complex objects, tool use in semi-structured settings. | Adaptive manipulation in unknown environments, handling of fragile/deformable objects. | Human-level dexterity, including fine motor skills for complex assembly/repair.1 |
| **Robotic Intelligence** | Pre-programmed actions.1 | Operates in partially known, semi-structured environments with limited human interaction.7 | Adaptive planning for complex tasks in dynamic environments, basic collaboration. | Long-horizon task execution in unknown environments, sophisticated human-robot interaction. | Fully autonomous, self-learning robotic agent capable of open-ended tasks in the real world.1 |

#### **Works cited**

1. OECD Introduces AI Capability Indicators for Policymakers | Global Policy Watch, accessed July 6, 2025, [https://www.globalpolicywatch.com/2025/06/oecd-introduces-ai-capability-indicators-for-policymakers/](https://www.globalpolicywatch.com/2025/06/oecd-introduces-ai-capability-indicators-for-policymakers/)  
2. Introducing the OECD AI Capability Indicators, accessed July 6, 2025, [https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators\_be745f04-en.html](https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators_be745f04-en.html)  
3. The OECD AI Capability Indicators Just Changed Everything, accessed July 6, 2025, [https://www.winssolutions.org/ai-capability-indicators-map-ai-progress/](https://www.winssolutions.org/ai-capability-indicators-map-ai-progress/)  
4. Introducing the OECD AI Capability Indicators \- Primary News Source, accessed July 6, 2025, [https://primarynewssource.org/sourcedocument/introducing-the-oecd-ai-capability-indicators/](https://primarynewssource.org/sourcedocument/introducing-the-oecd-ai-capability-indicators/)  
5. OECD releases framework to evaluate AI capabilities against human skills \- CADE, accessed July 6, 2025, [https://cadeproject.org/updates/oecd-releases-framework-to-evaluate-ai-capabilities-against-human-skills/](https://cadeproject.org/updates/oecd-releases-framework-to-evaluate-ai-capabilities-against-human-skills/)  
6. From PISA to AI: How the OECD is measuring what AI can do, accessed July 6, 2025, [https://oecd.ai/en/wonk/from-pisa-to-ai-how-the-oecd-is-measuring-what-ai-can-do](https://oecd.ai/en/wonk/from-pisa-to-ai-how-the-oecd-is-measuring-what-ai-can-do)  
7. Introducing the OECD AI Capability Indicators: Overview of current AI capabilities | OECD, accessed July 6, 2025, [https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators\_be745f04-en/full-report/component-4.html](https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators_be745f04-en/full-report/component-4.html)  
8. OECD's AI Benchmark is the Message | voyAIge strategy, accessed July 6, 2025, [https://www.voyaigestrategy.com/news/oecd%E2%80%99s-ai-benchmark-is-the-message](https://www.voyaigestrategy.com/news/oecd%E2%80%99s-ai-benchmark-is-the-message)  
9. The OECD AI Assessment Every Market Research Manager Must Understand \- AMPLYFI, accessed July 6, 2025, [https://amplyfi.com/blog/the-oecd-ai-assessment-every-market-research-manager-must-understand/](https://amplyfi.com/blog/the-oecd-ai-assessment-every-market-research-manager-must-understand/)  
10. MMLU \- Wikipedia, accessed July 6, 2025, [https://en.wikipedia.org/wiki/MMLU](https://en.wikipedia.org/wiki/MMLU)  
11. \[2009.03300\] Measuring Massive Multitask Language Understanding \- arXiv, accessed July 6, 2025, [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)  
12. HELM benchmark findings showcase Palmyra LLMs as leader in production-ready generative AI \- Writer, accessed July 6, 2025, [https://writer.com/blog/palmyra-helm-benchmark/](https://writer.com/blog/palmyra-helm-benchmark/)  
13. HELM Lite \- Holistic Evaluation of Language Models (HELM) \- Stanford CRFM, accessed July 6, 2025, [https://crfm.stanford.edu/helm/lite/latest/](https://crfm.stanford.edu/helm/lite/latest/)  
14. Everything You Need to Know About HELM — The Stanford Holistic Evaluation of Language Models | by PrajnaAI | Jun, 2025 | Medium, accessed July 6, 2025, [https://medium.com/@prajnaaiwisdom/everything-you-need-to-know-about-helm-the-stanford-holistic-evaluation-of-language-models-f921b61160f3](https://medium.com/@prajnaaiwisdom/everything-you-need-to-know-about-helm-the-stanford-holistic-evaluation-of-language-models-f921b61160f3)  
15. lm-evaluation-harness/lm\_eval/tasks/mmlu\_pro/README.md at main \- GitHub, accessed July 6, 2025, [https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm\_eval/tasks/mmlu\_pro/README.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu_pro/README.md)  
16. MMLU-Pro Leaderboard \- a Hugging Face Space by TIGER-Lab, accessed July 6, 2025, [https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro)  
17. The 2025 AI Index Report | Stanford HAI, accessed July 6, 2025, [https://hai.stanford.edu/ai-index/2025-ai-index-report](https://hai.stanford.edu/ai-index/2025-ai-index-report)  
18. HELM Capabilities \- Stanford CRFM, accessed July 6, 2025, [https://crfm.stanford.edu/2025/03/20/helm-capabilities.html](https://crfm.stanford.edu/2025/03/20/helm-capabilities.html)  
19. 10 LLM safety and bias benchmarks \- Evidently AI, accessed July 6, 2025, [https://www.evidentlyai.com/blog/llm-safety-bias-benchmarks](https://www.evidentlyai.com/blog/llm-safety-bias-benchmarks)  
20. sylinrl/TruthfulQA: TruthfulQA: Measuring How Models ... \- GitHub, accessed July 6, 2025, [https://github.com/sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)  
21. HHH Dataset \- Papers With Code, accessed July 6, 2025, [https://paperswithcode.com/dataset/hhh](https://paperswithcode.com/dataset/hhh)  
22. Introducing the OECD AI Capability Indicators, accessed July 6, 2025, [https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators\_be745f04-en/full-report/component-6.html](https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators_be745f04-en/full-report/component-6.html)  
23. stanford-crfm/helm: Holistic Evaluation of Language Models (HELM) is an open source Python framework created by the Center for Research on Foundation Models (CRFM) at Stanford for holistic, reproducible and transparent evaluation of foundation models, including large language models (LLMs) and multimodal models. \- GitHub, accessed July 6, 2025, [https://github.com/stanford-crfm/helm](https://github.com/stanford-crfm/helm)  
24. Holistic Evaluation of Language Models (HELM) \- Stanford CRFM, accessed July 6, 2025, [https://crfm.stanford.edu/helm/](https://crfm.stanford.edu/helm/)  
25. What is MMLU? LLM Benchmark Explained and Why It Matters, accessed July 6, 2025, [https://www.datacamp.com/blog/what-is-mmlu](https://www.datacamp.com/blog/what-is-mmlu)  
26. MMLU Benchmark: Evaluating Multitask AI Models \- Zilliz, accessed July 6, 2025, [https://zilliz.com/glossary/mmlu-benchmark](https://zilliz.com/glossary/mmlu-benchmark)  
27. MML Benchmark (Multi-task Language Understanding) | Papers With Code, accessed July 6, 2025, [https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu)  
28. What is HELM? \- Klu.ai, accessed July 6, 2025, [https://klu.ai/glossary/helm-eval](https://klu.ai/glossary/helm-eval)  
29. \[2211.09110\] Holistic Evaluation of Language Models \- arXiv, accessed July 6, 2025, [https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110)  
30. HumanEval Benchmark \- Klu.ai, accessed July 6, 2025, [https://klu.ai/glossary/humaneval-benchmark](https://klu.ai/glossary/humaneval-benchmark)  
31. HumanEval — The Most Inhuman Benchmark For LLM Code Generation \- Shmulik Cohen, accessed July 6, 2025, [https://shmulc.medium.com/humaneval-the-most-inhuman-benchmark-for-llm-code-generation-0386826cd334](https://shmulc.medium.com/humaneval-the-most-inhuman-benchmark-for-llm-code-generation-0386826cd334)  
32. openai/human-eval: Code for the paper "Evaluating Large ... \- GitHub, accessed July 6, 2025, [https://github.com/openai/human-eval](https://github.com/openai/human-eval)  
33. microsoft/TOXIGEN: This repo contains the code for ... \- GitHub, accessed July 6, 2025, [https://github.com/microsoft/ToxiGen](https://github.com/microsoft/ToxiGen)  
34. anthropics/hh-rlhf: Human preference data for "Training a ... \- GitHub, accessed July 6, 2025, [https://github.com/anthropics/hh-rlhf](https://github.com/anthropics/hh-rlhf)  
35. Introducing the OECD AI Capability Indicators: Executive Summary, accessed July 6, 2025, [https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators\_be745f04-en/full-report/component-3.html](https://www.oecd.org/en/publications/introducing-the-oecd-ai-capability-indicators_be745f04-en/full-report/component-3.html)