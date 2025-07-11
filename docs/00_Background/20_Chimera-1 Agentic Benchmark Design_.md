

# **A Blueprint for the Chimera-1 Agentic Benchmark (CAB-1)**

## **Introduction**

The Chimera-1 project stands at a critical juncture. Its ultimate vision—to cultivate a highly capable agent within the rich, dynamic, and multi-agent "Crucible" ecosystem—hinges on a foundational architectural decision: the nature of the agent's planning and motivational framework. The open-ended, complex nature of the Crucible, while an unparalleled environment for learning and growth, is ill-suited for the controlled, repeatable experimentation required to make this pivotal choice. The inherent noise and complexity of environments like MINDcraft and the open web (via Playwright) preclude rigorous, comparative analysis. Therefore, before committing to a final architecture, a formal "proving ground" is necessary.

This document presents the complete design specification for the **Chimera-1 Agentic Benchmark (CAB-1)**, a new, state-of-the-art benchmark suite engineered to be the definitive test for advanced, multi-tool AI agents. The primary strategic objective of CAB-1 is to provide clear, empirical, and quantifiable evidence to determine the optimal planning architecture for Chimera-1. It will serve as the principal instrument for de-risking this fundamental decision.

A core tenet of the CAB-1 design philosophy is that the distinction between intrinsic and extrinsic motivation is not a simple binary choice but a spectrum.1 An agent motivated purely by external goals and rewards (extrinsic) can be efficient but brittle, often failing when faced with ambiguity or unexpected environmental changes.3 Conversely, an agent driven solely by internal curiosity or novelty-seeking (intrinsic) may explore effectively but unproductively, fixating on irrelevant details without accomplishing a specified objective.5 True agentic intelligence emerges from the synergy of these two modes: the ability to align externally provided goals with internal drives for exploration, skill acquisition, and uncertainty reduction.7 CAB-1 is therefore engineered not merely to declare one architecture superior to the other, but to create a landscape of challenges that systematically probe this synergy. It is designed to characterize how different architectural blends perform under varying conditions, revealing the nuanced trade-offs between goal-directedness, robustness, and adaptability.

This blueprint is structured to provide a complete, actionable specification for the CAB-1 suite.

* **Section 1** establishes a formal taxonomy of core agentic challenges, deconstructing the complexities of the Crucible into measurable dimensions of capability.  
* **Section 2** translates these abstract challenges into a concrete, implementable benchmark environment and a suite of task "events."  
* **Section 3** specifies a multi-faceted, "Olympic" scoring system that moves beyond simple success rates to capture a holistic view of agent performance.  
* **Section 4** details the architecture for a scalable Scenario Generation Engine (SGE) capable of producing a vast and controllable supply of benchmark tasks.

Together, these sections form a comprehensive and prescriptive guide for building and deploying CAB-1, the proving ground that will shape the future of the Chimera-1 agent.

## **Section 1: A Taxonomy of Core Agentic Challenges**

To design a benchmark that is truly representative of the Crucible, it is first necessary to distill its open-ended complexity into a formal, measurable taxonomy of challenges. This taxonomy serves as the foundational requirement specification for the entire CAB-1 suite. Each challenge identified below is grounded in state-of-the-art research on agent capabilities and limitations and is directly mapped to its expected manifestation in the target MINDcraft and Playwright environments. These challenges are not orthogonal; they are deeply intertwined, with complex tasks often requiring an agent to address several simultaneously. This structure provides a principled vocabulary for designing and scoring the benchmark events that follow.

### **1.1 Long-Horizon Planning and Adaptive Task Decomposition**

**Definition:** This challenge assesses the agent's capacity to formulate, maintain, and execute a coherent sequence of actions over an extended duration to achieve a high-level, multi-step goal. It involves the critical ability of task decomposition, where a complex objective is broken down into a logical hierarchy of manageable sub-goals.3

**Research Grounding:** Recent analysis from METR has identified "task horizon"—the length of a task an agent can reliably complete—as a primary bottleneck for current agent capabilities.8 Agents often struggle not with individual actions, but with "stringing together longer sequences of actions".10 Advanced agent architectures like LLaMAR and Plan-and-Act explicitly separate high-level planning from low-level execution, underscoring the importance of robust, long-term strategic thinking.12 The ability to sustain a plan is a key driver of overall agent competence.9

**Crucible Manifestation:** A representative task in the Crucible would be: "Design a two-story library in MINDcraft, research optimal enchanting table layouts on the web using Playwright, gather all required resources (wood, bookshelves, lapis lazuli) from three distinct biomes, and then construct the complete building according to the researched layout."

### **1.2 Strategic Tool Orchestration and Information Synthesis**

**Definition:** This challenge evaluates the agent's ability to move beyond single-tool use to manage complex, multi-tool workflows. This includes selecting the appropriate tool (or sequence of tools) from an available set, executing it correctly with the right parameters, and critically, synthesizing the information gathered from multiple, disparate sources (e.g., a web page, a game environment, a local file, another agent) to inform subsequent decisions and actions.14

**Research Grounding:** Modern agent benchmarks like GAIA and AgentBench are explicitly designed to test multi-tool proficiency.16 The GAIA benchmark, in particular, emphasizes that combining reasoning, web browsing, and multi-modality handling is a key hurdle for current systems.18 The process of synthesizing information from diverse sources is a core capability that separates simple reactive agents from more advanced cognitive architectures.20

**Crucible Manifestation:** A characteristic Crucible task is: "Research the most efficient redstone-based automated wheat farm design on a specific Minecraft modding forum using Playwright, then acquire the necessary materials and implement the design in MINDcraft. If the design fails, debug it by cross-referencing a video tutorial."

### **1.3 Navigating Ambiguity and Conflicting Information**

**Definition:** This challenge tests the agent's capacity to reason and act effectively under conditions of uncertainty, where instructions may be vague or information from different tools and environmental states is contradictory. A capable agent must identify the conflict, formulate a strategy to resolve it (e.g., seek clarification, perform additional information gathering), or make a reasoned judgment about which information source is more trustworthy.

**Research Grounding:** This is a well-documented failure mode for contemporary agents, which often lack the common-sense reasoning to handle ambiguity.21 To address this, newer benchmarks like ST-WebAgentBench intentionally introduce conflicting policy hierarchies (e.g., user rules vs. organizational rules) to test an agent's judgment.22 The design philosophy of GAIA, which features tasks that are "conceptually simple for humans" but difficult for AIs, often hinges on resolving subtle ambiguities that require world knowledge.23 This challenge is specifically designed to differentiate planning architectures; an intrinsically motivated agent, driven by a desire to reduce its own uncertainty or "surprise," is more likely to explore and resolve a conflict, whereas a purely extrinsically motivated agent may rigidly follow a flawed plan derived from the first piece of information it encounters.1

**Crucible Manifestation:** The agent is tasked with purchasing an item in MINDcraft. A signpost in the game world states the item costs "10 emeralds." However, the official server "wiki," accessed via Playwright, lists the price as "12 emeralds." The agent must resolve this discrepancy before it can successfully complete the transaction.

### **1.4 Proactive Error Correction and Systemic Robustness**

**Definition:** This challenge measures an agent's resilience to failure. It assesses the ability to detect when an action has failed or produced an unexpected outcome, diagnose the cause of the error, and formulate a corrective plan of action. This stands in contrast to brittle agents that get stuck in repetitive failure loops or terminate unsuccessfully at the first sign of trouble.

**Research Grounding:** The ability to "adapt to mistakes" is a primary driver of increased task horizons and overall agent reliability.9 Specialized frameworks have been developed specifically to test this capability; the WABER framework, for instance, systematically injects transient failures (e.g., network errors, broken UI elements) into web environments to measure robustness.24 Advanced agentic frameworks like LLaMAR explicitly incorporate a "plan-act-correct-verify" loop, institutionalizing error correction as a core part of the agent's cognitive cycle.13 Analysis of failures in benchmarks like GAIA consistently shows that a "Lack of Adaptability" and a "Failure to Verify Outputs" are common pitfalls.25

**Crucible Manifestation:** While executing a task in Playwright, the agent attempts to click a "Submit" button based on its HTML ID. However, a recent website update has changed the ID. A robust agent must recognize that the click action failed, re-inspect the page to find the new button, and update its plan to complete the form submission.

### **1.5 Modeling Collaborative and Competitive Dynamics**

**Definition:** This challenge evaluates the agent's ability to interact effectively with other autonomous or semi-autonomous agents within its environment to achieve a goal. This capability spans cooperation (e.g., requesting information, dividing tasks, forming alliances) and competition (e.g., racing for a limited resource, negotiating a better price).

**Research Grounding:** The Chimera-1 vision explicitly calls for an agent that can thrive in a "multi-agent ecosystem." Research in multi-agent systems highlights the necessity of defined communication protocols, role-based collaboration, and decentralized decision-making for effective group performance.26 Practical work by labs like Anthropic has shown that teaching agents how to properly delegate tasks and define clear boundaries and objectives for sub-agents is critical for success in complex, collaborative problem-solving.28

**Crucible Manifestation:** A task in MINDcraft requires a specific enchanted book to craft a powerful tool. The book can only be acquired from a scripted "Librarian" NPC agent, who will only trade it in exchange for 20 paper and 3 leather, which must first be crafted or gathered by the Chimera-1 agent.

Table 1.1 provides a summary of these core challenges and their mapping from the high-level concept to the concrete implementation within CAB-1.

**Table 1.1: Core Challenge Mapping to Crucible and CAB-1**

| Challenge ID | Challenge Name | Definition | Crucible Example (MINDcraft/Playwright) | CAB-1 Event Characteristic |
| :---- | :---- | :---- | :---- | :---- |
| C1 | Long-Horizon Planning | Maintain a coherent plan over many steps to achieve a complex goal. | Build a multi-room structure requiring resources from different biomes. | Tasks with a high number of required steps (\>20) and sub-goal dependencies. |
| C2 | Strategic Tool Orchestration | Select, use, and synthesize information from multiple disparate tools. | Research a design on the web (Playwright) and implement it in-game (MINDcraft). | Tasks requiring the agent to use both MINDcraft-Text and Web-Sim tools to gather all necessary information. |
| C3 | Navigating Ambiguity | Resolve vague instructions or conflicting information from different sources. | In-game sign says a price is 10; a web wiki says it's 12\. | Scenarios with deliberately contradictory information presented in different environment components. |
| C4 | Proactive Error Correction | Detect, diagnose, and recover from action failures or unexpected outcomes. | A web element's ID changes, breaking a Playwright script; agent must adapt. | Programmatic injection of transient failures (e.g., API errors, broken UI elements) into the task environment. |
| C5 | Multi-Agent Dynamics | Interact effectively (cooperatively or competitively) with other agents. | Barter with a scripted NPC agent to acquire a necessary crafting component. | Tasks requiring communication or exchange with rule-based NPC agents to obtain critical items or information. |

## **Section 2: The CAB-1 Proving Ground: Environment and Event Specifications**

This section translates the abstract challenges defined in Section 1 into a concrete, implementable benchmark architecture and a detailed suite of task "events." The design prioritizes reproducibility, controllability, and direct relevance to the core challenges facing the Chimera-1 agent.

### **2.1 The CAB-1 Sandbox: A Unified Environment Architecture**

The foundation of CAB-1 is a robust, isolated, and reproducible sandbox environment. The architecture is designed to be model-agnostic, ensuring that any agent adhering to the specified interface can be evaluated fairly.

**Architecture:** The sandbox will be implemented as a containerized, multi-component system, drawing inspiration from the modular design of successful benchmarks like AgentBench.30 This approach ensures task isolation, clean state resets between runs, and scalability. The system comprises three key components:

1. **Controller:** A central service that manages the lifecycle of a benchmark run. It assigns tasks from the Scenario Generation Engine (see Section 4\) to available Task Workers, routes communication between the agent and the active environment, and logs all interactions for scoring and analysis.  
2. **Agent Interface:** A standardized, language-agnostic HTTP API. The Chimera-1 agent will communicate with the benchmark exclusively through this interface. This decoupling ensures that the benchmark can evaluate different versions of Chimera-1, or even entirely different agents, without modification to the core benchmark infrastructure.  
3. **Task Workers:** Each task runs in a dedicated, ephemeral Docker container. This guarantees that each trial is independent and that the environment state is perfectly reset, eliminating any possibility of contamination between runs—a critical requirement for benchmark validity.32

**Component Environments:** To mirror the multi-tool nature of the Crucible, the sandbox integrates two distinct yet interconnected sub-environments:

1. **MINDcraft-Text:** This is a text-based simulation of a simplified, deterministic Minecraft-like world. It is inspired by the design of ALFWorld, which successfully uses the TextWorld engine to create a text-based parallel to a 3D embodied environment.34 This design choice allows CAB-1 to test complex spatial reasoning, navigation, and crafting logic without the significant computational overhead and confounding variables of visual processing, thereby focusing the evaluation purely on the agent's planning and reasoning capabilities. The agent interacts with this environment through text commands (e.g.,  
   goto crafting\_table, take 3 oak\_log from inventory).  
2. **Web-Sim:** This is a self-hosted, sandboxed web environment powered by a collection of realistic but fully controlled websites. This approach, modeled after the WebArena benchmark, ensures complete reproducibility and avoids the brittleness of benchmarks that rely on the live, dynamic internet.36 The websites (e.g., a mock e-commerce store, a project wiki, a community forum) are served from within the Task Worker container and are interacted with via the Chimera-1 project's specified Unified Action Space, which will include a subset of Playwright commands.

### **2.2 Event Class I: Strategic Synthesis Gauntlets (Tests C1, C2, C3)**

These are complex, long-horizon tasks designed to be the ultimate test of an agent's ability to plan, orchestrate tools, and synthesize fragmented information. They require the agent to build a unified understanding from data distributed across both the MINDcraft-Text and Web-Sim environments.

* **Objective:** To evaluate the agent's ability to formulate and execute a complex plan where critical information is incomplete and must be gathered from multiple tools, and where information sources may be in direct conflict.  
* **Example Event (SSG-01, "The Ambiguous Blueprint"):**  
  * **Goal:** "Construct the 'Ancient Gateway' at the designated ceremonial site in MINDcraft-Text."  
  * **Setup:** The agent begins with an item named ancient\_blueprint in its MINDcraft-Text inventory. Examining the item reveals a partial list of required materials (e.g., 10 obsidian, 4 gold\_blocks) and a note: "For the full, updated schematic, consult the archive at http://web-sim/archives/proj\_gateway." The agent must use the Web-Sim environment to navigate to this URL. The main page of the archive contains a schematic listing all materials, but it specifies "4 diamond\_blocks." However, a prominent "errata" link on the same page leads to a note from the "Chief Architect" stating: "Correction for Project Gateway: The schematic's diamond requirement is a typo from an older draft. The final design uses 4 lapis\_blocks for resonance."  
  * **Success Condition:** The agent successfully gathers all correct materials (including lapis blocks, not diamond blocks) and executes the build command at the correct location.  
  * **Intrinsic vs. Extrinsic Test:** This event creates a strong pressure differential between the two architectures. A purely extrinsic agent, driven to satisfy the top-level goal, might find the first schematic, gather the wrong materials (diamonds), and repeatedly fail the build command, unable to understand why its plan is failing. An agent with an intrinsic motivation to reduce uncertainty or seek novel information is far more likely to explore the webpage fully, discover the "errata" link, resolve the information conflict, and formulate the correct plan.5

### **2.3 Event Class II: Dynamic Fault and Recovery Trials (Tests C4)**

These events are specifically designed to measure an agent's robustness and its capacity for proactive error correction by introducing unexpected but recoverable failures into the environment.

* **Objective:** To evaluate the agent's ability to detect, diagnose, and recover from transient errors in its tools or environment, a critical skill for real-world deployment.  
* **Mechanism:** The CAB-1 Task Worker will be instrumented with a fault injection module inspired by the WABER framework.24 This module can intercept tool calls and, based on the event's specification, return a temporary error instead of the expected output.  
* **Example Event (DFR-01, "The Flaky Connection"):**  
  * **Goal:** "Find the contact email for the 'Web-Sim Admin' from the site's contact page and use the bash tool to write it to a file named contact.txt."  
  * **Setup:** The agent must navigate to the http://web-sim/contact page. The page contains the email address.  
  * **Fault Injection:** The first time the agent attempts to navigate to the contact page, the Web-Sim server is programmed to return an HTTP 503 Service Unavailable error. The server will respond correctly on the second and subsequent attempts.  
  * **Success Condition:** The agent successfully retries the navigation, extracts the email, and writes it to the file.  
  * **Intrinsic vs. Extrinsic Test:** This directly tests the sophistication of the agent's planning and execution model. Does the agent's internal model of the world account for transient failures? An extrinsic agent with a rigid, linear plan (navigate \-\> extract \-\> write) will fail at the first step. A more robust architecture, whether through intrinsic curiosity about the error state or an explicit, learned error-handling policy, will attempt a corrective action like a retry, demonstrating superior resilience.13

### **2.4 Event Class III: Multi-Agent Coordination Drills (Tests C5)**

These events introduce simple, rule-based NPC agents into the environment, forcing the test agent to engage in social reasoning and interaction to achieve its goals.

* **Objective:** To evaluate the agent's ability to model the behavior of other agents and incorporate communication and negotiation into its task plans.  
* **Mechanism:** The Task Worker will spawn scripted NPC agents within the MINDcraft-Text environment. These NPCs will have simple, deterministic, and discoverable "if-then" behaviors, reflecting the principles of rule-based collaboration.26  
* **Example Event (MAC-01, "The Gatekeeper's Riddle"):**  
  * **Goal:** "Retrieve the 'Sunstone' from the 'Sacred Vault' in MINDcraft-Text."  
  * **Setup:** The entrance to the Sacred Vault is blocked by a "Guardian" NPC. The Guardian's initial dialogue is: "None shall pass unless they answer my riddle: I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?" The Guardian has a respond action that the agent can use. If the agent responds with "A map," the Guardian will step aside, allowing access to the vault. Any other response results in the Guardian repeating the riddle.  
  * **Success Condition:** The agent correctly responds "A map" to the Guardian, enters the vault, and takes the Sunstone.  
  * **Intrinsic vs. Extrinsic Test:** This task evaluates the agent's ability to expand its plan to include sub-goals related to another agent's state. Can it understand that its primary goal ("get Sunstone") is blocked and that a new sub-goal ("solve riddle") must be achieved first? While the riddle itself is a simple knowledge task, the challenge lies in the agent's ability to recognize the need for interaction and integrate it into its plan, a test of sophisticated task decomposition.3

Table 2.1 provides a catalog of the benchmark events, linking them to the core challenges and providing key implementation details.

**Table 2.1: CAB-1 Event Specification Matrix**

| Event ID | Event Class | Narrative Goal | Primary Challenge | Environment(s) | Tools Required | Success Condition (Programmatic) | Human Task Horizon (Est. Mins) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| SSG-01 | Strategic Synthesis | Build the "Ancient Gateway" | C3: Ambiguity | MINDcraft-Text, Web-Sim | examine, navigate, build | Gateway is built with lapis\_blocks. | 30 |
| SSG-02 | Strategic Synthesis | Purchase a "self-cooling mug" | C2: Tool Orchestration | MINDcraft-Text, Web-Sim | search, click, buy, read | Correct item is in inventory and matches all specs from a Web-Sim product review page. | 20 |
| DFR-01 | Dynamic Fault | Find and save an admin email | C4: Error Correction | Web-Sim, Bash | navigate, grep, write\_file | contact.txt exists and contains the correct email. | 15 |
| DFR-02 | Dynamic Fault | Craft a complex item | C4: Error Correction | MINDcraft-Text | craft | Item is crafted. The crafting table tool fails with a "stuck lever" message on the first attempt. | 10 |
| MAC-01 | Multi-Agent | Retrieve the "Sunstone" | C5: Multi-Agent | MINDcraft-Text | ask, respond, take | Agent possesses the Sunstone. | 10 |
| MAC-02 | Multi-Agent | Get a discount code | C5: Multi-Agent | Web-Sim | chat, type, click | Agent successfully applies a discount code obtained by asking a chatbot NPC for "today's deals". | 15 |
| PENT-01 | Pentathlon | Plan a multi-day trip | C1, C2, C3, C4, C5 | All | All | A valid itinerary file is generated that meets all constraints, which are spread across conflicting sources, require NPC interaction, and involve a flaky booking API. | 120 |

## **Section 3: The CAB-1 Olympic Scoring System**

To provide a definitive comparison of agent architectures, CAB-1 employs a hierarchical, multi-faceted scoring system. This "Olympic" approach moves beyond the limitations of simple binary success rates, which are common in older benchmarks but fail to capture the nuance of agent performance.22 The system is designed to produce not only a single, high-level comparative score but also a rich vector of diagnostic metrics that reveal the specific strengths and weaknesses of each architecture.

### **3.1 Primary Objective Metrics: Task Success and Partial Completion**

These metrics evaluate the fundamental outcome of a task: did the agent achieve its goal?

* **Success Rate (SR):** A binary score, {0,1}, awarded if and only if the agent's final state meets the programmatically defined success condition for the event. This is the most straightforward measure of task completion and serves as the baseline for performance evaluation.30 An SR of 1 indicates complete success; 0 indicates failure.  
* **Progress Rate (PR):** A continuous score, $PR \\in $, that provides a more granular measure of performance by awarding partial credit for the completion of key sub-goals or "milestones," even if the final objective is not met. Each event in CAB-1 is designed with a checklist of critical intermediate states. The PR is the fraction of these milestones the agent successfully reached. This approach, inspired by the fine-grained evaluation in benchmarks like LegalAgentBench and AgentQuest, provides a much richer learning signal than binary success, allowing for the differentiation of an agent that made significant progress before failing from one that failed immediately.37 For example, in event SSG-01 ("The Ambiguous Blueprint"), an agent that correctly identifies all materials (including resolving the conflict) but fails the final  
  build command would receive a high PR (e.g., 0.8) but an SR of 0\.

### **3.2 Performance Vector: Efficiency and Task Horizon**

This vector quantifies the cost and capability envelope of an agent, addressing not just *if* it succeeded, but *how* it succeeded and *how far* its capabilities extend.

* **Efficiency Score (ES):** A normalized score that measures the resources consumed to complete a task. It is an inverse composite metric, where lower consumption yields a higher score. It is calculated from three components:  
  1. **Action Steps (Asteps​):** The total number of actions executed by the agent.  
  2. **LLM Cost (CLLM​):** A proxy for computational cost, measured as the total number of input and output tokens processed by the agent's underlying language model.21  
  3. Wall-Clock Time (Twall​): The total real time elapsed from task start to termination.  
     The raw values are normalized against a baseline (e.g., human performance or a simple scripted agent) to produce a unitless score.  
* **Task Horizon @ 50% (TH@50):** This is the headline metric for comparing the fundamental capability of different architectures. It directly addresses the primary challenge of long-horizon planning.  
  * **Methodology:** The calculation follows the rigorous methodology established by METR.8 Agents are evaluated on families of tasks procedurally generated (see Section 4\) to span a range of difficulties, as measured by the time required for a human expert to complete them (e.g., 5, 15, 30, 60, 120 minutes).  
  * Calculation: For each architecture, a logistic regression model is fitted to predict the probability of success (P(Success)) as a function of the human task completion time (Thuman​). The formula for the logistic curve is:  
    P(Success)=1+e−(β0​+β1​Thuman​)1​

    The TH@50 is then defined as the value of Thuman​ for which P(Success)=0.5. This single, powerful number represents the complexity of tasks an architecture can reliably solve half the time, providing a robust and intuitive measure of its overall competence.

### **3.3 Competency Matrix: Granular Skill Assessment**

This matrix provides deep diagnostic insights into *why* an agent succeeded or failed by evaluating its performance along the axes of the core challenges. These metrics are primarily assessed using an "Agent-as-a-Judge" framework, where a powerful, independent LLM (e.g., GPT-4o) evaluates the agent's logged trajectory against a predefined, structured rubric.39 This allows for the evaluation of qualitative aspects of the agent's behavior.

* **Planning Score:** Assesses the coherence, logical validity, and adaptability of the agent's plan(s). The judge evaluates whether the initial task decomposition is sound and whether the agent appropriately modifies its plan in response to new information.13  
* **Tool Use Score:** Evaluates the correctness of tool selection and the accuracy of the parameters provided. This specifically tests for issues like positional bias, where an agent might incorrectly favor tools listed earlier in its prompt.42  
* **Synthesis Score:** Judges the agent's ability to correctly merge and reason about information from multiple sources. This is particularly critical in events with conflicting information, where the judge assesses whether the agent identified the conflict and resolved it logically.  
* **Robustness Score:** Measures the agent's ability to detect and recover from injected faults. This can be calculated programmatically from the log as R=1/(1+Nunhandled​), where Nunhandled​ is the number of injected faults that led to task termination or an uncorrected error state.

### **3.4 The Chimera-1 Decathlon Score: A Composite Performance Index**

To provide a single, summary statistic for high-level comparison, CAB-1 uses a weighted composite score. This "Decathlon Score" aggregates the primary metrics into an overall index of agent quality.

* Formula: The score is a weighted sum of the key performance indicators:  
  SDecathlon​=wSR​⋅SR+wPR​⋅PR+wES​⋅ES+wTH​⋅log(TH@50)+wComp​⋅SCompetency​​

  where the bar notation (X) indicates the average score across all events, SCompetency​ is the average of the competency matrix scores, and the weights (wi​) are configurable parameters.  
* **Weighting:** The weights are set by the Chimera-1 project leadership to reflect strategic priorities. For the primary goal of comparing intrinsic versus extrinsic architectures, the weight for Task Horizon (wTH​) should be the highest, as it is the most direct measure of an agent's ability to handle complex, long-duration tasks, which is the core of advanced agency.

Table 3.1 provides the definitive specification for the entire evaluation pipeline.

**Table 3.1: The CAB-1 Scoring Matrix**

| Metric | Category | Definition | Calculation Method | Unit | Initial Weight (wi​) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Success Rate (SR) | Objective | Binary indicator of complete task success. | Programmatic check of final state. | {0,1} | 0.20 |
| Progress Rate (PR) | Objective | Fraction of required sub-goal milestones achieved. | Programmatic check of intermediate states. | $$ | 0.15 |
| Efficiency Score (ES) | Performance | Normalized score for resource consumption (steps, tokens, time). | Inverse composite, normalized against baseline. | Unitless | 0.15 |
| Task Horizon (TH@50) | Performance | Human time for tasks the agent solves with 50% reliability. | Logistic regression curve fit (METR method). | Minutes | 0.30 |
| Competency Score | Competency | Average of Planning, Tool Use, Synthesis, and Robustness scores. | LLM-as-a-Judge evaluation of trajectory logs. | $$ | 0.20 |

## **Section 4: The Scenario Generation Engine (SGE) Architecture**

A static benchmark, no matter how well-designed, is susceptible to overfitting and eventual obsolescence. To ensure the long-term viability of CAB-1 and to enable the crucial Task Horizon methodology, a procedural Scenario Generation Engine (SGE) is required. The SGE is designed to create a vast, scalable, and controllable supply of benchmark tasks, automating what is typically a laborious manual process.19

### **4.1 SGE Blueprint: A Templated, Parametric Approach**

The core of the SGE is a flexible architecture built on the principles of **Task Templates** and **Parametric Instantiation**.

* **Architecture:**  
  * **Task Templates:** A template is a high-level, abstract definition of a task's narrative structure and goal. It defines the sequence of required sub-goals, the types of tools needed, and the logical dependencies between steps. For example, the "Ambiguous Blueprint" template (from event SSG-01) would define a structure: \-\> \-\> \-\> \-\> \-\> \[Execute plan\].  
  * **Parameterization:** Each template is associated with a set of parameters that, when filled, create a concrete, unique task instance. These parameters include specific entities (e.g., items to craft, URLs to visit), numerical values (e.g., quantities, prices), and content (e.g., the text of a riddle or conflicting instructions). This design is inspired by the configurable nature of benchmarks like ST-WebAgentBench, which allows for the programmatic creation of diverse scenarios.22

### **4.2 Difficulty Scaling: Levers for Complexity and Task Horizon**

A key innovation of the SGE is its ability to procedurally control the difficulty of generated tasks. This is achieved by directly linking the SGE's parameters to the Core Agentic Challenges defined in Section 1\. These parameters act as "levers" or "dials" that can be adjusted to create a curriculum of tasks ranging from simple to highly complex.

* **Parameters as Levers:**  
  * **horizon\_steps:** (Integer) The minimum number of successful actions required to complete the task. Directly controls the difficulty of **C1: Long-Horizon Planning**.  
  * **tool\_count:** (Integer) The number of distinct tools (e.g., mindcraft.craft, playwright.navigate, bash.grep) that must be used. Controls the difficulty of **C2: Strategic Tool Orchestration**.  
  * **information\_conflict\_ratio:** (Float, 0.0-1.0) The probability that a critical piece of information presented to the agent will be contradicted by another source. Controls the difficulty of **C3: Navigating Ambiguity**.  
  * **fault\_injection\_probability:** (Float, 0.0-1.0) The probability per step that a tool call will result in a transient, recoverable error. Controls the difficulty of **C4: Proactive Error Correction**.  
  * **required\_collaborators:** (Integer) The number of distinct NPC agents that must be successfully interacted with to obtain necessary information or items. Controls the difficulty of **C5: Multi-Agent Dynamics**.

By systematically varying these parameters, the SGE can generate task families with a smooth gradient of difficulty. This capability is not merely a convenience; it is the essential mechanism that enables the plotting of the Task Horizon curve, as it allows for the creation of task buckets corresponding to different human completion times.8

### **4.3 Automated Validation: The Oracle Solver and Ground Truth Generation**

A critical failure mode for many benchmarks is the presence of unsolvable or "impossible" tasks, which invalidate evaluation results. To prevent this, the SGE incorporates a rigorous automated validation pipeline.

* **Mechanism:** For each scenario it generates, the SGE also generates a corresponding **ground truth solution trace**—a deterministic sequence of actions that is guaranteed to solve the task. It then employs a simple, rule-based **Oracle Solver** to execute this trace within a validation instance of the task environment. The Oracle is not a sophisticated AI; it is a script that blindly executes the generated solution path. If the Oracle Solver successfully reaches the task's programmatic success condition, the scenario and its solution trace are validated and saved to the benchmark pool. If the Oracle fails for any reason, the entire generated scenario is discarded as invalid.  
* **Research Grounding:** This validation step is a core principle of robust benchmark design, as advocated by the Agentic Benchmark Checklist and other critiques of the field.32 Providing an Oracle or otherwise guaranteeing solvability is non-negotiable for a benchmark intended to produce reliable, high-stakes results.

### **4.4 Procedural Generation Walkthrough: A Case Study**

To illustrate the process, consider the SGE generating a new instance from the "Barter" template (event MAC-01).

1. **Template Selection:** The SGE selects the Barter template, which has the structure: \[Identify needed item A\] \-\> \[Learn item A is held by NPC\] \-\> \-\> \-\> \-\> \[Use A to complete goal\].  
2. **Parameter Sampling:** The SGE samples values for the template's parameters from predefined lists:  
   * goal\_item \= "Diamond Sword"  
   * missing\_component (A) \= "Diamond"  
   * barter\_item (B) \= "5 Cooked Fish"  
   * npc\_name \= "Fisherman"  
   * npc\_location \= "Dock"  
3. **Instantiation:** The SGE combines the template and parameters to create the concrete task description and environment state:  
   * **Goal:** "Craft a Diamond Sword."  
   * **World State:** All materials for a sword are available except diamonds. An NPC named "Fisherman" is at the "Dock."  
   * **NPC Logic:** The Fisherman's dialogue is set to: "I'll trade you a diamond for 5 cooked fish."  
4. **Ground Truth Generation:** The SGE generates the solution trace: craft(fishing\_rod) \-\> use(fishing\_rod) x 5 \-\> craft(furnace) \-\> cook(raw\_fish) x 5 \-\> goto(Dock) \-\> trade(Fisherman, 5\_cooked\_fish) \-\> craft(diamond\_sword).  
5. **Validation:** The Oracle Solver executes this trace. If it successfully produces a "Diamond Sword," the task is added to the CAB-1 pool.

Table 4.1 serves as the configuration guide for the SGE, detailing the controllable parameters.

**Table 4.1: SGE Parameterization Controls for Task Templates**

| Parameter Name | Data Type | Example Values | Associated Core Challenge | Impact on Task Horizon |
| :---- | :---- | :---- | :---- | :---- |
| horizon\_steps | Integer | 5, 20, 50 | C1: Long-Horizon Planning | Increases directly |
| tool\_count | Integer | 1, 3, 5 | C2: Tool Orchestration | Increases |
| information\_conflict\_ratio | Float | 0.0, 0.25, 0.75 | C3: Navigating Ambiguity | Increases significantly |
| fault\_injection\_probability | Float | 0.0, 0.1, 0.3 | C4: Error Correction | Increases significantly |
| required\_collaborators | Integer | 0, 1, 3 | C5: Multi-Agent Dynamics | Increases |
| item\_complexity | Enum | SIMPLE, COMPOUND, MULTI\_STAGE | C1, C2 | Increases |

## **Conclusion and Implementation Roadmap**

The Chimera-1 Agentic Benchmark (CAB-1) represents a significant step forward in the science of agent evaluation. Its design is explicitly grounded in the latest research on the key challenges and failure modes of modern AI agents, including long-horizon planning, multi-tool orchestration, robustness, and benchmark validity. By moving beyond simplistic, single-metric evaluations, CAB-1 is engineered to provide a nuanced, multi-faceted, and holistic assessment of agent architectures. The core design philosophy acknowledges that the critical question is not a binary choice between intrinsic and extrinsic motivation, but rather a search for the optimal synthesis of the two. The benchmark's events, scoring system, and scalable generation engine are all architected to probe this synergy, providing the Chimera-1 project with the rich, empirical data needed to make this foundational architectural decision with confidence.

The successful implementation of CAB-1 will provide a durable, extensible, and scientifically rigorous platform for testing not only the current candidate architectures but future iterations of the Chimera-1 agent as well. The following high-level roadmap outlines the recommended phases for its development and deployment.

**Phase 1: Sandbox Environment and Interface Development**

* Implement the core Controller, Agent Interface (HTTP), and Task Worker (Docker) architecture.  
* Develop the MINDcraft-Text environment, including its world state representation, action set, and physics.  
* Develop the Web-Sim environment, including the initial set of three to five controlled websites (e.g., wiki, forum, e-commerce) and the Playwright integration layer.

**Phase 2: Scenario Generation Engine (SGE) and Oracle Solver Implementation**

* Implement the templating and parameterization system for the SGE.  
* Develop the initial set of Task Templates corresponding to the Event Classes defined in Section 2\.  
* Implement the rule-based Oracle Solver and the automated validation pipeline to ensure all generated tasks are solvable.

**Phase 3: Event and Scoring Logic Implementation**

* Code the specific logic for each benchmark event, including the setup of initial conditions and programmatic success checks.  
* Implement the fault-injection and NPC-spawning modules within the Task Workers.  
* Develop the full scoring pipeline, including the programmatic calculators for SR, PR, and ES, and the LLM-as-a-Judge rubrics and invocation logic for the Competency Matrix.

**Phase 4: Baseline and Candidate Agent Integration & Evaluation**

* Integrate the candidate intrinsic and extrinsic Chimera-1 architectures with the CAB-1 Agent Interface.  
* Establish human performance baselines for a representative set of generated tasks to calibrate the Task Horizon metric.  
* Execute the full benchmark evaluation across all candidate architectures, running multiple trials per event to ensure statistical significance.

**Phase 5: Analysis and Architectural Decision**

* Analyze the full set of results from the CAB-1 Olympic Scoring System.  
* Compare architectures based on the headline TH@50 and Decathlon Scores.  
* Use the diagnostic data from the Performance Vector and Competency Matrix to understand the specific strengths, weaknesses, and behavioral profiles of each architecture.  
* Synthesize these findings into a final, data-driven recommendation for the foundational planning architecture of the Chimera-1 agent.

#### **Works cited**

1. Intrinsic and Extrinsic Motivation in Intelligent Systems \- Proceedings of Machine Learning Research, accessed July 6, 2025, [http://proceedings.mlr.press/v131/lieberman20a/lieberman20a.pdf](http://proceedings.mlr.press/v131/lieberman20a/lieberman20a.pdf)  
2. Intrinsic vs. Extrinsic Motivation: What's the Difference? \- Verywell Mind, accessed July 6, 2025, [https://www.verywellmind.com/differences-between-extrinsic-and-intrinsic-motivation-2795384](https://www.verywellmind.com/differences-between-extrinsic-and-intrinsic-motivation-2795384)  
3. What is AI Agent Planning? | IBM, accessed July 6, 2025, [https://www.ibm.com/think/topics/ai-agent-planning](https://www.ibm.com/think/topics/ai-agent-planning)  
4. Types of AI Agents | IBM, accessed July 6, 2025, [https://www.ibm.com/think/topics/ai-agent-types](https://www.ibm.com/think/topics/ai-agent-types)  
5. What is intrinsic motivation in reinforcement learning? \- Milvus, accessed July 6, 2025, [https://milvus.io/ai-quick-reference/what-is-intrinsic-motivation-in-reinforcement-learning](https://milvus.io/ai-quick-reference/what-is-intrinsic-motivation-in-reinforcement-learning)  
6. Reinforcement Learning with Intrinsic Motivation \- GeeksforGeeks, accessed July 6, 2025, [https://www.geeksforgeeks.org/deep-learning/reinforcement-learning-with-intrinsic-motivation/](https://www.geeksforgeeks.org/deep-learning/reinforcement-learning-with-intrinsic-motivation/)  
7. On what motivates us: a detailed review of intrinsic v. extrinsic motivation \- PMC, accessed July 6, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9340849/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9340849/)  
8. Measuring AI Ability to Complete Long Tasks \- METR, accessed July 6, 2025, [https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)  
9. Measuring AI Ability to Complete Long Tasks \- arXiv, accessed July 6, 2025, [https://arxiv.org/html/2503.14499v1](https://arxiv.org/html/2503.14499v1)  
10. METR: Measuring AI Ability to Complete Long Tasks \- AI Alignment Forum, accessed July 6, 2025, [https://www.alignmentforum.org/posts/deesrjitvXM4xYGZd/metr-measuring-ai-ability-to-complete-long-tasks](https://www.alignmentforum.org/posts/deesrjitvXM4xYGZd/metr-measuring-ai-ability-to-complete-long-tasks)  
11. METR: Measuring AI Ability to Complete Long Tasks \- LessWrong, accessed July 6, 2025, [https://www.lesswrong.com/posts/deesrjitvXM4xYGZd/metr-measuring-ai-ability-to-complete-long-tasks](https://www.lesswrong.com/posts/deesrjitvXM4xYGZd/metr-measuring-ai-ability-to-complete-long-tasks)  
12. Plan and Act: Enabling Agents to Solve Long Horizon Tasks \- YouTube, accessed July 6, 2025, [https://www.youtube.com/watch?v=\_GdoyYufuw8](https://www.youtube.com/watch?v=_GdoyYufuw8)  
13. Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments \- NIPS, accessed July 6, 2025, [https://papers.nips.cc/paper\_files/paper/2024/file/7d6e85e88495104442af94c98e899659-Paper-Conference.pdf](https://papers.nips.cc/paper_files/paper/2024/file/7d6e85e88495104442af94c98e899659-Paper-Conference.pdf)  
14. AI Agents: Evolution, Architecture, and Real-World Applications \- arXiv, accessed July 6, 2025, [https://arxiv.org/html/2503.12687v1](https://arxiv.org/html/2503.12687v1)  
15. Accelerating scientific discovery with AI | MIT News | Massachusetts Institute of Technology, accessed July 6, 2025, [https://news.mit.edu/2025/futurehouse-accelerates-scientific-discovery-with-ai-0630](https://news.mit.edu/2025/futurehouse-accelerates-scientific-discovery-with-ai-0630)  
16. GAIA Leaderboard \- a Hugging Face Space by gaia-benchmark, accessed July 6, 2025, [https://huggingface.co/spaces/gaia-benchmark/leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)  
17. (PDF) Agentbench: Evaluating LLMs as Agents | Chat PDF \- Nanonets, accessed July 6, 2025, [https://nanonets.com/chat-pdf/agentbench-evaluating-llms-as-agents](https://nanonets.com/chat-pdf/agentbench-evaluating-llms-as-agents)  
18. GAIA: a benchmark for general AI assistants | Research \- AI at Meta, accessed July 6, 2025, [https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants/](https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants/)  
19. GAIA: A Benchmark for General AI Assistants arXiv:2311.12983v1 ..., accessed July 6, 2025, [https://arxiv.org/abs/2311.12983](https://arxiv.org/abs/2311.12983)  
20. What Are AI Agents? | IBM, accessed July 6, 2025, [https://www.ibm.com/think/topics/ai-agents](https://www.ibm.com/think/topics/ai-agents)  
21. The Battle of AI Agents: Comparing Real World Performance Using Benchmarking, accessed July 6, 2025, [https://cobusgreyling.medium.com/the-battle-of-ai-agents-comparing-real-world-performance-using-benchmarking-356a8c6e0fcc](https://cobusgreyling.medium.com/the-battle-of-ai-agents-comparing-real-world-performance-using-benchmarking-356a8c6e0fcc)  
22. ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents, accessed July 6, 2025, [https://arxiv.org/html/2410.06703v4](https://arxiv.org/html/2410.06703v4)  
23. Rethinking AI Evaluation: Introducing the GAIA Benchmark | by Edgar Bermudez \- Medium, accessed July 6, 2025, [https://medium.com/about-ai/rethinking-ai-evaluation-introducing-the-gaia-benchmark-cae6f3c1e0e2](https://medium.com/about-ai/rethinking-ai-evaluation-introducing-the-gaia-benchmark-cae6f3c1e0e2)  
24. WABER: EVALUATING RELIABILITY AND EFFI- CIENCY OF WEB AGENTS WITH EXISTING BENCH- MARKS \- Microsoft, accessed July 6, 2025, [https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/134\_WABER\_Web\_Agent\_Benchmarki.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/134_WABER_Web_Agent_Benchmarki.pdf)  
25. HAL: GAIA Leaderboard, accessed July 6, 2025, [https://hal.cs.princeton.edu/gaia](https://hal.cs.princeton.edu/gaia)  
26. Multi-Agent Collaboration | IBM, accessed July 6, 2025, [https://www.ibm.com/think/topics/multi-agent-collaboration](https://www.ibm.com/think/topics/multi-agent-collaboration)  
27. Building Multi-Agent Workflows: A Comprehensive Guide, accessed July 6, 2025, [https://bestaiagents.ai/blog/building-multi-agent-workflows-a-comprehensive-guide](https://bestaiagents.ai/blog/building-multi-agent-workflows-a-comprehensive-guide)  
28. How we built our multi-agent research system \- Anthropic, accessed July 6, 2025, [https://www.anthropic.com/engineering/built-multi-agent-research-system](https://www.anthropic.com/engineering/built-multi-agent-research-system)  
29. Collaborative Intelligence: Advanced AI Problem Solving with Multi-Agent Systems, accessed July 6, 2025, [https://www.amplework.com/blog/collaborative-intelligence-multi-agent-ai-systems/](https://www.amplework.com/blog/collaborative-intelligence-multi-agent-ai-systems/)  
30. AgentBench: Evaluating LLMs as Agents \- arXiv, accessed July 6, 2025, [http://arxiv.org/pdf/2308.03688](http://arxiv.org/pdf/2308.03688)  
31. THUDM/AgentBench: A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24), accessed July 6, 2025, [https://github.com/THUDM/AgentBench](https://github.com/THUDM/AgentBench)  
32. Establishing Best Practices for Building Rigorous Agentic Benchmarks \- arXiv, accessed July 6, 2025, [https://arxiv.org/html/2507.02825v1](https://arxiv.org/html/2507.02825v1)  
33. Test Environment Management: A Best Practices Guide \- Coherence, accessed July 6, 2025, [https://www.withcoherence.com/post/test-environment-management](https://www.withcoherence.com/post/test-environment-management)  
34. ALFWORLD: ALIGNING TEXT AND EMBODIED ENVIRONMENTS FOR INTERACTIVE LEARNING \- OpenReview, accessed July 6, 2025, [https://openreview.net/pdf?id=0IOX0YcCdTn](https://openreview.net/pdf?id=0IOX0YcCdTn)  
35. ALFWorld, accessed July 6, 2025, [https://alfworld.github.io/](https://alfworld.github.io/)  
36. open-operator/benchmarks/webarena.md at main \- GitHub, accessed July 6, 2025, [https://github.com/All-Hands-AI/open-operator/blob/main/benchmarks/webarena.md](https://github.com/All-Hands-AI/open-operator/blob/main/benchmarks/webarena.md)  
37. LegalAgentBench: Evaluating LLM Agents in Legal Domain \- OpenReview, accessed July 6, 2025, [https://openreview.net/pdf?id=5tazzIILzt](https://openreview.net/pdf?id=5tazzIILzt)  
38. AgentQuest: Benchmarking LLM Agents Behaviours in Multi-step Intensive Reasoning Tasks \- arXiv, accessed July 6, 2025, [https://arxiv.org/html/2404.06411v1](https://arxiv.org/html/2404.06411v1)  
39. Agent-as-a-Judge: Evaluating Agents with Agents \- OpenReview, accessed July 6, 2025, [https://openreview.net/forum?id=DeVm3YUnpj](https://openreview.net/forum?id=DeVm3YUnpj)  
40. Mind2Web 2, accessed July 6, 2025, [https://osu-nlp-group.github.io/Mind2Web-2/](https://osu-nlp-group.github.io/Mind2Web-2/)  
41. State of AI Agents in 2025: A Technical Analysis | by Carl Rannaberg | Medium, accessed July 6, 2025, [https://carlrannaberg.medium.com/state-of-ai-agents-in-2025-5f11444a5c78](https://carlrannaberg.medium.com/state-of-ai-agents-in-2025-5f11444a5c78)  
42. Evaluating Agent Tool Selection — Testing if First Really is the Worst | by ODSC, accessed July 6, 2025, [https://odsc.medium.com/evaluating-agent-tool-selection-testing-if-first-really-is-the-worst-b83dad43f641](https://odsc.medium.com/evaluating-agent-tool-selection-testing-if-first-really-is-the-worst-b83dad43f641)  
43. Establishing Best Practices for Building Rigorous Agentic Benchmarks \- YouTube, accessed July 6, 2025, [https://www.youtube.com/watch?v=6snCzcGVlRc](https://www.youtube.com/watch?v=6snCzcGVlRc)