

# **A Blueprint for the Agentic Pre-training Corpus**

## **Section 1: Foundational Principles of the Agentic Corpus**

### **1.1 Introduction: Defining the Programmatic Agent and its Core Curriculum**

The development of a world-class programmatic agent, herein referred to as Chimera-1, necessitates a paradigm shift in training data philosophy. It requires moving beyond simple command-response datasets to a meticulously engineered *curriculum* designed to teach complex reasoning and tool use. The fundamental unit of this curriculum is the **agentic action trajectory**. This data structure represents the complete cognitive and operational sequence of an agent performing a task. It encompasses the high-level goal, the decomposition of that goal into a coherent plan, the selection and execution of appropriate tools, the observation of intermediate results, and the capacity for iterative refinement based on those observations.

The ultimate objective is to cultivate an agent that can reason, act, and interact within complex digital environments, mirroring the capabilities of advanced agentic Large Language Models (LLMs).1 The quality, diversity, and complexity of the training curriculum are the primary determinants of the agent's eventual proficiency in real-world, unconstrained scenarios. A massive but low-quality dataset will produce an agent capable of mimicry; a curated, complex curriculum will produce an agent capable of genuine problem-solving.

### **1.2 Design Philosophy: The Three Pillars of the Chimera-1 Curriculum**

The construction of this agentic pre-training corpus will be guided by three foundational principles:

1. **Real-World Grounding:** The curriculum must be fundamentally rooted in authentic, human-authored procedures. By learning from artifacts created by human experts, the agent will internalize the common patterns, idioms, practical constraints, and implicit knowledge that govern real-world tool use. This grounding ensures the agent's solutions are not just theoretically sound but also practically viable.  
2. **Complex Reasoning and Planning:** The curriculum must explicitly teach multi-step, non-linear planning and sophisticated tool composition. Trajectories will be selected and generated to push the agent beyond simple sequential execution, forcing it to learn conditional logic, error handling, and dynamic plan adjustment. This focus directly addresses the known limitations of standard autoregressive models in performing hierarchical planning, which is essential for advanced agency.2  
3. **Verifiable Correctness and Quality:** Every trajectory admitted into the final curriculum must pass through a rigorous, multi-stage validation pipeline. This process will confirm not only that the trajectory is functionally correct (i.e., the code executes without error) but also that it is logically sound, efficient, and safe, ensuring the agent learns high-quality, reliable behaviors.

### **1.3 The Dataset as a "Curricular Scaffold"**

A key strategic decision in the design of this corpus is to construct it not as a monolithic, uniformly sampled dataset, but as a "curricular scaffold." This approach involves explicitly stratifying the dataset into tiers of increasing difficulty and complexity. The rationale for this structure is drawn from established principles in machine learning, particularly curriculum learning, where training on a sequence of progressively harder tasks has been shown to expedite learning and improve generalization in agents.3

The user's request for a "curriculum" implies a structured learning path, not merely a data repository. The proposed data sourcing and quality filtration pipeline will naturally generate the metadata necessary to implement this structure. For example, a trajectory's source (e.g., a highly structured Ansible playbook versus a complex, synthetically generated scenario), its plan complexity (e.g., number of steps, conditional branches), and its quality validation scores can all be used as features for stratification.

This enables the creation of a tiered corpus:

* **Level 1 (Foundational Skills):** Comprising high-confidence, single-goal trajectories mined from well-structured sources like Ansible playbooks. These teach the agent the basic syntax and semantics of common tools.  
* **Level 2 (Intermediate Tool Use):** Featuring multi-step workflows extracted from sources like GitHub Actions, teaching the agent to chain commands and manage simple dependencies.  
* **Level 3 (Advanced Planning & Reasoning):** Consisting of complex, synthetically generated trajectories that include sophisticated conditional logic, dynamic error handling, and abstract planning.

This scaffolded structure provides two significant advantages. First, it allows for a more sophisticated, staged fine-tuning strategy for Chimera-1, potentially accelerating the learning process. Second, it provides a structured benchmark for evaluating the agent's capabilities at distinct levels of complexity, offering a granular view of its developing skills.

## **Section 2: A Dual-Track Strategy for Knowledge Sourcing**

To build a corpus that is both grounded in reality and sufficiently complex, a dual-track sourcing strategy is essential. The first track focuses on mining existing digital artifacts to capture human expertise, while the second uses synthetic generation to scale complexity and diversity beyond what can be found in existing data.

### **2.1 Track 1: Mining Digital Artifacts for Latent Trajectories**

This track reverse-engineers human-authored workflows from open-source repositories to extract structured action trajectories. This process grounds the agent's knowledge in battle-tested, real-world procedures.

#### **2.1.1 High-Structure Sources: Ansible Playbooks and GitHub Actions Workflows**

**Rationale:** Automation artifacts like Ansible playbooks and GitHub Actions workflows are ideal starting points. They are authored in a structured format (YAML) and explicitly codify human intent for automation, defining goals, task sequences, parameters, and conditional logic.4

**Methodology (Ansible):** Ansible playbooks can be parsed using standard libraries like PyYAML to convert them into Python dictionaries.6 The playbook's structure directly maps to an agentic trajectory. A

play defines the high-level goal, tasks represent individual tool calls, and vars provide the parameters. The execution order is well-defined (pre\_tasks, roles, tasks, post\_tasks, handlers), providing a clear sequence of operations.7 Ansible Roles are particularly valuable as they represent modular, reusable sub-trajectories that can be extracted as self-contained functions.12 Furthermore, the

changed status in a task's output is a direct signal of whether an action had a tangible effect, providing a crucial piece of observation data for the trajectory.15

**Methodology (GitHub Actions):** Workflow files (.github/workflows/\*.yml) provide a rich source of CI/CD and DevOps trajectories.16 The

name and run-name fields serve as the high-level goal.4 The

jobs block contains a sequence of steps. Each step with a run key is a direct shell command execution, while a uses key (e.g., actions/checkout@v4) signifies a call to a pre-packaged tool.18 Analyzing the most frequently used actions across a large corpus of repositories can reveal common, high-value automation patterns.21

**Data Yield:** This approach will yield high-fidelity, structured trajectories primarily in the domains of DevOps, CI/CD, and infrastructure management.

#### **2.1.2 Semi-Structured Sources: Jupyter Notebooks for Analytical Workflows**

**Rationale:** Jupyter Notebooks (.ipynb files) are a treasure trove of analytical workflows, combining code, explanatory text, and execution outputs in a single document.25 They offer a richer view of a user's intent compared to pure scripts.

**Methodology:** An .ipynb file is a JSON document that can be programmatically parsed using libraries like nbformat or command-line tools like jq.26 The key is to correlate

markdown cells with subsequent code cells; the markdown often serves as a natural language description of the goal for the code that follows.27 The existence of tools like Jupybara and Jupyter Agent, which can generate entire notebooks from high-level prompts, demonstrates the strong semantic link between the natural language and code components of a notebook, suggesting that the reverse process—extracting structured goals from existing notebooks—is highly feasible.25 We can also scan for conventions like special comment tags (e.g.,

\#ayx\_input) that tools use to define notebook I/O, as these are strong signals for parsing.31

**Data Yield:** Trajectories for data analysis, data cleaning, model training, visualization, and scientific computing.32

#### **2.1.3 Low-Structure Sources: Parsing Shell Scripts via Abstract Syntax Trees**

**Rationale:** Shell scripts are ubiquitous but lack the explicit structure of YAML files, making intent extraction challenging. Simple parsing with regular expressions is brittle and prone to failure.

**Methodology:** A robust approach requires converting the script into an Abstract Syntax Tree (AST). For Bash scripts, the Python library bashlex is the ideal tool, as it is a direct port of the internal parser used by GNU Bash itself, ensuring high-fidelity parsing.33 The AST provides a structured, hierarchical representation of the script's commands, pipelines, redirections, and control flow (loops, conditionals). By traversing this tree, we can reliably extract the sequence of tool calls and their arguments. As a pre-processing step, running a static analysis tool like

ShellCheck can identify syntax errors and bad practices, improving the quality of scripts before they enter the parsing pipeline.34 The high-level goal of the script must often be inferred, either from comments within the script, associated documentation (like a README file), or by using an LLM to generate a summary of the script's function based on its parsed AST.

**Data Yield:** A vast and highly diverse set of trajectories covering a wide range of tasks, but with higher uncertainty in the inferred goal and plan. This data will require the most rigorous downstream validation.

### **2.2 Track 2: Synthetic Generation of Complex Trajectories**

This track leverages a powerful "teacher" LLM (e.g., GPT-4o, Claude 3.5 Sonnet) to generate novel, complex trajectories. This overcomes the limitations of mined data, which may be repetitive, overly simplistic, or lack coverage of important edge cases.

#### **2.2.1 The "Evol-Instruct" Generation Pipeline**

**Rationale:** Simply prompting an LLM to "write a script to do X" often yields low-quality or generic results. A more sophisticated method is needed to generate complex and diverse data. The "Evol-Instruct" methodology, originally developed for creating instruction-following data, can be adapted for generating agentic trajectories.36

**Methodology:** This is an iterative process:

1. **Seed Generation:** Begin with a simple, high-level goal (e.g., "Analyze a CSV file and plot a histogram"). These seeds can be sourced from the goals extracted during the mining track or from a human-curated list.  
2. **Initial Trajectory Generation:** The teacher model generates a basic trajectory for the seed goal.  
3. **Evolution Prompting:** A randomly selected "evolution prompt" is applied to the initial trajectory to increase its complexity. Examples of evolution prompts include:  
   * **Add Constraints:** "...but do it for a 10GB file without loading it all into memory."  
   * **Increase Reasoning Steps:** "...but first, identify and impute missing values using the column median."  
   * **Incorporate Error Handling:** "...and if the file is not found, log an error and exit gracefully."  
   * **Generalize:** "Rewrite this trajectory to be a reusable function that accepts a file path and column name as arguments."  
4. **Response Generation:** The teacher model generates a new, more complex trajectory that satisfies the evolved prompt.  
5. **Filtering and Iteration:** The evolved trajectory is sent to the quality filtration pipeline (Section 4). If it passes, it is added to the corpus. This process can be repeated multiple times to create progressively more sophisticated trajectories.36 This evolutionary approach is critical for ensuring dataset diversity.36

#### **2.2.2 Ensuring Coherent Tool Use: Graph-Based Sampling and Planning**

A primary failure mode in synthetic generation is the creation of illogical or irrelevant tool sequences. Research has shown that asking an LLM to create a task from a random set of tools often results in simplistic or incoherent outputs.39 To overcome this, we will adopt the "sampling-planning-generation" process inspired by the TOOLFLOW framework.39

This structured approach separates strategic planning from tactical code generation:

1. **Tool Graph Construction:** We will first construct a knowledge graph of available tools (e.g., common CLI commands, Python libraries, APIs). Nodes represent tools, and edges represent semantic relationships. Two tools are related if they share parameter types (e.g., get\_weather and book\_flight both use a location parameter) or if one tool's output can serve as another's input.  
2. **Subgraph Sampling:** To generate a new trajectory, we will sample a connected subgraph from this tool graph, rather than a random set of tools. This ensures the selected tools are inherently related and can be logically combined.  
3. **Hierarchical Plan Generation:** The teacher model is first prompted to generate a high-level, multi-step plan using the sampled tool subset. This crucial step forces the model to focus on the overall logic and strategy *before* writing any code. This mitigates the tendency of autoregressive models to get lost in local details without a global plan.2  
4. **Trajectory Generation:** With a coherent plan in place, the model then "executes" the plan step-by-step, generating the detailed thought process and specific tool calls required for each stage.

This method is critical for generating the complex, multi-step, and logically sound trajectories that are essential for teaching advanced agentic behavior but are rarely found in mined data.

### **2.3 Hybrid Sourcing Recommendation and Synergy**

The mining and synthetic generation tracks are not independent silos; they must be integrated into a symbiotic feedback loop to produce the highest quality corpus. Mined data provides the real-world grounding that can prevent synthetic generation from producing unrealistic or irrelevant trajectories. Conversely, synthetic generation provides the complexity and diversity that mined data often lacks.

The recommended synergistic workflow is as follows:

* **Grounding the Generator:** The universe of available tools, APIs, and command patterns used in the synthetic track's tool graph will be populated directly from the artifacts discovered in the mining track.  
* **Learning from Real Errors:** Common failure modes and error messages identified from the execution logs of mined trajectories will be used to create specific "error handling" evolution prompts for the synthetic generator.  
* **Seeding with Real Goals:** The high-level goals inferred from mined artifacts (e.g., from Ansible play names or Jupyter notebook markdown) will serve as the initial seeds for the "Evol-Instruct" pipeline.

A balanced corpus is recommended, with a proposed mix of **60% mined data** (for robustness and coverage of common patterns) and **40% synthetic data** (for complexity, diversity, and coverage of edge cases).

**Table 1: Comparative Analysis of Data Sourcing Strategies**

| Sourcing Method | Data Source Examples | Key Strengths | Key Weaknesses | Extraction/Generation Cost | Primary Contribution to Curriculum |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Mined (High-Structure)** | Ansible Playbooks, GitHub Actions | High-fidelity, structured, clear intent | Limited domain (DevOps, CI/CD), can be repetitive | Low | Foundational skills, common patterns |
| **Mined (Semi-Structure)** | Jupyter Notebooks | Rich context (code \+ text), analytical workflows | Intent requires inference, noisy outputs | Medium | Data science & analysis trajectories |
| **Mined (Low-Structure)** | Shell Scripts (Bash, PowerShell) | Massive volume, high diversity of tasks | Unstructured, intent is ambiguous, requires AST parsing | High | Broad coverage of diverse tool use |
| **Synthetic (Evol-Instruct)** | LLM-generated from seeds | High complexity, controllable diversity | Can lack real-world grounding, risk of hallucination | High (API costs) | Advanced reasoning, error handling |
| **Synthetic (Graph-Planned)** | LLM-generated from tool subgraphs | Logically coherent multi-tool use, complex plans | Dependent on quality of tool graph, computationally intensive | Very High | Sophisticated planning, tool composition |

## **Section 3: The "Action Transcript" Schema: A Universal Language for Agentic Behavior**

To unify the heterogeneous data from both sourcing tracks, a standardized schema is required. The **Action Transcript** is proposed as this universal format, designed to be rich enough to capture not just *what* the agent did, but also *why* it did it and *what happened* as a result.

### **3.1 Core Components of an Action Trajectory**

An agentic trajectory is deconstructed into five fundamental components:

1. **High-Level Goal:** The ultimate objective the agent is trying to achieve, specified in natural language.  
2. **Thought Process (Chain-of-Thought):** The agent's internal monologue or reasoning process. This is a critical component that makes the agent's decision-making process transparent. It includes the initial plan, the decomposition of the goal into sub-tasks, reflections on previous steps, and self-corrections. This is inspired by methodologies like ReAct, which interleave reasoning and action.40  
3. **Tool Calls:** The sequence of actual commands, API calls, or functions executed by the agent. This must capture the precise tool name, all parameters, and any relevant environmental context.  
4. **Observations:** The raw, unadulterated output returned from each tool call. This includes stdout, stderr, API responses in JSON or other formats, and system exit codes. This is the agent's "sensory input" from the environment.  
5. **Outcome:** A summary of the final state after the trajectory is complete. This includes a status (e.g., success, failure), a natural language summary of what was accomplished, and a list of any final artifacts produced.

### **3.2 Proposed JSON Schema Definition: ActionTranscript-v1.0**

The following JSON schema provides a formal structure for the Action Transcript. It is designed to be both human-readable for analysis and machine-parsable for training and evaluation.

**Table 2: Detailed "Action Transcript" JSON Schema Specification**

| Field Name | Data Type | Description | Required | Example |
| :---- | :---- | :---- | :---- | :---- |
| trajectory\_id | String | A unique UUID for the trajectory. | Y | "traj\_a1b2c3d4" |
| metadata | Object | Metadata about the trajectory's origin and characteristics. | Y | {"source": "mined",...} |
| metadata.source | Enum | The origin of the trajectory. Values: mined, synthetic, human-authored. | Y | "mined" |
| metadata.source\_details | String | URL to the source GitHub repo, or name of the teacher model. | Y | "https://github.com/user/repo/playbook.yml" |
| metadata.domain | String | The primary domain of the task. | N | "devops" |
| metadata.tags | Array of Strings | Keywords describing the tools or concepts involved. | N | \["docker", "nginx", "ssl"\] |
| metadata.complexity\_score | Float | A calculated score representing the trajectory's complexity. | N | 3.75 |
| goal | Object | The high-level objective. | Y | {"natural\_language\_description":...} |
| goal.natural\_language\_description | String | The user-facing goal in plain English. | Y | "Set up a secure Nginx web server." |
| trajectory | Array of Objects | The sequence of steps taken by the agent. | Y | \[{"step\_id": 1,...}\] |
| trajectory.step.step\_id | Integer | The sequential identifier for the step. | Y | 1 |
| trajectory.step.thought | String | The agent's reasoning for this specific step. | Y | "First, I need to install Nginx..." |
| trajectory.step.action | Object | The tool call to be executed. | Y | {"tool\_name": "ansible.builtin.apt",...} |
| trajectory.step.action.tool\_name | String | The name of the tool, command, or module. | Y | "ansible.builtin.apt" |
| trajectory.step.action.tool\_code | String | The raw code or command executed. | N | "apt-get install nginx \-y" |
| trajectory.step.action.parameters | Object | Structured parameters for the tool call. | N | {"name": "nginx", "state": "present"} |
| trajectory.step.observation | Object | The results returned from the tool execution. | Y | {"exit\_code": 0, "stdout": "...",...} |
| trajectory.step.observation.exit\_code | Integer | The exit code of the command (0 for success). | Y | 0 |
| trajectory.step.observation.stdout | String | The standard output from the command. | Y | "Nginx installed successfully." |
| trajectory.step.observation.stderr | String | The standard error output from the command. | Y | "" |
| trajectory.step.observation.artifacts\_generated | Array of Strings | Paths to any files created during this step. | N | \`\` |
| final\_outcome | Object | The final result of the entire trajectory. | Y | {"status": "success",...} |
| final\_outcome.status | Enum | Final status. Values: success, failure, error. | Y | "success" |
| final\_outcome.summary | String | A natural language summary of the final outcome. | Y | "The Nginx server was installed and configured." |
| final\_outcome.final\_artifacts | Array of Strings | A list of all artifacts produced by the trajectory. | N | \["/etc/nginx/sites-available/default"\] |
| quality\_scores | Object | Scores from the validation pipeline. | Y | {"programmatic\_validation":...} |

### **3.3 Annotated Examples of the Schema in Practice**

Example 1: Mined Trajectory from an Ansible Playbook  
This example shows how a simple Ansible task to install Nginx would be mapped to the ActionTranscript schema.

JSON

{  
  "trajectory\_id": "traj\_mined\_nginx\_install\_001",  
  "metadata": {  
    "source": "mined",  
    "source\_details": "https://example.com/ansible-repo/playbooks/webserver.yml",  
    "domain": "devops",  
    "tags": \["ansible", "nginx", "apt"\]  
  },  
  "goal": {  
    "natural\_language\_description": "Install the Nginx web server package."  
  },  
  "trajectory":  
      }  
    }  
  \],  
  "final\_outcome": {  
    "status": "success",  
    "summary": "The Nginx package was successfully installed on the target host.",  
    "final\_artifacts":  
  },  
  "quality\_scores": {... }  
}

Example 2: Synthetic Trajectory for Troubleshooting  
This example illustrates a more complex, synthetically generated trajectory for a data analysis task, highlighting the rich thought process.

JSON

{  
  "trajectory\_id": "traj\_synth\_dataclean\_002",  
  "metadata": {  
    "source": "synthetic",  
    "source\_details": "TeacherModel: GPT-4o, Evol-Prompt: 'Incorporate error handling'",  
    "domain": "data\_analysis",  
    "tags": \["python", "pandas", "data\_cleaning"\]  
  },  
  "goal": {  
    "natural\_language\_description": "Load 'sales\_data.csv', calculate the average 'price', and handle potential missing values or file errors."  
  },  
  "trajectory":.isnull().any():\\n  mean\_price \= df\['price'\].mean()\\n  df\['price'\].fillna(mean\_price, inplace=True)\\n  print(f\\"Filled {df\['price'\].isnull().sum()} missing values with mean {mean\_price}\\")\\nelse:\\n  print('No missing values in price column.')"  
      },  
      "observation": {  
        "exit\_code": 0,  
        "stdout": "Filled 12 missing values with mean 15.75",  
        "stderr": ""  
      }  
    },  
    {  
      "step\_id": 3,  
      "thought": "Finally, with the data cleaned, I can safely calculate and print the average price.",  
      "action": {  
        "tool\_name": "python",  
        "tool\_code": "average\_price \= df\['price'\].mean()\\nprint(f'Average price: {average\_price}')"  
      },  
      "observation": {  
        "exit\_code": 0,  
        "stdout": "Average price: 15.75",  
        "stderr": ""  
      }  
    }  
  \],  
  "final\_outcome": {  
    "status": "success",  
    "summary": "Successfully loaded data, imputed 12 missing price values, and calculated the average price to be 15.75.",  
    "final\_artifacts":  
  },  
  "quality\_scores": {... }  
}

## **Section 4: The Multi-Stage Quality Filtration and Validation Pipeline**

### **4.1 Rationale: The Necessity of a Multi-Faceted Approach**

Raw data from both mining and synthetic generation is inherently noisy and imperfect. A single validation method is insufficient to guarantee quality. Programmatic checks can verify executability but cannot assess intent or efficiency. LLM-based judges can evaluate logic but cannot guarantee functional correctness. Therefore, a multi-stage pipeline that combines the strengths of each approach is required to holistically assess trajectory quality.

### **4.2 Stage 1: Programmatic Validation in Sandboxed Environments**

**Goal:** To verify the functional correctness and safety of each trajectory. The primary questions are: Does the code execute without crashing? Does it produce the expected side effects? Does it do so without posing a security risk?

**Methodology:**

* **Sandboxing:** All code execution must occur within a secure, isolated sandbox.42 This will be implemented using containerization technology (e.g., Docker) to create ephemeral environments with restricted network access, a temporary filesystem, and limited system call privileges. This containment is critical to safely "detonate" and analyze potentially untrusted code without risking harm to the host infrastructure.44  
* **Execution and Instrumentation:** The tool\_code from each step of an ActionTranscript is executed sequentially within the sandbox. The exit\_code, stdout, and stderr are captured from the execution environment and used to populate the observation field in the transcript. This provides a detailed, factual record of the execution trace.  
* **State-Based Validation:** For trajectories where a clear success condition can be defined (e.g., "a file named report.csv should be created," "the web server should return a 200 status code"), the final state of the sandbox filesystem and network services is programmatically checked to verify the outcome.

**Output:** A trajectory is flagged as programmatic\_pass or programmatic\_fail. The captured observations are not just a validation signal; they are a critical input for the subsequent LLM-as-a-Judge stage, grounding its evaluation in factual execution data.

### **4.3 Stage 2: LLM-as-a-Judge for Semantic and Logical Evaluation**

**Goal:** To assess the quality of the trajectory's underlying plan and reasoning. This stage moves beyond functional correctness to evaluate semantic properties: Is the approach logical and efficient? Does it align with the high-level goal? Does it represent a good solution?

**Methodology:**

* **Judge Model:** A powerful, well-aligned LLM (e.g., GPT-4o, Claude 3.5 Sonnet) will serve as the "judge".47  
* **Grounded Evaluation:** The judge's evaluation must be grounded in the results of the programmatic validation. The prompt to the judge will include the full ActionTranscript, complete with the observation data captured in Stage 1\. This prevents the judge from hallucinating execution outcomes.  
* **Evaluation Methodologies:**  
  * **Pairwise Comparison:** For tasks where multiple solutions are possible, a pairwise comparison is often more reliable than absolute scoring.47 The system can generate two or more alternative trajectories for the same goal and prompt the judge: "Given the goal and execution results, which of these two trajectories is a better solution and why?" This forces a relative judgment that is less susceptible to scoring biases.  
  * **Reference-Guided Scoring:** For mined data where the original artifact (e.g., the Ansible playbook) can serve as a "gold standard," the judge can be asked to score the extracted trajectory's fidelity to the original source.  
* **Mitigating Challenges:**  
  * **Cost:** LLM-based evaluation can be costly, especially at scale.48 A tiered approach will be used, applying cheaper, faster models for initial screening and reserving the most powerful (and expensive) judge models for the most complex or high-value trajectories.  
  * **Bias:** LLM judges can exhibit biases, such as favoring longer, more verbose solutions.51 These will be mitigated through carefully designed prompts that de-emphasize length, randomizing the order of trajectories in pairwise comparisons, and periodically calibrating the judge against a human-validated "gold standard" set.47  
  * **Explainability:** The judge will be explicitly prompted to provide a detailed rationale for its scores. This is crucial not only for understanding the quality of a given trajectory but also for debugging and improving the evaluation process itself.47

**Table 3: LLM-as-a-Judge Evaluation Rubric for Agentic Trajectories**

| Evaluation Dimension | Description | Scoring Criteria (1-5 Scale) | Example of High-Scoring Rationale (Score: 5\) | Example of Low-Scoring Rationale (Score: 2\) |
| :---- | :---- | :---- | :---- | :---- |
| **Correctness** | Does the plan logically achieve the goal, given the execution observations? | 1: Illogical, fails to meet goal. 3: Partially meets goal. 5: Logically sound, fully achieves goal. | "The plan correctly identifies the dependencies, installs them in the right order, and verifies the service status, successfully achieving the goal." | "The plan attempts to start the service before installing its dependencies, which is logically incorrect and caused the observed failure." |
| **Efficiency** | Is the plan concise? Does it avoid redundant steps or inefficient commands? | 1: Highly inefficient. 3: Some redundant steps. 5: Optimal, no wasted steps. | "The use of a single apt-get install command for all packages is highly efficient and minimizes network calls." | "The plan installs packages one by one in a loop, which is significantly slower than installing them all in a single command." |
| **Robustness** | Does the plan include appropriate checks and error handling? | 1: Brittle, no error handling. 3: Basic checks. 5: Comprehensive error handling and validation. | "The trajectory correctly uses a try-except block to handle potential file I/O errors, making it robust to missing input files." | "The script assumes the input file always exists. The observed FileNotFoundError shows the plan is not robust." |
| **Readability** | Does the code adhere to common conventions, style, and best practices? | 1: Obfuscated, hard to read. 3: Readable but inconsistent. 5: Clean, well-commented, follows best practices. | "The code is well-formatted, uses descriptive variable names, and includes comments explaining the complex logic, adhering to PEP 8 standards." | "The code uses single-letter variable names and lacks comments, making it very difficult to understand the logic without deep analysis." |
| **Safety** | Does the plan avoid risky commands (e.g., rm \-rf) without proper safeguards? | 1: Contains dangerous, unchecked commands. 3: Some risk. 5: Follows principle of least privilege, includes safety checks. | "The plan correctly checks if the target directory is empty before attempting a recursive delete, preventing accidental data loss." | "The use of rm \-rf $TARGET\_DIR without validating the variable's value is extremely dangerous and could wipe the filesystem." |

### **4.4 Stage 3: Human-in-the-Loop Review for Gold-Standard Curation**

**Goal:** To create a small but exceptionally high-quality "gold" dataset for final model evaluation and to resolve ambiguities that automated systems cannot.

**Methodology:** A small percentage (e.g., 1-2%) of trajectories will be routed for manual review by human domain experts (e.g., senior DevOps engineers, principal data scientists). This subset will include:

* Trajectories flagged for significant disagreement between the programmatic and LLM-as-a-Judge stages.  
* The most complex and novel trajectories generated by the synthetic pipeline.  
* A random sample of all passed trajectories to spot-check for systemic quality issues.

These experts will review, correct, and annotate the trajectories, creating a "gold standard" set. This set serves three critical purposes:

1. **Final Evaluation Benchmark:** This curated set will be the holdout test set used to measure the final performance of the trained Chimera-1 agent.  
2. **Judge Calibration:** It will be used to fine-tune and calibrate the LLM-as-a-Judge models, ensuring their evaluations align more closely with human expert preferences.  
3. **High-Value Fine-Tuning:** It can be reserved for a final, high-quality fine-tuning pass to polish the agent's capabilities on the most important and nuanced tasks.

## **Section 5: The End-to-End Curation Workflow: From Raw Artifact to Chimera-1 Curriculum**

This section synthesizes the sourcing and quality components into a complete, end-to-end workflow for building the multi-million-sample "Agentic Pre-training Corpus."

### **5.1 Workflow Architecture Overview**

The curation pipeline is designed as a scalable, parallel processing system. It begins with two distinct acquisition funnels (Mining and Synthetic Generation) that feed into a unified normalization and validation stream. This stream processes each trajectory through the three-stage quality filter, after which the validated data is deduplicated, structured, and assembled into the final, tiered curriculum.

### **5.2 Detailed Stages of the Curation Pipeline**

1. **Acquisition:** Parallel crawlers and API clients gather raw digital artifacts (cloning GitHub repositories, downloading script archives) and store them in a raw data lake (e.g., Amazon S3).  
2. **Parsing & Normalization:** A fleet of specialized parsers, each tailored to a specific source type (Ansible, GitHub Actions, Shell Scripts, Jupyter Notebooks), runs on the raw data. The output of every parser is a preliminary ActionTranscript object that conforms to the schema defined in Section 3\. This step unifies the heterogeneous sources into a common format.  
3. **Synthetic Generation:** In parallel, the synthetic generation engine (as detailed in Section 2.2) operates. It is seeded with goals and grounded with tool knowledge derived from the mining track. Its output is also a stream of ActionTranscript objects.  
4. **Quality Pipeline Execution:** The unified stream of normalized trajectories from both tracks enters the quality pipeline.  
   * **Stage 1 (Programmatic Validation):** Trajectories are placed on a distributed message queue and consumed by a scalable pool of sandboxed execution workers (e.g., managed by Kubernetes). Execution results (stdout, stderr, exit\_code, artifacts) are written back to the ActionTranscript record. Failed trajectories are either discarded or flagged for analysis.  
   * **Stage 2 (LLM-as-a-Judge):** Programmatically validated trajectories are sent via API to the LLM-as-a-Judge service. The resulting scores and rationales are written back to the quality\_scores field of the transcript.  
   * **Stage 3 (Human-in-the-Loop):** A small, targeted subset of trajectories is routed to a human review interface for gold-standard curation.  
5. **Deduplication and Hashing:** After passing the quality pipeline, content-aware hashing (e.g., min-hashing) is applied to the core components of each trajectory (such as the sequence of tool\_code and thought texts). This allows for the identification and removal of near-duplicate trajectories. Deduplicating training data is a critical step to improve model performance and prevent memorization issues.52  
6. **Final Dataset Assembly:** The final set of validated, high-quality, and unique trajectories is organized into the "Curricular Scaffold." Using the metadata fields (complexity\_score, source, domain, etc.), the data is partitioned into difficulty tiers and stored in a queryable, efficient format (e.g., versioned Parquet files in a data warehouse).

**Table 4: Required Technologies and Tools for the Curation Workflow**

| Pipeline Stage | Required Technology/Tool | Rationale/Purpose |
| :---- | :---- | :---- |
| **Acquisition** | Git, Python requests library | To programmatically clone repositories and download files from the web. |
| **Parsing & Normalization** | PyYAML, bashlex, nbformat | To parse structured (YAML) and unstructured (Bash, Jupyter) source files into a common AST/dictionary format. |
| **Synthetic Generation** | GPT-4o / Claude 3.5 Sonnet API | To serve as the "teacher" model for generating complex, novel trajectories. |
| **Sandboxed Execution** | Docker, Kubernetes | To provide secure, isolated, and scalable environments for executing untrusted code from trajectories. |
| **LLM-as-a-Judge** | GPT-4o / Claude 3.5 Sonnet API | To serve as the "judge" model for evaluating the logical quality and efficiency of trajectories. |
| **Data Storage** | Amazon S3, Google BigQuery / Snowflake | For storing raw artifacts, intermediate data, and the final structured Parquet dataset in a scalable and queryable manner. |
| **Workflow Orchestration** | Airflow / Prefect / Dagster | To manage the complex dependencies and scheduling of the multi-stage data processing pipeline. |

### **5.3 Scaling Operations for a Multi-Million Sample Corpus**

Achieving a corpus of millions of high-quality samples requires a robust and scalable infrastructure. The most computationally intensive stages are synthetic generation (requiring numerous LLM API calls) and sandboxed execution (requiring significant CPU and memory). A cloud-based architecture is recommended, leveraging Kubernetes to manage a large cluster of containerized workers for these tasks. The entire workflow is designed to be highly parallelizable, as each raw artifact can be parsed and validated independently. Cost management strategies will be critical, including using tiered LLM judges (cheaper models for bulk processing, expensive models for final checks), leveraging cloud provider spot instances for stateless execution workers, and implementing intelligent caching to avoid re-processing identical artifacts.

## **Section 6: Strategic Recommendations and Future Directions**

### **6.1 Summary of Strategic Recommendations**

To construct the premier training corpus for the Chimera-1 programmatic agent, the following strategic actions are recommended:

1. **Adopt a Hybrid Data Sourcing Strategy:** Do not rely on a single source. A synergistic combination of mining real-world artifacts (for grounding and common patterns) and synthetic generation (for complexity and diversity) will produce the most robust and capable curriculum.  
2. **Standardize on the ActionTranscript-v1.0 Schema:** This rich, universal format is essential for unifying heterogeneous data sources and capturing the full context of agentic behavior, including the crucial "thought" process.  
3. **Implement the Three-Stage Quality Pipeline:** Mandate that every trajectory passes through programmatic validation, LLM-as-a-Judge evaluation, and a final human review loop. This ensures the curriculum is not only functionally correct but also logically sound and aligned with expert human judgment.  
4. **Structure the Final Dataset as a "Curricular Scaffold":** Organize the final corpus into explicit tiers of difficulty. This will enable a more effective, staged training regimen for Chimera-1 and provide a structured framework for evaluating its progress.

### **6.2 Future Directions and Open Research Questions**

The creation of this corpus is a foundational step. Future work should explore more dynamic and advanced methods for agent training.

* **Self-Curating Curricula:** Once an initial version of Chimera-1 is trained, it can be used to improve its own curriculum. By deploying the agent to solve new problems, its failures can be captured as new, highly valuable training data. The agent itself can help identify gaps in its knowledge, which can then guide the generation of new synthetic trajectories to fill those gaps, creating a powerful self-improving feedback loop.3  
* **Dynamic Trajectory Augmentation:** The current proposal focuses on creating a static, offline pre-training dataset. A future evolution would be to move towards online learning, where the agent's live interactions with a dynamic environment continuously generate new training data. This would allow the agent to adapt and learn from an ever-expanding stream of experience, reducing the need for massive, upfront data collection efforts.1  
* **Multi-Agent Trajectories:** This blueprint focuses on single-agent trajectories. A critical next step is to source and generate trajectories that involve the collaboration of multiple agents. This would require extending the ActionTranscript schema to support multiple actors and their interactions, opening the door to training agents that can solve problems collaboratively, a key area of ongoing research.25

#### **Works cited**

1. Agentic Large Language Models, a survey \- arXiv, accessed July 6, 2025, [https://arxiv.org/html/2503.23037v2](https://arxiv.org/html/2503.23037v2)  
2. LLM's Next Step: Agents, Database or Sequence modeling? | by Sulbha Jain \- Medium, accessed July 6, 2025, [https://medium.com/@sulbha.jindal/llms-next-step-agents-database-or-sequence-modeling-98e9597d1594](https://medium.com/@sulbha.jindal/llms-next-step-agents-database-or-sequence-modeling-98e9597d1594)  
3. (PDF) Evolutionarily-Curated Curriculum Learning for Deep Reinforcement Learning Agents, accessed July 6, 2025, [https://www.researchgate.net/publication/330439452\_Evolutionarily-Curated\_Curriculum\_Learning\_for\_Deep\_Reinforcement\_Learning\_Agents](https://www.researchgate.net/publication/330439452_Evolutionarily-Curated_Curriculum_Learning_for_Deep_Reinforcement_Learning_Agents)  
4. Workflow syntax for GitHub Actions, accessed July 6, 2025, [https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)  
5. Ansible Playbooks: Complete Guide with Examples \- Spacelift, accessed July 6, 2025, [https://spacelift.io/blog/ansible-playbooks](https://spacelift.io/blog/ansible-playbooks)  
6. How to parse & detect Ansible playbook tasks? \- Stack Overflow, accessed July 6, 2025, [https://stackoverflow.com/questions/46213890/how-to-parse-detect-ansible-playbook-tasks](https://stackoverflow.com/questions/46213890/how-to-parse-detect-ansible-playbook-tasks)  
7. Roles — Ansible Community Documentation, accessed July 6, 2025, [https://docs.ansible.com/ansible/latest/playbook\_guide/playbooks\_reuse\_roles.html](https://docs.ansible.com/ansible/latest/playbook_guide/playbooks_reuse_roles.html)  
8. Ansible playbooks — Ansible Community Documentation, accessed July 6, 2025, [https://docs.ansible.com/ansible/latest/playbook\_guide/playbooks\_intro.html](https://docs.ansible.com/ansible/latest/playbook_guide/playbooks_intro.html)  
9. Handlers: running operations on change — Ansible Community Documentation, accessed July 6, 2025, [https://docs.ansible.com/ansible/latest/playbook\_guide/playbooks\_handlers.html](https://docs.ansible.com/ansible/latest/playbook_guide/playbooks_handlers.html)  
10. Order of operations during Ansible playbook parsing explained\! \- DevOpsSchool.com, accessed July 6, 2025, [https://www.devopsschool.com/blog/order-of-operations-during-ansible-playbook-parsing-explained/](https://www.devopsschool.com/blog/order-of-operations-during-ansible-playbook-parsing-explained/)  
11. Ansible playbooks — Ansible Community Documentation, accessed July 6, 2025, [https://docs.ansible.com/ansible/latest/user\_guide/playbooks\_intro.html](https://docs.ansible.com/ansible/latest/user_guide/playbooks_intro.html)  
12. What is an Ansible Role—and how is it used? \- Red Hat, accessed July 6, 2025, [https://www.redhat.com/en/topics/automation/what-is-an-ansible-role](https://www.redhat.com/en/topics/automation/what-is-an-ansible-role)  
13. Ansible Roles: Basics, Creating & Using \- Spacelift, accessed July 6, 2025, [https://spacelift.io/blog/ansible-roles](https://spacelift.io/blog/ansible-roles)  
14. Roles in Ansible Playbook \- Medium, accessed July 6, 2025, [https://medium.com/cloudnloud/roles-in-ansible-playbook-ffbe4574641b](https://medium.com/cloudnloud/roles-in-ansible-playbook-ffbe4574641b)  
15. How to interpret 'changed=1' in Ansible playbook recap \- LabEx, accessed July 6, 2025, [https://labex.io/tutorials/ansible-how-to-interpret-changed-1-in-ansible-playbook-recap-415810](https://labex.io/tutorials/ansible-how-to-interpret-changed-1-in-ansible-playbook-recap-415810)  
16. How to Catch Exposed AWS Secrets in GitHub Actions Logs | by MongoDB \- Medium, accessed July 6, 2025, [https://medium.com/mongodb/how-to-catch-exposed-aws-secrets-in-github-actions-logs-58b18a820560](https://medium.com/mongodb/how-to-catch-exposed-aws-secrets-in-github-actions-logs-58b18a820560)  
17. Understanding GitHub Actions, accessed July 6, 2025, [https://docs.github.com/articles/getting-started-with-github-actions](https://docs.github.com/articles/getting-started-with-github-actions)  
18. Build a CI/CD workflow with Github Actions, accessed July 6, 2025, [https://github.com/readme/guides/sothebys-github-actions](https://github.com/readme/guides/sothebys-github-actions)  
19. CI/CD in Node.js with GitHub Actions \- LogRocket Blog, accessed July 6, 2025, [https://blog.logrocket.com/ci-cd-node-js-github-actions/](https://blog.logrocket.com/ci-cd-node-js-github-actions/)  
20. Understanding GitHub Actions to Automate Workflows (With Examples), accessed July 6, 2025, [https://www.softwaretestinghelp.com/github-actions/](https://www.softwaretestinghelp.com/github-actions/)  
21. CICD Patterns with GitHub Actions and Docker \- Hosting Data Apps \- Analythium Solutions, accessed July 6, 2025, [https://hosting.analythium.io/cicd-patterns-with-github-actions-and-docker/](https://hosting.analythium.io/cicd-patterns-with-github-actions-and-docker/)  
22. How to build a CI/CD pipeline with GitHub Actions in four simple steps, accessed July 6, 2025, [https://github.blog/enterprise-software/ci-cd/build-ci-cd-pipeline-github-actions-four-steps/](https://github.blog/enterprise-software/ci-cd/build-ci-cd-pipeline-github-actions-four-steps/)  
23. (PDF) Empirical Analysis of CI/CD Tools Usage in GitHub Actions Workflows \- ResearchGate, accessed July 6, 2025, [https://www.researchgate.net/publication/381455154\_Empirical\_Analysis\_of\_CICD\_Tools\_Usage\_in\_GitHub\_Actions\_Workflows](https://www.researchgate.net/publication/381455154_Empirical_Analysis_of_CICD_Tools_Usage_in_GitHub_Actions_Workflows)  
24. \[2409.02366\] The Hidden Costs of Automation: An Empirical Study on GitHub Actions Workflow Maintenance \- arXiv, accessed July 6, 2025, [https://arxiv.org/abs/2409.02366](https://arxiv.org/abs/2409.02366)  
25. Unlocking Actionable Insights with Jupybara: A Multi-Agent AI ..., accessed July 6, 2025, [https://www.tableau.com/blog/jupybara-multi-agent-ai-assistant-data-analysis-and-storytelling](https://www.tableau.com/blog/jupybara-multi-agent-ai-assistant-data-analysis-and-storytelling)  
26. What is an ipynb File? \- Roboflow Blog, accessed July 6, 2025, [https://blog.roboflow.com/what-is-an-ipynb-file/](https://blog.roboflow.com/what-is-an-ipynb-file/)  
27. Get only the code out of Jupyter Notebook \- Stack Overflow, accessed July 6, 2025, [https://stackoverflow.com/questions/54350254/get-only-the-code-out-of-jupyter-notebook](https://stackoverflow.com/questions/54350254/get-only-the-code-out-of-jupyter-notebook)  
28. extract python code from jupyter notebook \- YouTube, accessed July 6, 2025, [https://www.youtube.com/watch?v=QImOVwUHDD0](https://www.youtube.com/watch?v=QImOVwUHDD0)  
29. Parsing the output of a Jupyter Notebook, accessed July 6, 2025, [https://discourse.jupyter.org/t/parsing-the-output-of-a-jupyter-notebook/12524](https://discourse.jupyter.org/t/parsing-the-output-of-a-jupyter-notebook/12524)  
30. Using Jupyter Agent for data exploration: a practical guide | AI ..., accessed July 6, 2025, [https://norahsakal.com/blog/jupyter-agent-data-exploration/](https://norahsakal.com/blog/jupyter-agent-data-exploration/)  
31. Jupyter Flow \- Alteryx Help Documentation, accessed July 6, 2025, [https://help.alteryx.com/current/en/designer/tools/laboratory/jupyter-flow.html](https://help.alteryx.com/current/en/designer/tools/laboratory/jupyter-flow.html)  
32. Custom training notebook tutorials | Vertex AI | Google Cloud, accessed July 6, 2025, [https://cloud.google.com/vertex-ai/docs/tutorials/custom-training-pipelines/custom-training-jupyter-notebooks](https://cloud.google.com/vertex-ai/docs/tutorials/custom-training-pipelines/custom-training-jupyter-notebooks)  
33. idank/bashlex: Python parser for bash \- GitHub, accessed July 6, 2025, [https://github.com/idank/bashlex](https://github.com/idank/bashlex)  
34. ShellCheck – shell script analysis tool, accessed July 6, 2025, [https://www.shellcheck.net/](https://www.shellcheck.net/)  
35. ShellCheck: A static analysis tool for shell scripts \- Hacker News, accessed July 6, 2025, [https://news.ycombinator.com/item?id=26504661](https://news.ycombinator.com/item?id=26504661)  
36. Using LLMs for Synthetic Data Generation: The Definitive Guide \- Confident AI, accessed July 6, 2025, [https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms)  
37. How to create LLM test datasets with synthetic data \- Evidently AI, accessed July 6, 2025, [https://www.evidentlyai.com/llm-guide/llm-test-dataset-synthetic-data](https://www.evidentlyai.com/llm-guide/llm-test-dataset-synthetic-data)  
38. Measuring Diversity in Synthetic Datasets \- arXiv, accessed July 6, 2025, [https://arxiv.org/html/2502.08512v1](https://arxiv.org/html/2502.08512v1)  
39. TOOLFLOW: Boosting LLM Tool-Calling Through ... \- ACL Anthology, accessed July 6, 2025, [https://aclanthology.org/2025.naacl-long.214.pdf](https://aclanthology.org/2025.naacl-long.214.pdf)  
40. arxiv.org, accessed July 6, 2025, [https://arxiv.org/html/2409.17166v1](https://arxiv.org/html/2409.17166v1)  
41. ScriptSmith: A Unified LLM Framework for Enhancing IT Operations via Automated Bash Script Generation, Assessment, and Refinement, accessed July 6, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/35147/37302](https://ojs.aaai.org/index.php/AAAI/article/view/35147/37302)  
42. What Is Sandboxing? \- Palo Alto Networks, accessed July 6, 2025, [https://www.paloaltonetworks.com/cyberpedia/sandboxing](https://www.paloaltonetworks.com/cyberpedia/sandboxing)  
43. What Is a Sandbox Environment? Meaning & Setup | Proofpoint US, accessed July 6, 2025, [https://www.proofpoint.com/us/threat-reference/sandbox](https://www.proofpoint.com/us/threat-reference/sandbox)  
44. Sandbox (computer security) \- Wikipedia, accessed July 6, 2025, [https://en.wikipedia.org/wiki/Sandbox\_(computer\_security)](https://en.wikipedia.org/wiki/Sandbox_\(computer_security\))  
45. The Ultimate Guide to Sandbox Environments: Safe & Efficient Software Testing, accessed July 6, 2025, [https://dev.to/testwithtorin/the-ultimate-guide-to-sandbox-environments-safe-efficient-software-testing-lb5](https://dev.to/testwithtorin/the-ultimate-guide-to-sandbox-environments-safe-efficient-software-testing-lb5)  
46. Sandboxing Security: A Practical Guide \- Perception Point, accessed July 6, 2025, [https://perception-point.io/guides/sandboxing/sandboxing-security-practical-guide/](https://perception-point.io/guides/sandboxing/sandboxing-security-practical-guide/)  
47. LLM as a Judge \- Humanloop, accessed July 6, 2025, [https://humanloop.com/blog/llm-as-a-judge](https://humanloop.com/blog/llm-as-a-judge)  
48. Tuning LLM Judge Design Decisions for 1/1000 of the Cost | OpenReview, accessed July 6, 2025, [https://openreview.net/forum?id=cve4NOiyVp](https://openreview.net/forum?id=cve4NOiyVp)  
49. LLM-as-a-judge on Amazon Bedrock Model Evaluation | Artificial Intelligence, accessed July 6, 2025, [https://aws.amazon.com/blogs/machine-learning/llm-as-a-judge-on-amazon-bedrock-model-evaluation/](https://aws.amazon.com/blogs/machine-learning/llm-as-a-judge-on-amazon-bedrock-model-evaluation/)  
50. LLM as a Judge \- Primer and Pre-Built Evaluators \- Arize AI, accessed July 6, 2025, [https://arize.com/llm-as-a-judge/](https://arize.com/llm-as-a-judge/)  
51. The Hidden Cost of LLM-as-a-Judge: When More Evaluation Means Less Value, accessed July 6, 2025, [https://www.soumendrak.com/blog/llm-evals/](https://www.soumendrak.com/blog/llm-evals/)  
52. Extracting training data from Large Language Models \- YouTube, accessed July 6, 2025, [https://www.youtube.com/watch?v=C3pMyeQJlDE](https://www.youtube.com/watch?v=C3pMyeQJlDE)  
53. Notebooks | AutoGen 0.2 \- Microsoft Open Source, accessed July 6, 2025, [https://microsoft.github.io/autogen/0.2/docs/notebooks/](https://microsoft.github.io/autogen/0.2/docs/notebooks/)  
54. Agentic Framework: How can we effectively use Large language model based Agents, accessed July 6, 2025, [https://www.researchgate.net/post/Agentic\_Framework\_How\_can\_we\_effectively\_use\_Large\_language\_model\_based\_Agents](https://www.researchgate.net/post/Agentic_Framework_How_can_we_effectively_use_Large_language_model_based_Agents)