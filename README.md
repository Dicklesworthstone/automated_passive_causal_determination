# Automated Passive Causal Determination

*(by Jeffrey Emanuel, 3/17/2025)*

## Overview

This project systematically tackles one of the most profound and practically important challenges in data science and statistical inference: **automatically determining causal relationships purely from observational (passive) data**.

The methodology employed here integrates advanced machine learning models (Transformer architectures implemented in JAX/Flax, XGBoost random forests), rigorous statistical validation, and a principled concept of **parsimony** (model simplicity) as a proxy for causal correctness.  

This repository provides a complete pipeline—from downloading publicly available benchmark datasets, intelligently exploring potential causal relationships, modeling these relationships at scale, to inferring a final Directed Acyclic Graph (DAG) explicitly representing causal relationships with confidence scores.

This project began as a free-form discussion (which you can read in its entirety [here](https://github.com/Dicklesworthstone/automated_passive_causal_determination/blob/main/gpt_45_discussion_and_project_creation.md)) with OpenAI's ChatGPT-4.5 about causal inference, which organically turned into a discussion about how inadequate current causal methods are and what would be a better approach. While I conceived of the approach and the major outline of how it should work, GPT-4.5 translated that into code (with some prodding and coaxing by me). Consider this a first draft and a concrete instantiation of the basic underlying idea more than a final, working project.

---

## Motivation and Core Insight

### The Fundamental Challenge of Causality

Traditional statistics clearly differentiates correlation from causation. Yet, in real-world settings, especially with purely observational data, causality remains notoriously difficult to infer. Classic causal inference techniques (such as randomized controlled trials) often are impractical or prohibitively expensive, leaving researchers relying heavily on assumptions that are challenging to verify.

A variety of formal systems—like Judea Pearl’s **do-calculus**—attempt to bring mathematical rigor to causal inference. However, practical limitations of do-calculus, particularly the requirement for pre-specified causal structures (DAGs), severely constrain its practical utility.

Our detailed exploration of these issues led us to question the practical applicability of existing formal causal inference methods. Specifically, we observed:

- Do-calculus provides theoretical completeness, but practically demands unrealistic certainty about the underlying causal DAG.
- Real-world causal inference typically operates under uncertainty. Determining causation without experimentation or active intervention often appears circular and limited.

### The Parsimony Principle

Reflecting on historical examples (such as Ptolemy's epicycles vs. Newtonian mechanics), we identified a crucial insight:

**True causal models inherently simplify prediction.**  
Correct causal relationships significantly reduce the complexity needed to accurately predict outcomes. Conversely, incorrect or incomplete causal assumptions inevitably force the use of increasingly complex models to fit data.  

Hence, we propose **parsimony**—the principle that the simplest model achieving robust predictive accuracy likely captures true causal relationships—as a practical heuristic for causal inference.

---

## Practical Implementation

To operationalize our theoretical insights, we developed an **automated causal discovery framework** integrating the following components:

### 1. Data Acquisition and Preparation (`causal_data_retrieval_and_preparation.py`)

This script asynchronously downloads and preprocesses publicly available benchmark datasets selected for their intuitive and clearly-understood causal structures. Specifically, we selected datasets such as:

- **US Traffic Accidents dataset:** Intuitive causal relationships (weather causing accidents).
- **Beijing Air Quality dataset:** Weather clearly causing variations in pollution levels.
- **Rossmann Retail Sales dataset:** Promotions and store conditions directly affecting sales.

These datasets are preprocessed using standard scaling, encoding categorical variables, removing problematic rows and columns, ensuring robust downstream analysis.

---

### 2. Automated Causal Discovery Pipeline (`automated_passive_causal_determination.py`)

The heart of this project, this script implements an innovative automated causal discovery pipeline built around two powerful machine learning paradigms:

#### Transformer Models (JAX/Flax):

- Real transformer implementation with positional encoding and multi-head self-attention.
- Model size and complexity are intelligently scaled using a Chinchilla-inspired heuristic, dynamically adjusting parameters based on dataset size.
- Transformers were chosen for their expressive power and flexibility, allowing us to robustly model complex, nonlinear relationships present in the observational data.

#### XGBoost Models:

- Included as an alternate baseline due to their proven predictive performance and ease of interpreting feature importance.
- Allows a transparent comparison and validation of results against transformer-based outcomes.

#### Key Methodological Steps:

- **Randomized Subset Sampling:**  
  For each target variable, the script intelligently samples random subsets of other variables as predictors, modeling every plausible causal hypothesis implicitly and systematically.

- **Parsimony-Performance Tradeoff:**  
  Models are evaluated using cross-validation (CV) and independent validation losses. Crucially, **model complexity** (number of parameters, model export size) is explicitly factored into a combined score that prioritizes simplicity alongside predictive accuracy.

- **DAG Edge Confidence Calculation:**  
  Top-performing (lowest complexity-loss combined score) models for each output variable provide statistical evidence to estimate the probability that each directed edge (input → output) truly represents a causal relationship.

- **Database Storage (SQLite):**  
  All models, performance metrics, complexity measurements, and DAG inference results are transparently stored in a SQLite database, ensuring reproducibility and auditability.

---

### 3. Final Comprehensive Analysis (`automated_passive_causal_determination_final_analysis.py`)

This script thoroughly examines and synthesizes the results from the modeling step. Its responsibilities include:

- **Edge Confidence Analysis:**  
  Clearly distinguishing strongly supported causal edges from less certain ones by explicitly analyzing edge confidence scores.

- **Top Model Identification:**  
  Analyzing top-performing models (top 2% by combined score), explicitly interpreting model simplicity and accuracy trade-offs.

- **Global DAG Synthesis:**  
  Integrating evidence across all top-performing models into a final inferred DAG with clear, probabilistic confidence scores for every inferred causal relationship.

---

### 4. Coordination Script (`coordination_script_for_automated_causal_discovery_process.py`)

Finally, to streamline execution and reproducibility, we provide a coordination script that sequentially executes the above scripts, explicitly documenting each step through extensive logging messages. This script ensures a clear, reproducible causal discovery pipeline, explicitly echoing our initial motivation and deep theoretical reasoning behind each methodological choice.

---

## Installation and Usage

**Setup (with `uv`):**
```shell
uv pip sync
```

**Run the complete pipeline:**
```shell
python coordination_script_for_automated_causal_discovery_process.py
```

---

## Summary of Core Contributions and Insights

To recap, our innovative automated causal determination framework is driven by several core insights:

- **Parsimony as Causal Proxy:**  
  True causal relationships naturally simplify prediction. Thus, model simplicity combined with robust accuracy is a practical heuristic for inferring causality.

- **Model-Agnostic Methodology:**  
  By simultaneously leveraging Transformers (powerful, flexible) and XGBoost (interpretable, reliable), our approach robustly validates causal inference outcomes.

- **Practical, Intuitive Validation:**  
  By explicitly incorporating well-understood real-world datasets, our causal inference pipeline provides intuitive validation points, ensuring results align meaningfully with human intuition.

- **Explicit, Probabilistic DAG Output:**  
  Final causal relationships are expressed explicitly through a DAG with well-defined confidence scores, enabling actionable and interpretable downstream usage.

This project thus provides a comprehensive, innovative approach to automated causal discovery, explicitly addressing key limitations in classical methods (like do-calculus), leveraging modern machine learning techniques, and grounding theoretical insights in practical validation.

---

## Contributing and Future Work

We welcome community contributions and further validation on additional datasets. Open issues, discussions, or PRs to continue improving the robustness, interpretability, and scalability of automated causal discovery techniques are highly encouraged.

---

## License

MIT License.

---

## Authors

- Jeffrey Emanuel ([GitHub](https://github.com/Dicklesworthstone))

---

## Acknowledgments

- Inspired by and built upon extensive theoretical and practical discussions regarding the limitations of existing causal inference methods and the crucial role of parsimony and complexity considerations in scientific modeling.
