# Contribution — Pranjaly

## Overview
My primary contribution was implementing the **main emoji-aware model** and ensuring our project produced **reproducible, well-documented experiments and analyses**. I also supported the research direction by reviewing relevant literature on emoji-aware sarcasm modeling and helping translate those ideas into a feasible NLU project within our deadline.

---

## Research & Project Direction
- Reviewed prior work on sarcasm detection and emoji-aware modeling, including papers that use **gating / fusion** mechanisms to combine multiple representations.
- Helped refine the project into a clear and testable research question:

  **RQ1:** Does keeping emojis improve sarcasm detection compared to removing emojis?  
  **RQ2:** Does an explicit emoji-aware fusion model outperform standard transformer baselines?

- Motivated the design choice to evaluate both:
  - **in-sequence emoji modeling** (text + emojis together), and
  - **explicit emoji modeling** (separate emoji stream + fusion),
  to understand which integration strategy works better.

---

## Main Model Implementation
- Implemented the project’s **main model**: an emoji-aware **Gated Fusion** architecture.
- Key idea: represent the tweet using two inputs:
  - a **text stream** (emoji removed) to model linguistic content, and
  - an **emoji stream** (emoji-only) to model emoji cues separately.
- Implemented fusion variants to enable controlled comparisons:
  - **Gated fusion (main)**
  - **Concat fusion (ablation)**
  - **Emoji-only (ablation / sanity check)**

This model design supports NLU-style analysis because it explicitly tests whether emojis provide additional pragmatic meaning beyond literal text.

---

## Training & Experiment Workflow
- Built and/or integrated the **training pipeline** for the main model and model variants:
  - consistent hyperparameters across runs,
  - checkpoint saving for best dev performance,
  - prediction saving for later analysis,
  - run metadata saved for reproducibility.
- Ran multiple controlled experiments and ensured outputs were organized and traceable using structured result folders and config logging.

---

## Evaluation & Analysis
- Implemented (or consolidated) an evaluation workflow that reports:
  - **Macro-F1** (primary metric due to class imbalance),
  - **Accuracy**,
  - **Confusion matrix**,
  - performance on **emoji vs no-emoji subsets**.
- Conducted **error analysis** by inspecting false positives/false negatives and selecting representative examples.
- Summarized findings into slide-ready insights, including:
  - evidence that emojis can improve performance (comparison of emoji removed vs emoji kept),
  - limitations of explicit fusion under sparse/ambiguous emoji signals,
  - qualitative patterns observed in errors (context dependence, pragmatic intent, emoji ambiguity).

---

## Documentation & Deliverables Support
- Contributed to project communication and deliverables by:
  - helping structure the final results summary and interpretation,
  - preparing slide content (research question, ablation explanation, error analysis examples),
  - ensuring outputs are easy to reproduce and present.

---

## Key Takeaway from My Contribution
My work enabled the project to go beyond “training one model” and instead provide a **controlled NLU investigation**:
- How much emojis help sarcasm detection,
- Whether explicit emoji fusion provides benefits beyond a strong transformer baseline,
- What kinds of errors remain and why sarcasm is difficult for language models.


