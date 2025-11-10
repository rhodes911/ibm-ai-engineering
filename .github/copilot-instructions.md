# Copilot Instructions (Repo-wide)

## General
- Always begin by outlining a short plan as a commented checklist before coding.
- Prefer clear, idiomatic Python 3.10+ with type hints and docstrings.
- Use Black formatting and Pylint-friendly code; keep functions small and pure.
- Explain non-trivial steps in comments (why, not just what).

## Notebooks
- For every .ipynb file, start with a "Run Me First" cell that installs/imports dependencies and sets seeds.
- Use deterministic seeds and note library versions at the end of each notebook.
- Show minimal, reproducible examples; avoid long cells—split into logical steps.

## ML / DL
- Structure training scripts as: config → data → model → train → eval → save.
- Log metrics clearly; prefer confusion matrix / ROC for classifiers, MAE/MSE for regressors.
- In TensorFlow/Keras and PyTorch, separate model definition from training loops.
- Provide a minimal training/evaluation example after defining any new model.

## Tests & Quality
- Generate basic unit tests when adding utilities using pytest style.
- Include a quick usage example in the docstring of every public function.

## Files & Project Hygiene
- Never commit credentials or large data; respect .gitignore.
- If adding dependencies, update requirements.txt and pin exact versions when possible.

## Responses in Chat
- When asked for code, return a single, complete file unless told otherwise.
- If unsure, ask one concise clarifying question, then propose a sensible default.
