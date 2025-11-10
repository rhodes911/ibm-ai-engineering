# IBM AI Engineering with Python, PyTorch & TensorFlow

This repository contains my work and notes for the IBM AI Engineering Professional Certificate. It includes:

- **Jupyter notebooks** for labs and experiments (in `notebooks/`).
- **Markdown notes** summarizing key concepts (in `notes/`).
- A **progress tracker** (`progress.md`) to log daily and weekly study sessions.
- **Templates and instructions** to guide Copilot and maintain consistent quality.
- A **`.gitignore`** to ignore build and data files.
- A **`requirements.txt`** to capture Python dependencies.

## Structure

```
.
├── notebooks/          # Jupyter notebooks for coursework and experiments
├── notes/              # Markdown notes and summaries
├── data/               # Data files and datasets (ignored by git)
├── .github/
│   └── copilot-instructions.md   # Repo-wide Copilot instructions
├── .docs/
│   ├── notebooks.instructions.md # Additional Copilot instructions for notebooks
│   └── api.instructions.md       # Additional Copilot instructions for API code
├── .vscode/
│   └── settings.json             # VS Code settings (Copilot instructions)
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignore rules
├── progress.md        # Study progress log
└── README.md          # This file
```

## Getting Started

Clone the repository and set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

Make sure to update `progress.md` as you complete modules and labs.
