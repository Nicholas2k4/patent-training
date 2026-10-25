# Patent training and evaluation

This repository contains the patent datasets, preparation scripts, evaluation
scripts, and research notebooks. Run commands from this directory using Python 3.12.

## Local setup on macOS

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --no-compile -r requirements.txt
python -m pip check
```

The local environment includes the dependencies for all root Python scripts and
the local Manus/upload notebooks. Select `.venv/bin/python` as the interpreter
when opening a notebook in VS Code.

## Credentials and model downloads

`extract_pdf.py` reads `GEMINI_API` from `.env`.
`run_test_set_manus.ipynb` reads `MANUS_API_KEY` from `.env`.
Use `hf auth login` for access to gated/private Hugging Face repositories;
the existing `.secrets` file contains the Hugging Face tokens.
Keep `.env` and `.secrets` untracked.

Token analysis and truncation use the Llama, Mistral, and Qwen tokenizers.
The first run downloads any missing tokenizer files. Llama requires access to
the corresponding gated repository. BERTScore uses `roberta-large`; its model
weights must be available locally or downloaded on the first evaluation.

## Local commands

```sh
python extract_pdf.py
python prepare_llm_dataset.py
python prepare_llm_dataset_truncated.py
python prepare_test_dataset.py
python analyze_json_token.py
python analyze_dataset_token.py
python eval_compare.py
python eval_metrics.py --help
```

The preparation, analysis, and evaluation commands write their configured
datasets, CSV files, figures, or metrics. `extract_pdf.py` calls Gemini for PDFs
that do not already have a matching JSON file in `test_jsons/`.
For evaluation, pass a predictions JSON file to `eval_metrics.py`; use
`--out-dir` to choose where its results are saved.

## GPU research notebooks

`train_llm_lora.ipynb` and the training/model-inference notebooks in
`colab_runs/` are written for Google Colab, Google Drive, and a CUDA GPU.
They are not local macOS training entry points. All dependencies are declared in
the single `requirements.txt`: Linux/Colab also installs the training packages,
while macOS skips those packages to keep the local environment smaller.

The `wandb/` directory holds generated experiment logs and is not needed to
run the scripts. It can be deleted; the Python package `wandb` is installed in
the Linux training environment for notebooks that enable experiment tracking.
