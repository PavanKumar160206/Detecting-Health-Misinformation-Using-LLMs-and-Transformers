# Health Misinformation Detection

This project evaluates transformer models and a large language model for classifying health-related claims against evidence. Each example is assigned one of three labels:

- `Supports` - the evidence supports the claim
- `Refutes` - the evidence contradicts the claim
- `Neutral` - the evidence is insufficient or unrelated

The experiments use the **HealthVer** dataset and are implemented as Jupyter notebooks. The repository also contains a T5-based data augmentation experiment and locally saved fine-tuned model checkpoints.

## Repository contents

### Dataset

The CSV files are already included in the project root. Each file contains `id`, `evidence`, `claim`, `label`, `topic_ip`, and `question` columns.

| Split | File | Rows |
| --- | --- | ---: |
| Training | `healthver_train.csv` | 10,590 |
| Development | `healthver_dev.csv` | 1,917 |
| Test | `healthver_test.csv` | 1,823 |
| Augmented training | `healthver_train_augmented.csv` | 11,679 |

The augmented training set adds 1,089 `Refutes` examples generated with T5 paraphrasing.

### Experiments and checkpoints

| Experiment | Notebook | Saved checkpoint |
| --- | --- | --- |
| BioBERT baseline | `Models/Basline-BioBert/BioBert_Model.ipynb` | `Models/Basline-BioBert/biobert_baseline/` |
| BioBERT with updated weights | `Models/BioBert_With weights updated/BioBert_Updated_weights.ipynb` | `Models/BioBert_With weights updated/biobert_weighted/` |
| ClinicalBERT | `Models/ClinicalBert/ClinicalBert.ipynb` | None |
| SciBERT | `Models/SciBert/SciBert.ipynb` | `Models/SciBert/scibert_weighted/` |
| T5 data augmentation | `Models/T5_Augementation/DataAgumentation_With_T5.ipynb` | `Models/T5_Augementation/healthver_train_augmented.csv` |
| Meta-Llama 3 8B Instruct | `Models/Meta-Llama-3-8B-Instruct/nlp-llma.ipynb` | None |

The saved checkpoints contain model weights and tokenizer/configuration files. They are large binary artifacts, so retraining is not required when a checkpoint is available.

## Setup

Python 3.10 or later and a CUDA-capable GPU are recommended for model training. A GPU is effectively required for the Llama experiment.

Create and activate a virtual environment, then install the packages used by the notebooks:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install torch transformers datasets scikit-learn pandas numpy \
    sentence-transformers nltk jupyter accelerate peft bitsandbytes evaluate
```

Install the PyTorch build appropriate for your CUDA version if the default `torch` package is not suitable for your system.

## Running the experiments

Start Jupyter from the project root:

```bash
jupyter lab
```

Open the notebook for the experiment you want to run. The model notebooks expect the CSV files to be available from the project root. Run the T5 notebook first if you need to regenerate `healthver_train_augmented.csv`.

The Llama notebook was configured for a hosted GPU environment and includes its own package installation and Hugging Face authentication steps. A Hugging Face access token and sufficient GPU memory are required to run it.

## Project structure

```text
NLP Project/
|-- README.md
|-- healthver_train.csv
|-- healthver_dev.csv
|-- healthver_test.csv
|-- healthver_train_augmented.csv
|-- Models/
|   |-- Basline-BioBert/
|   |-- BioBert_With weights updated/
|   |-- ClinicalBert/
|   |-- Meta-Llama-3-8B-Instruct/
|   |-- SciBert/
|   `-- T5_Augementation/
|-- Health-Fact-Checking/
`-- dldpub.py
```

`Health-Fact-Checking/` is a separate reference data-processing codebase with its own README and requirements file. `dldpub.py` is a legacy PUBHEALTH download/preprocessing script and is not required for the HealthVer notebook experiments.

## Results

### Transformer models on the raw HealthVer dataset

| Model | Refute | Neutral | Support | Macro-F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| BioBERT | 0.6486 | 0.8446 | 0.6844 | 0.7258 | 0.7559 |
| RoBERTa | 0.6637 | 0.8446 | 0.6768 | 0.7284 | 0.7538 |
| ClinicalBERT | 0.5947 | 0.8178 | 0.6300 | 0.6808 | 0.7147 |
| SciBERT | 0.6307 | 0.8011 | 0.6249 | 0.6855 | 0.7121 |

### BioBERT ablation study

| Configuration | Refute | Neutral | Support | Macro-F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.6486 | 0.8446 | 0.6844 | 0.7258 | 0.7559 |
| With class weights | 0.6650 | 0.8342 | 0.6685 | 0.7226 | 0.7512 |
| With augmentation | 0.7264 | 0.8368 | 0.6926 | 0.7519 | 0.7720 |

### Zero-shot Meta-Llama-3-8B-Instruct

| Model | Refute | Neutral | Support | Macro-F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Meta-Llama-3-8B-Instruct | 0.7251 | 0.8654 | 0.7589 | 0.7831 | 0.8391 |

The best reported result is **0.8391 accuracy** and **0.7831 macro F1** from the zero-shot Meta-Llama-3-8B-Instruct experiment. Results can vary when notebooks are rerun because of model, hardware, and random-seed differences.

## Dataset reference

HealthVer: [Evidence-based Fact-Checking of Health-related Claims](https://aclanthology.org/2021.findings-emnlp.297/).
