## Declaration on the Use of AI

We utilized Google's Gemini 2.5 AI assistant to aid in specific aspects of this project's development and documentation. The AI's role was limited to that of an assistant, and its contributions include:

* **Report Refinement:** Assisting with refining grammar, improving clarity, and ensuring consistent styling for the accompanying LaTeX research paper.
* **Code Development:** Aiding in the debugging process by suggesting the placement of print statements to help trace program execution and inspect the state of variables.

All core logic, model implementation, experimental design, and final analysis were conducted by the project authors.


# Grammatical Error Correction with T5 and BART

This project systematically compares the performance of \texttt{T5} and \texttt{BART} models on the Grammatical Error Correction (GEC) task.

We implement and evaluate four model variants:
* `t5-base`
* `flan-t5-base`
* `bart-base`
* `bart-large`

The core of our methodology is a **two-stage fine-tuning pipeline** followed by an evaluation step. Each model's directory contains the three scripts necessary for this process.

## Project Structure

The code is organized by model architecture and size. Each model has its own directory containing the scripts, which have been pre-configured with the correct model names and paths.


## Prerequisites

1.  **Python 3.8+**
2.  **PyTorch** (with CUDA support recommended)
3.  **Hugging Face Transformers** and other dependencies. You can install all required packages using:
    ```bash
    pip install -r requirements.txt 
    ```
    *(You may need to create a `requirements.txt` file with libraries like `transformers`, `datasets`, `pandas`, `torch`, `tqdm`, `scikit-learn`)*

4.  **Data:**
    * All training and validation datasets (Lang-8, NUCLE, FCE, WI+Locness) must be placed in a root folder named `pre-processing data/` as shown in the structure above.
    * The final test file must be placed in a root folder named `preprocessing test data/`.
    * *Note: The scripts use hardcoded relative paths to this data.*

## How to Run a Model

The process involves three steps, executed in order. You must `cd` into the specific model directory you wish to run.

The following example uses `t5-base`.

---

### Step 1: Stage 1 - General Adaptation (training.py)

This script trains the base model (e.g., `t5-base`) on the large-scale Lang-8 corpus for general error correction.

1.  Navigate to the model's directory:
    ```bash
    cd src/t5/t5-base/
    ```

2.  Run the training script:
    ```bash
    python training.py
    ```

* **What it does:** Loads `t5-base` from Hugging Face, trains it on `lang8.train.auto.bea19`, and saves the resulting model checkpoint to a local directory named `./t5-gec-model`.

---

### Step 2: Stage 2 - Specialization (finetuning.py)

This script loads the checkpoint from Stage 1 (`./t5-gec-model`) and fine-tunes it on the smaller, high-quality "gold-standard" datasets (NUCLE, FCE, WI+Locness).

1.  Make sure you are still in the `src/t5/t5-base/` directory.

2.  Run the fine-tuning script:
    ```bash
    python finetuning.py
    ```

* **What it does:** Loads the model from `./t5-gec-model`, fine-tunes it, and saves the final, specialized model to `./t5-gec-finetune-model`. This is the final model used for evaluation.

---

### Step 3: Evaluation (evaluation.py)

This script loads the final model from Stage 2 (`./t5-gec-finetune-model`) and runs inference on the test set.

1.  Make sure you are still in the `src/t5/t5-base/` directory.

2.  Ensure the test file (`bea2019-testdata`) is in the `preprocessing test data/` folder at the project root.

3.  Run the evaluation script:
    ```bash
    python evaluation.py
    ```

* **What it does:** Loads the specialized model, generates corrections for every line in the test file, and saves the predictions to `predictions.txt` in the same directory.

---

## Running Other Models

To run `flan-t5`, `bart-base`, or `bart-large`, simply follow the exact same three steps, but `cd` into the corresponding directory first.

**Example for `bart-large`:**

```bash
# 1. Go to the correct directory
cd src/bart/bart-large/

# 2. Run Stage 1 training
python training.py

# 3. Run Stage 2 fine-tuning
python finetuning.py

# 4. Run evaluation
python evaluation.py

```

# How to Run the ESC Model

The ESC model is trained on the `bea-2019` dataset. The trained model is saved at `gec-imp/models/model.pt`.

## Steps to Evaluate the ESC Model

### 1. Prepare the Test File
Place your test file in `gec-imp/test-text/` and name it `source.txt`.

### 2. Output from Base Models
If required, place the outputs from base models in the same `gec-imp/test-text/` directory.

### 3. Set the Experiment Directory

```bash
export EXP_DIR=gec-imp
```

### 4. Run the Evaluation Script

```bash
python run.py --test \
  --data_dir $EXP_DIR/dev-text \
  --m2_dir $EXP_DIR/dev-m2 \
  --model_path $EXP_DIR/models/model.pt \
  --vocab_path $EXP_DIR/vocab.idx \
  --output_path $EXP_DIR/outputs/dev.out
```

This command runs inference on the test set, using the checkpoint stored at `gec-imp/models/model.pt`. Ensure that `vocab.idx` and all the necessary data directories exist under `gec-imp`.

### 5. Retrieve the Output
The output will be saved at:

```text
gec-imp/outputs/test.out
```

## Requirements

Ensure the following directories and files exist:
- `gec-imp/test-text/` - Test data directory
- `gec-imp/dev-text/` - Development data directory
- `gec-imp/dev-m2/` - M2 format files directory
- `gec-imp/models/model.pt` - Trained model checkpoint
- `gec-imp/vocab.idx` - Vocabulary index file
- `gec-imp/outputs/` - Output directory (will be created if it doesn't exist)