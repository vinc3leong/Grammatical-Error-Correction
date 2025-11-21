## Declaration on the Use of AI

We utilized Google's Gemini 2.5 AI assistant to aid in specific aspects of this project's development and documentation. The AI's role was limited to that of an assistant, and its contributions include:

* **Report Refinement:** Assisting with refining grammar, improving clarity, and ensuring consistent styling for the accompanying LaTeX research paper.
* **Code Development:** Aiding in the debugging process by suggesting the placement of print statements to help trace program execution and inspect the state of variables.

All core logic, model implementation, experimental design, and final analysis were conducted by the project authors.

# Grammatical Error Correction with BART

This project implements fine-tuning of BART models (`bart-base` and `bart-large`) for the Grammatical Error Correction (GEC) task using the BEA-2019 shared task dataset.

## Project Structure

```
code/
├── bart_final.py        
├── bart_validation.py    
├── bart_testing.py        
├── run_bart_base.sh      
├── run_bart_large.sh   
└── README.md        

data/
├── train/            
├── validation/           
└── test/                
```

## Prerequisites

1. **Python 3.8+**
2. **PyTorch** (with CUDA support recommended)
3. **GPU Access** on SoC server
4. **Dependencies:** The training script will automatically install:
   - `torch>=2.2`
   - `transformers>=4.43`
   - `accelerate>=0.33`
   - `datasets>=2.20`
   - `evaluate>=0.4`
   - `sacrebleu>=2.0`
   - `errant>=3.0`
   - `spacy>=3.7`
   - `sentencepiece>=0.1.99`
   - `numpy>=1.24`

## Data Preparation

Before running the training script, you must prepare and place the preprocessed data files in the correct directories.

## How to Run

### Step 1: Allocate GPU on SoC Server

Request a GPU node for training:

```bash
srun --gpus=1 --time=24:00:00 --mem=64G --pty bash
```

### Step 2: Run Training

The training script handles everything automatically: environment setup, dependency installation, training, and validation.

**Option A: Interactive Mode**
```bash
cd ~/NLP/BART
./run_bart_base.sh
```

**What it does:**
* Creates a virtual environment with all dependencies
* Loads `facebook/bart-base` (or `facebook/bart-base`)
* Trains on all files in `data/train/`
* Validates on all files in `data/validation/` after every epoch
* Uses ERRANT F0.5 score to select the best model
* Implements early stopping with patience
* Saves the final model to `~/NLP/BART/runs/bart-base-gec-TIMESTAMP/`

**Training Progress:**
* Validation runs automatically after each epoch
* Best model is selected based on validation F0.5 score
* Training stops early if no improvement for 3 epochs (configurable)
* All metrics (BLEU, Precision, Recall, F0.5) are printed during training

### Step 3: Run Testing

After training completes, run the testing script on the CoNLL-2014 test set:

```bash
cd ~/NLP/BART
python bart_testing.py \
  --model_path ./runs/bart-base-gec-TIMESTAMP \
  --test_m2 ./data/test/official-2014.combined.m2 \
  --output_dir ./test_output \
  --batch_size 32 \
  --beam 5 \
  --fp16
```

**Replace `TIMESTAMP`** with the actual timestamp from your training run (e.g., `20250113-143022`).

**What it does:**
* Loads the trained model
* Extracts source sentences from the M2 file
* Generates corrections using beam search
* Computes ERRANT scores against the gold standard
* Saves predictions to `test_output/test.pred`
* Prints final test results (Precision, Recall, F0.5)

**Output files:**
* `test_output/test.src` - Original test sentences
* `test_output/test.pred` - Model predictions
* `test_output/test.system.m2` - System M2 file
* `test_output/test_results.txt` - ERRANT evaluation results