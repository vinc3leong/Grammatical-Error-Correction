import torch
from transformers import BartForConditionalGeneration, BartTokenizer 
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_PATH = './bart-large-gec-finetune-model' 
TEST_FILE_SRC = '../../../preprocessing test data/bea2019-testdata'
OUTPUT_FILE = 'predictions_bart_large.txt'


MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
INFERENCE_BATCH_SIZE = 32 
NUM_BEAMS = 5


def run_evaluation():
    print("--- Starting GEC Evaluation for BART ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory not found at {MODEL_PATH}")
        print("Please make sure you have trained the model and it is saved at the correct location.")
        return

    print(f"Loading model and tokenizer from {MODEL_PATH}...")
    try:
        model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
        tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()

    print(f"Loading test data from {TEST_FILE_SRC}...")
    try:
        with open(TEST_FILE_SRC, 'r', encoding='utf-8') as f:
            source_lines_raw = f.readlines()
    except FileNotFoundError:
        print(f"Error: Test file not found at {TEST_FILE_SRC}")
        print("Please download it and place it in the same directory as this script.")
        return

    source_lines = []
    for line in source_lines_raw:
        line = line.strip()
        if line:
            source_lines.append(line)

    print(f"Loaded {len(source_lines)} sentences to correct.")

    all_predictions = []
    print(f"Generating corrections in batches of {INFERENCE_BATCH_SIZE}...")

    with torch.no_grad():
        for i in tqdm(range(0, len(source_lines), INFERENCE_BATCH_SIZE), desc="Generating corrections"):
            
            batch_lines = source_lines[i:i + INFERENCE_BATCH_SIZE]
            
            inputs = tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,  
                truncation=True,
                max_length=MAX_INPUT_LENGTH
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True
            )

            corrected_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            all_predictions.extend(corrected_batch)

    print(f"\nSaving {len(all_predictions)} predictions to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for line in all_predictions:
                f.write(line + '\n')
    except Exception as e:
        print(f"Error writing to output file: {e}")
        return

    print("\n--- Evaluation Finished ---")
    print(f"Successfully generated corrections and saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_evaluation()