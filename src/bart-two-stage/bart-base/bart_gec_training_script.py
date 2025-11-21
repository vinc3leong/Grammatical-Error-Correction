import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer,          
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings("ignore")

TRAIN_SRC_FILES = [
    "../../../pre-processing data/Lang-8 Corpus/lang8.train.auto.bea19.src"
]

TRAIN_TGT_FILES = [
    "../../../pre-processing data/Lang-8 Corpus/lang8.train.auto.bea19.tgt"
]

DEV_SRC_FILES = [
    "../../../pre-processing data/fce_v2.1.bea19/fce.dev.gold.bea19.src",
    "../../../pre-processing data/wi+locness_v2.1.bea19/A.dev.gold.bea19.src"
]

DEV_TGT_FILES = [
    "../../../pre-processing data/fce_v2.1.bea19/fce.dev.gold.bea19.tgt",
    "../../../pre-processing data/wi+locness_v2.1.bea19/A.dev.gold.bea19.tgt"
]

# Model configuration
MODEL_NAME = 'facebook/bart-base'  
OUTPUT_DIR = './bart-base-gec-model' 
LOGGING_DIR = './bart-logs'       


# Training hyperparameters
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16  
LEARNING_RATE = 5e-5
NUM_EPOCHS = 2

def load_and_combine_data(src_files, tgt_files):
    """Loads data from parallel .src and .tgt files and combines them."""
    pairs = []

    for src_file, tgt_file in zip(src_files, tgt_files):
        print(f"Loading data from {src_file} and {tgt_file}...")
        try:
            with open(src_file, 'r', encoding='utf-8') as f_src, \
                 open(tgt_file, 'r', encoding='utf-8') as f_tgt:

                for src_line, tgt_line in zip(f_src, f_tgt):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()

                    if src_line and tgt_line:
                        pairs.append({"input_text": src_line, "target_text": tgt_line})

        except FileNotFoundError:
            print(f"Warning: File not found. Skipping {src_file} and {tgt_file}")
            print("Please make sure all .src and .tgt files are in the same directory as this script.")
        except Exception as e:
            print(f"Error reading {src_file}/{tgt_file}: {e}")

    if not pairs:
        print("Error: No data was loaded. Please check your file paths and content.")
        return None

    return pd.DataFrame(pairs)

def tokenize_function(examples, tokenizer):
    """Tokenizes the input and target texts."""

    # Tokenize inputs
    model_inputs = tokenizer(
        examples['input_text'],
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )

    labels = tokenizer(
        examples['target_text'],
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    training_df = load_and_combine_data(TRAIN_SRC_FILES, TRAIN_TGT_FILES)
    validation_df = load_and_combine_data(DEV_SRC_FILES, DEV_TGT_FILES)

    if training_df is None or training_df.empty:
        return

    if validation_df is None or validation_df.empty:
        return

    train_dataset = Dataset.from_pandas(training_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    # --- Initialize Tokenizer and Model ---
    print(f"\nLoading tokenizer and model for '{MODEL_NAME}'...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    # --- Tokenize Datasets ---
    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer}
    )

    tokenized_validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer}
    )

    cols_to_remove = train_dataset.column_names
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(cols_to_remove)

    cols_to_remove = validation_dataset.column_names
    tokenized_validation_dataset = tokenized_validation_dataset.remove_columns(cols_to_remove)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=10000,
        logging_dir=LOGGING_DIR,
        logging_steps=100,          
        eval_strategy="steps",       
        eval_steps=500,             
        save_strategy="steps",       
        save_steps=500,
        load_best_model_at_end=True, 
        metric_for_best_model="loss",
        greater_is_better=False,     
        push_to_hub=False,
        fp16=torch.cuda.is_available(), 
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Start Training ---
    print("\nStarting model training...")
    trainer.train(resume_from_checkpoint=True)
    print("Training finished.")

    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()