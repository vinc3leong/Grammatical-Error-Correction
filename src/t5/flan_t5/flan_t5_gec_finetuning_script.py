import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings("ignore")

TRAIN_SRC_FILES = [
    "../../pre-processing data/nucle/nucle.train.gold.bea19.src",
    "../../pre-processing data/fce_v2.1.bea19/fce.train.gold.bea19.src",
    "../../pre-processing data/wi+locness_v2.1.bea19/A.train.gold.bea19.src",
    "../../pre-processing data/wi+locness_v2.1.bea19/ABC.train.gold.bea19.src",
    "../../pre-processing data/wi+locness_v2.1.bea19/B.train.gold.bea19.src",
    "../../pre-processing data/wi+locness_v2.1.bea19/C.train.gold.bea19.src"

]

TRAIN_TGT_FILES = [
    "../../pre-processing data/nucle/nucle.train.gold.bea19.tgt",
    "../../pre-processing data/fce_v2.1.bea19/fce.train.gold.bea19.tgt",
    "../../pre-processing data/wi+locness_v2.1.bea19/A.train.gold.bea19.tgt",
    "../../pre-processing data/wi+locness_v2.1.bea19/ABC.train.gold.bea19.tgt",
    "../../pre-processing data/wi+locness_v2.1.bea19/B.train.gold.bea19.tgt",
    "../../pre-processing data/wi+locness_v2.1.bea19/C.train.gold.bea19.tgt"
]

DEV_SRC_FILES = [
    "../../pre-processing data/wi+locness_v2.1.bea19/ABCN.dev.gold.bea19.src",
    "../../pre-processing data/wi+locness_v2.1.bea19/B.dev.gold.bea19.src",
    "../../pre-processing data/wi+locness_v2.1.bea19/C.dev.gold.bea19.src",
    "../../pre-processing data/wi+locness_v2.1.bea19/N.dev.gold.bea19.src"
]

DEV_TGT_FILES = [
    "../../pre-processing data/wi+locness_v2.1.bea19/ABCN.dev.gold.bea19.tgt",
    "../../pre-processing data/wi+locness_v2.1.bea19/B.dev.gold.bea19.tgt",
    "../../pre-processing data/wi+locness_v2.1.bea19/C.dev.gold.bea19.tgt",
    "../../pre-processing data/wi+locness_v2.1.bea19/N.dev.gold.bea19.tgt"
]


# Model configuration
MODEL_NAME = 'google/flan-t5-base'
OUTPUT_DIR = './flan-t5-gec-finetune-model'
LOGGING_DIR = './flan-logs'

TASK_PREFIX = "Correct the grammatical error in this sentence: "

# Training hyperparameters
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5

def load_and_combine_data(src_files, tgt_files):
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
                        pairs.append({"input_text": TASK_PREFIX + src_line, "target_text": tgt_line})

        except FileNotFoundError:
            print(f"Warning: File not found. Skipping {src_file} and {tgt_file}")
        except Exception as e:
            print(f"Error reading {src_file}/{tgt_file}: {e}")

    if not pairs:
        print("Error: No data was loaded.")
        return None

    return pd.DataFrame(pairs)

def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples['input_text'],
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
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
        print("Exiting: Training data failed to load.")
        return
    if validation_df is None or validation_df.empty:
        print("Exiting: Validation data failed to load.")
        return

    train_dataset = Dataset.from_pandas(training_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    print(f"\nLoading tokenizer and model for '{MODEL_NAME}'...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

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
        warmup_steps=500,
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
        fp16=False,
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

    print("\nStarting model training...")
    trainer.train()
    print("Training finished.")

    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
