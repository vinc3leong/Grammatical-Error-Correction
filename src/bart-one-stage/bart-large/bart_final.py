#!/usr/bin/env python3

import os
import re
import math
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoConfig,
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)

from bart_validation import calculate_validation_score

try:
    import evaluate
    has_evaluate = True
except:
    has_evaluate = False


def load_sentence_pairs(folder):
    all_pairs = []
    source_files = sorted([f for f in os.listdir(folder) if f.endswith(".src")])
    
    for src_file in source_files:
        base_name = src_file[:-4]
        src_path = os.path.join(folder, src_file)
        tgt_path = os.path.join(folder, base_name + ".tgt")
        
        with open(src_path) as f1, open(tgt_path) as f2:
            sources = [line.strip() for line in f1]
            targets = [line.strip() for line in f2]
        
        for src, tgt in zip(sources, targets):
            all_pairs.append({"source": src, "target": tgt})
    
    return all_pairs


class GrammarDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_source_len, max_target_len):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        inputs = self.tokenizer(
            pair["source"],
            max_length=self.max_source_len,
            truncation=True
        )
        
        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer(
                pair["target"],
                max_length=self.max_target_len,
                truncation=True
            )
        
        inputs["labels"] = outputs["input_ids"]
        return inputs


def clean_text(text):
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+([)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text.strip()


def decode_model_output(predictions, labels, tokenizer):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.asarray(predictions)
    if predictions.ndim == 3:
        predictions = predictions.argmax(-1)
    
    labels = np.asarray(labels)
    
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    predictions = np.where(predictions < tokenizer.vocab_size, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    pred_sentences = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_sentences = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    return pred_sentences, label_sentences


def generate_corrections(model, tokenizer, sentences, max_len, num_beams, batch_size=32):
    model.eval()
    corrections = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=num_beams, max_length=max_len, early_stopping=True)
        corrections.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return corrections


def calculate_bleu_scores(predictions, references):
    def count_ngrams(words, n):
        ngrams = {}
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def get_corpus_bleu(hyps, refs, n):
        matches = total = hyp_len = ref_len = 0
        for hyp, ref in zip(hyps, refs):
            hyp_len += len(hyp)
            ref_len += len(ref)
            hyp_ng = count_ngrams(hyp, n)
            ref_ng = count_ngrams(ref, n)
            total += max(len(hyp) - n + 1, 0)
            for ng, cnt in hyp_ng.items():
                matches += min(cnt, ref_ng.get(ng, 0))
        
        if total == 0:
            return 0.0
        precision = matches / total
        bp = 0.0 if hyp_len == 0 else (1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len))
        return bp * precision * 100.0
    
    pred_tokens = [p.split() for p in predictions]
    ref_tokens = [r.split() for r in references]
    
    return {f'bleu{n}': get_corpus_bleu(pred_tokens, ref_tokens, n) for n in range(1, 5)}


class EpochLogger(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        print(f"\n{'='*70}\nEPOCH {epoch + 1} / {int(state.num_train_epochs)}\n{'='*70}\n")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n{'='*70}\nEPOCH {int(state.epoch)} COMPLETE\n{'='*70}\n")


class CheckpointLogger(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print(f"\nCheckpoint: step {state.global_step}, epoch {state.epoch:.2f}\n")


class BetterTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_score = 0.0
    
    def evaluate(self, *args, **kwargs):
        results = super().evaluate(*args, **kwargs)
        epoch = int(self.state.epoch) if self.state.epoch else 0
        f05 = results.get('eval_val_f0.5', 0)
        
        print(f"\n{'─'*70}\nValidation (Epoch {epoch})\n{'─'*70}")
        print(f"Loss: {results.get('eval_loss', 0):.4f}")
        print(f"BLEU: {results.get('eval_bleu', 0):.2f}")
        print(f"Precision: {results.get('eval_val_precision', 0):.4f}")
        print(f"Recall: {results.get('eval_val_recall', 0):.4f}")
        print(f"F0.5: {f05:.4f}{' NEW BEST!' if f05 > self.best_score else ''}\n{'─'*70}\n")
        
        if f05 > self.best_score:
            self.best_score = f05
        return results


def setup_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--train_subdir", type=str, default="train")
    parser.add_argument("--val_subdir", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--output_dir", type=str, default="./runs/bart-base-gec")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_src_len", type=int, default=128)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=200)
    return parser.parse_args()


def main():
    args = setup_arguments()
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train_folder = os.path.join(args.data_root, args.train_subdir)
    val_folder = os.path.join(args.data_root, args.val_subdir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.dropout = args.dropout
    model_config.attention_dropout = args.dropout
    model_config.activation_dropout = args.dropout
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=model_config)
    
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    
    train_pairs = load_sentence_pairs(train_folder)
    val_pairs = load_sentence_pairs(val_folder)
    
    train_dataset = GrammarDataset(train_pairs, tokenizer, args.max_src_len, args.max_tgt_len)
    val_dataset = GrammarDataset(val_pairs, tokenizer, args.max_src_len, args.max_tgt_len)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    
    val_sources = [pair['source'] for pair in val_pairs]
    
    if has_evaluate:
        bleu_metric = evaluate.load("sacrebleu")
        
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            pred_texts, ref_texts = decode_model_output(preds, labels, tokenizer)
            
            pred_texts = [clean_text(t) for t in pred_texts]
            ref_texts = [clean_text(t) for t in ref_texts]
            
            bleu_result = bleu_metric.compute(
                predictions=pred_texts,
                references=[[t] for t in ref_texts]
            )
            metrics = {"bleu": bleu_result["score"]}
            
            bleu_scores = calculate_bleu_scores(pred_texts, ref_texts)
            metrics.update({
                "bleu1": bleu_scores["bleu1"],
                "bleu2": bleu_scores["bleu2"],
                "bleu3": bleu_scores["bleu3"],
                "bleu4": bleu_scores["bleu4"],
            })
            
            errant_scores = calculate_validation_score(val_sources, pred_texts, ref_texts, args.output_dir)
            metrics.update({
                "val_precision": errant_scores["precision"],
                "val_recall": errant_scores["recall"],
                "val_f0.5": errant_scores["f0.5"],
            })
            
            return metrics
    else:
        compute_metrics = None
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        generation_num_beams=args.beam,
        load_best_model_at_end=True,
        metric_for_best_model="val_f0.5",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        seed=seed,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )
    
    callbacks = [
        EpochLogger(),
        CheckpointLogger(),
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    ]
    
    trainer = BetterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete in {elapsed/60:.1f} minutes. Model saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
