#!/usr/bin/env python3

import os
import sys
import argparse
import shutil
import subprocess

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_test_sentences(m2_file):
    sentences = []
    with open(m2_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("S "):
                sentence = " ".join(line[2:].strip().split())
                sentences.append(sentence)
    return sentences


def save_sentences(filepath, sentences):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(sentences) + "\n")


def find_errant_tool(tool_name):
    path = shutil.which(tool_name)
    if path:
        return path
    bin_folder = os.path.join(sys.prefix, 'Scripts' if os.name == 'nt' else 'bin')
    for ext in ['', '.exe', '.py']:
        full_path = os.path.join(bin_folder, tool_name + ext)
        if os.path.exists(full_path):
            return full_path
    return None


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


def create_m2_file(source_file, corrected_file, output_m2):
    tool = find_errant_tool("errant_parallel")
    cmd = [tool] if tool else [sys.executable, "-m", "errant_parallel"]
    subprocess.run(cmd + ["-orig", source_file, "-cor", corrected_file, "-out", output_m2], check=True, capture_output=True)


def score_with_m2(system_m2, gold_m2):
    for tool in ["errant_compare", "m2scorer"]:
        path = find_errant_tool(tool)
        if path:
            cmd = [path, "-hyp", system_m2, "-ref", gold_m2] if tool == "errant_compare" else [path, system_m2, gold_m2]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = (result.stdout + result.stderr).strip()
            if output:
                return output
    return None


def setup_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_m2", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_prefix", type=str, default="test")
    return parser.parse_args()


def main():
    args = setup_arguments()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    dtype = torch.float16 if args.fp16 and device == 'cuda' else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device).eval()
    
    sentences = get_test_sentences(args.test_m2)
    corrections = generate_corrections(model, tokenizer, sentences, args.max_length, args.beam, args.batch_size)
    
    src_file = os.path.join(args.output_dir, f"{args.output_prefix}.src")
    pred_file = os.path.join(args.output_dir, f"{args.output_prefix}.pred")
    m2_file = os.path.join(args.output_dir, f"{args.output_prefix}.system.m2")
    
    save_sentences(src_file, sentences)
    save_sentences(pred_file, corrections)
    create_m2_file(src_file, pred_file, m2_file)
    
    output = score_with_m2(m2_file, args.test_m2)
    if output:
        print(output)
        open(os.path.join(args.output_dir, "test_results.txt"), "w").write(output)


if __name__ == "__main__":
    main()
