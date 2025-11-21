#!/usr/bin/env python3

import os
import sys
import re
import argparse
import shutil
import subprocess


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


def create_m2_file(source_file, corrected_file, output_m2):
    tool = find_errant_tool("errant_parallel")
    cmd = [tool] if tool else [sys.executable, "-m", "errant_parallel"]
    subprocess.run(cmd + ["-orig", source_file, "-cor", corrected_file, "-out", output_m2], check=True, capture_output=True)


def extract_scores_from_output(output_text):
    p_match = re.search(r"Precision[^0-9]*([0-9.]+)", output_text, re.I)
    r_match = re.search(r"Recall[^0-9]*([0-9.]+)", output_text, re.I)
    f_match = re.search(r"F0[._-]?5[^0-9]*([0-9.]+)", output_text, re.I)
    
    if p_match and r_match and f_match:
        p = float(p_match.group(1))
        r = float(r_match.group(1))
        f = float(f_match.group(1))
        if p > 1.0:
            p, r, f = p/100, r/100, f/100
        return {"precision": p, "recall": r, "f0.5": f}
    
    m = re.search(r"(?m)^\s*\d+\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", output_text)
    if m:
        p, r, f = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if p > 1.0:
            p, r, f = p/100, r/100, f/100
        return {"precision": p, "recall": r, "f0.5": f}
    return None


def score_with_m2(system_m2, gold_m2):
    for tool in ["errant_compare", "m2scorer"]:
        path = find_errant_tool(tool)
        if path:
            cmd = [path, "-hyp", system_m2, "-ref", gold_m2] if tool == "errant_compare" else [path, system_m2, gold_m2]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = (result.stdout + result.stderr).strip()
            scores = extract_scores_from_output(output)
            if scores:
                return scores
    return None


def calculate_validation_score(sources, predictions, references, output_folder):
    temp_folder = os.path.join(output_folder, "temp_validation")
    os.makedirs(temp_folder, exist_ok=True)
    
    source_file = os.path.join(temp_folder, "val.src")
    prediction_file = os.path.join(temp_folder, "val.pred")
    reference_file = os.path.join(temp_folder, "val.ref")
    system_m2 = os.path.join(temp_folder, "val.system.m2")
    gold_m2 = os.path.join(temp_folder, "val.gold.m2")
    
    save_sentences(source_file, sources)
    save_sentences(prediction_file, predictions)
    save_sentences(reference_file, references)
    
    create_m2_file(source_file, prediction_file, system_m2)
    create_m2_file(source_file, reference_file, gold_m2)
    
    scores = score_with_m2(system_m2, gold_m2)
    
    for file in [source_file, prediction_file, reference_file, system_m2, gold_m2]:
        try:
            os.remove(file)
        except:
            pass
    try:
        os.rmdir(temp_folder)
    except:
        pass
    
    return scores if scores else {"precision": 0.0, "recall": 0.0, "f0.5": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--references", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./validation_output")
    args = parser.parse_args()
    
    with open(args.sources) as f:
        sources = [line.strip() for line in f]
    with open(args.predictions) as f:
        predictions = [line.strip() for line in f]
    with open(args.references) as f:
        references = [line.strip() for line in f]
    
    scores = calculate_validation_score(sources, predictions, references, args.output_dir)
    
    print(f"Precision: {scores['precision']:.4f}")
    print(f"Recall: {scores['recall']:.4f}")
    print(f"F0.5: {scores['f0.5']:.4f}")


if __name__ == "__main__":
    main()
