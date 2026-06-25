import json
import os
import pathlib
import csv
import re
from collections import defaultdict
from jiwer import wer as compute_wer_lib
from rouge_score import rouge_scorer


# KONFIGURASI 
PATENT_JSON_DIR = "patent_jsons"
MANUAL_DIR = "manual_extractions"
OUTPUT_CSV = "eval_extraction_results.csv"

SECTIONS = [
    "title",
    "technical_field",
    "background_art",
    "summary_of_invention",
    "brief_description_of_drawing",
    "detailed_description",
    "claims",
    "abstract"
]


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

def normalize_text(text):
    if isinstance(text, list):
        text = "\n".join(text)
    text = str(text)
    # Normalize whitespace
    text = " ".join(text.split())
    # Hapus punctuation 
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase untuk fair comparison
    text = text.lower()
    # Re-normalize whitespace setelah strip punctuation
    text = " ".join(text.split())
    return text.strip()

def compute_wer(reference, candidate):
    if not reference:
        return 0.0 if not candidate else 1.0
    if reference == candidate:
        return 0.0
    return compute_wer_lib(reference, candidate)

def compute_rouge_l(reference, candidate):
    if not reference and not candidate:
        return 1.0
    if not reference or not candidate:
        return 0.0
    if reference == candidate:
        return 1.0
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure


def main():
    manual_files = sorted(pathlib.Path(MANUAL_DIR).glob("*.json"))
    manual_files = [f for f in manual_files if f.name != "_sample_list.json"]
    
    if not manual_files:
        print(f"No file at '{MANUAL_DIR}'.")
        return
    
    print(f"Found {len(manual_files)} manual extraction files")
    
    all_results = []
    section_scores = defaultdict(lambda: {"wer": [], "rouge_l": []})
    
    for manual_file in manual_files:
        gemini_file = pathlib.Path(PATENT_JSON_DIR) / manual_file.name
        
        if not gemini_file.exists():
            print(f"Gemini file not found: {gemini_file.name}, skipped.")
            continue
        
        with open(manual_file, 'r') as f:
            manual_data = json.load(f)
        with open(gemini_file, 'r') as f:
            gemini_data = json.load(f)
        
        print(f"\n{manual_file.name}")
        print(f"{'─'*50}")
        
        file_scores = {"filename": manual_file.name}
        
        for section in SECTIONS:
            ref = normalize_text(manual_data.get(section, ""))
            cand = normalize_text(gemini_data.get(section, ""))
            
            if not ref:
                continue
            
            w = compute_wer(ref, cand)
            r = compute_rouge_l(ref, cand)
            
            section_scores[section]["wer"].append(w)
            section_scores[section]["rouge_l"].append(r)
            
            file_scores[f"{section}_wer"] = w
            file_scores[f"{section}_rouge_l"] = r
            
            print(f"{section:30s} | WER={w:.4f} | ROUGE-L={r:.4f}")
        
        # Calculate overall averages for this specific file
        file_wers = [file_scores[f"{s}_wer"] for s in SECTIONS if f"{s}_wer" in file_scores]
        file_rouges = [file_scores[f"{s}_rouge_l"] for s in SECTIONS if f"{s}_rouge_l" in file_scores]
        if file_wers:
            file_scores["overall_wer"] = sum(file_wers) / len(file_wers)
            file_scores["overall_rouge_l"] = sum(file_rouges) / len(file_rouges)
        else:
            file_scores["overall_wer"] = ""
            file_scores["overall_rouge_l"] = ""
            
        all_results.append(file_scores)
    
    # SUMMARY
    print(f"\n\n{'='*70}")
    print("SUMMARY PER SECTION (AVG)")
    print(f"{'='*70}")
    print(f"{'Section':30s} | {'WER':>8s} | {'ROUGE-L':>8s} | {'N':>3s}")
    print(f"{'─'*30}-+-{'─'*8}-+-{'─'*8}-+-{'─'*3}")
    
    overall = {"wer": [], "rouge_l": []}
    
    for section in SECTIONS:
        scores = section_scores[section]
        if not scores["wer"]:
            continue
        
        avg_wer = sum(scores["wer"]) / len(scores["wer"])
        avg_rouge = sum(scores["rouge_l"]) / len(scores["rouge_l"])
        n = len(scores["wer"])
        
        overall["wer"].extend(scores["wer"])
        overall["rouge_l"].extend(scores["rouge_l"])
        
        print(f"{section:30s} | {avg_wer:8.4f} | {avg_rouge:8.4f} | {n:3d}")
    
    if overall["wer"]:
        print(f"{'─'*30}-+-{'─'*8}-+-{'─'*8}-+-{'─'*3}")
        avg_wer = sum(overall["wer"]) / len(overall["wer"])
        avg_rouge = sum(overall["rouge_l"]) / len(overall["rouge_l"])
        n = len(overall["wer"])
        print(f"{'OVERALL':30s} | {avg_wer:8.4f} | {avg_rouge:8.4f} | {n:3d}")
    
    # EXPORT CSV 
    if all_results:
        fieldnames = ["filename"]
        for section in SECTIONS:
            fieldnames.extend([f"{section}_wer", f"{section}_rouge_l"])
        fieldnames.extend(["overall_wer", "overall_rouge_l"])
        
        # Buat baris OVERALL: rata-rata per section dari semua file
        overall_row = {"filename": "OVERALL (avg)"}
        for section in SECTIONS:
            scores = section_scores[section]
            if scores["wer"]:
                overall_row[f"{section}_wer"] = sum(scores["wer"]) / len(scores["wer"])
                overall_row[f"{section}_rouge_l"] = sum(scores["rouge_l"]) / len(scores["rouge_l"])
            else:
                overall_row[f"{section}_wer"] = ""
                overall_row[f"{section}_rouge_l"] = ""
                
        # Overall dari semua baris (grand average)
        if overall["wer"]:
            overall_row["overall_wer"] = sum(overall["wer"]) / len(overall["wer"])
            overall_row["overall_rouge_l"] = sum(overall["rouge_l"]) / len(overall["rouge_l"])
        else:
            overall_row["overall_wer"] = ""
            overall_row["overall_rouge_l"] = ""
        
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
            writer.writerow(overall_row)
        
        print(f"\nDetailed results exported to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
