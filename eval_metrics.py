import argparse
import csv
import json
import re
import sys
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bs_score 


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = re.sub(r"</?think\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


PATENT_FIELDS = [
    "title", "technical_field", "background_art",
    "summary_of_invention", "brief_description_of_drawing",
    "detailed_description",
]


def _find_balanced_json(text: str):
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    cand = text[start:]
    if in_str:
        cand += '"'
    cand += "}" * max(depth, 0)
    return cand


def _extract_fields_by_regex(text: str) -> dict:
    out = {}
    decoder = json.JSONDecoder()
    for key in PATENT_FIELDS:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*', text)
        if not m:
            continue
        try:
            value, _ = decoder.raw_decode(text[m.end():])
            if isinstance(value, str):
                out[key] = value
        except json.JSONDecodeError:
            continue
    return out


def _coerce_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


def parse_task2(raw: str) -> tuple[dict, str]:
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return {k: _coerce_str(d.get(k, "")) for k in PATENT_FIELDS}, "json_valid"
    except json.JSONDecodeError:
        pass
    js = _find_balanced_json(text)
    if js:
        try:
            d = json.loads(js)
            if isinstance(d, dict):
                return {k: _coerce_str(d.get(k, "")) for k in PATENT_FIELDS}, "json_recovered"
        except json.JSONDecodeError:
            pass
    fields = _extract_fields_by_regex(text)
    out = {k: fields.get(k, "") for k in PATENT_FIELDS}
    status = "regex_fallback" if any(v for v in out.values()) else "parse_failed"
    return out, status


# Metrics
def corpus_bleu(preds: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_bleu(preds, [refs]).score


def rouge_l_per_sample(preds: list[str], refs: list[str]) -> list[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def bertscore_per_sample(preds: list[str], refs: list[str], lang_hint: str = "en"):
    P, R, F = bs_score(
        preds, refs,
        model_type="roberta-large",
        num_layers=17,                # paper-recommended layer for roberta-large
        lang=lang_hint,
        rescale_with_baseline=True,
        verbose=False,
    )
    return P.tolist(), R.tolist(), F.tolist()


def _safe_pair(preds: list[str], refs: list[str]) -> tuple[list[str], list[str], list[int]]:
    keep_p, keep_r, keep_i = [], [], []
    for i, (p, r) in enumerate(zip(preds, refs)):
        if p and p.strip() and r and r.strip():
            keep_p.append(p)
            keep_r.append(r)
            keep_i.append(i)
    return keep_p, keep_r, keep_i


def evaluate(preds: list[str], refs: list[str], label: str = "") -> tuple[dict, list]:
    kp, kr, ki = _safe_pair(preds, refs)
    if not kp:
        empty = {"label": label, "n": 0, "n_skipped": len(preds),
                 "bleu": None, "rougeL_mean": None,
                 "bertscore_precision_mean": None, "bertscore_recall_mean": None,
                 "bertscore_f1_mean": None}
        return empty, []

    bleu = corpus_bleu(kp, kr)
    rouge_per = rouge_l_per_sample(kp, kr)
    P, R, F = bertscore_per_sample(kp, kr)

    summary = {
        "label": label,
        "n": len(kp),
        "n_skipped": len(preds) - len(kp),
        "bleu": bleu,
        "rougeL_mean": sum(rouge_per) / len(rouge_per),
        "bertscore_precision_mean": sum(P) / len(P),
        "bertscore_recall_mean":    sum(R) / len(R),
        "bertscore_f1_mean":        sum(F) / len(F),
    }
    per_sample = list(zip(ki, rouge_per, P, R, F))
    return summary, per_sample


def write_per_sample_csv(
    out_path: Path,
    predictions: list[dict],
    overall_per_sample: list,         
    per_field_per_sample: dict | None,  
) -> None:
    overall_lookup = {t[0]: t for t in overall_per_sample}
    field_lookups = {}
    if per_field_per_sample:
        for f, rows in per_field_per_sample.items():
            field_lookups[f] = {t[0]: t for t in rows}

    header = ["index", "rougeL", "bertscore_P", "bertscore_R", "bertscore_F1"]
    if field_lookups:
        for f in PATENT_FIELDS:
            header += [f"rougeL_{f}", f"bertP_{f}", f"bertR_{f}", f"bertF1_{f}"]

    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for i, p in enumerate(predictions):
            row = [p["index"]]
            if i in overall_lookup:
                _, rg, P, R, F = overall_lookup[i]
                row += [rg, P, R, F]
            else:
                row += ["", "", "", ""]
            if field_lookups:
                for f in PATENT_FIELDS:
                    rec = field_lookups[f].get(i)
                    if rec is None:
                        row += ["", "", "", ""]
                    else:
                        _, rg, P, R, F1 = rec
                        row += [rg, P, R, F1]
            w.writerow(row)


def mean_gen_time(preds_records: list[dict]) -> float | None:
    times = [p["gen_time_sec"] for p in preds_records
             if isinstance(p.get("gen_time_sec"), (int, float))]
    return sum(times) / len(times) if times else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_json", type=Path,
                    help="Path to predictions_<task>_<model>.json")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory to write metrics outputs (defaults to alongside the input)")
    args = ap.parse_args()

    in_path: Path = args.predictions_json
    if not in_path.exists():
        sys.exit(f"File not found: {in_path}")

    out_dir = args.out_dir or in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem 

    payload = json.load(open(in_path, "r", encoding="utf-8"))
    meta = payload.get("_meta", {})
    preds_records = payload["predictions"]
    task = meta.get("task", "")

    raw_preds = [strip_think_tags(p["prediction"]) for p in preds_records]
    raw_refs  = [strip_think_tags(p["reference"])  for p in preds_records]

    print("=" * 70)
    print(f"Predictions file : {in_path.name}")
    print(f"Task             : {task}")
    print(f"Model / adapter  : {meta.get('base_model')}  +  {meta.get('adapter')}")
    print(f"Decoding         : {meta.get('decoding')}  (seed={meta.get('seed')})")
    print(f"# predictions    : {len(preds_records)}")
    print("=" * 70)

    # Overall
    print("\nComputing BLEU / ROUGE-L / BERTScore on raw outputs...")
    summary, overall_per_sample = evaluate(raw_preds, raw_refs, label="overall")

    summary_out = {
        "_meta": {**meta, "input_file": in_path.name},
        "overall": summary,
    }
    mean_time = mean_gen_time(preds_records)
    if mean_time is not None:
        summary_out["mean_gen_time_sec"] = mean_time
    per_field_per_sample = None

    # Task 2: structural + per-field metrics
    if task == "task_2" or "task_2" in stem:
        print("\nParsing JSON outputs...")
        parsed_pairs = [parse_task2(p) for p in raw_preds]
        parsed_preds = [pair[0] for pair in parsed_pairs]
        parse_statuses = [pair[1] for pair in parsed_pairs]
        parsed_refs  = [parse_task2(r)[0] for r in raw_refs]

        n_total = len(raw_preds)
        status_counts = {s: parse_statuses.count(s) for s in
                         ["json_valid", "json_recovered", "regex_fallback", "parse_failed"]}
        json_valid_rate = status_counts["json_valid"] / n_total if n_total else 0.0
        json_recoverable_rate = (status_counts["json_valid"] + status_counts["json_recovered"]) / n_total if n_total else 0.0

        per_field_presence = {f: 0 for f in PATENT_FIELDS}
        coverage_counts = []  
        all6_count = 0
        for d in parsed_preds:
            present = sum(1 for f in PATENT_FIELDS if (d.get(f) or "").strip())
            coverage_counts.append(present)
            if present == len(PATENT_FIELDS):
                all6_count += 1
            for f in PATENT_FIELDS:
                if (d.get(f) or "").strip():
                    per_field_presence[f] += 1

        structural = {
            "n": n_total,
            "json_valid_rate":       json_valid_rate,
            "json_recoverable_rate": json_recoverable_rate,
            "parse_status_counts":   status_counts,
            "all_fields_present_rate": all6_count / n_total if n_total else 0.0,
            "avg_field_coverage":      (sum(coverage_counts) / (len(PATENT_FIELDS) * n_total)) if n_total else 0.0,
            "field_presence_rate":   {f: per_field_presence[f] / n_total for f in PATENT_FIELDS} if n_total else {},
        }
        summary_out["structural"] = structural

        print(f"  JSON-valid rate          : {structural['json_valid_rate']*100:.1f}%  "
              f"({status_counts['json_valid']}/{n_total})")
        print(f"  JSON-recoverable rate    : {structural['json_recoverable_rate']*100:.1f}%  "
              f"(strict + balanced substring)")
        print(f"  All 6 fields present     : {structural['all_fields_present_rate']*100:.1f}%  "
              f"({all6_count}/{n_total})")
        print(f"  Avg field coverage (/6)  : {structural['avg_field_coverage']*100:.1f}%")
        for f in PATENT_FIELDS:
            print(f"    presence[{f:32s}] = {structural['field_presence_rate'][f]*100:5.1f}%")

        print("\nComputing BLEU / ROUGE-L / BERTScore on parsed fields...")
        per_field = {}
        per_field_per_sample = {}
        for field in PATENT_FIELDS:
            fp = [d.get(field, "") for d in parsed_preds]
            fr = [d.get(field, "") for d in parsed_refs]
            field_summary, field_per_sample = evaluate(fp, fr, label=field)
            field_summary["presence_rate"] = structural["field_presence_rate"].get(field, 0.0)
            per_field[field] = field_summary
            per_field_per_sample[field] = field_per_sample
            s = field_summary
            if s["n"]:
                print(f"  {field:32s}  n={s['n']:3d}  presence={s['presence_rate']*100:5.1f}%  "
                      f"BLEU={s['bleu']:.2f}  "
                      f"ROUGE-L={s['rougeL_mean']:.4f}  "
                      f"BERT-F1={s['bertscore_f1_mean']:.4f}")
            else:
                print(f"  {field:32s}  presence={s['presence_rate']*100:5.1f}%  skipped (no valid pairs)")
        summary_out["per_field"] = per_field

    write_per_sample_csv(
        out_dir / f"{stem}_metrics_per_sample.csv",
        preds_records,
        overall_per_sample,
        per_field_per_sample,
    )

    summary_path = out_dir / f"{stem}_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    s = summary
    print("\n" + "=" * 70)
    print("OVERALL (raw text)")
    print("=" * 70)
    print(f"  n              : {s['n']} (skipped={s['n_skipped']})")
    print(f"  BLEU           : {s['bleu']:.2f}")
    print(f"  ROUGE-L (mean) : {s['rougeL_mean']:.4f}")
    print(f"  BERTScore P    : {s['bertscore_precision_mean']:.4f}")
    print(f"  BERTScore R    : {s['bertscore_recall_mean']:.4f}")
    print(f"  BERTScore F1   : {s['bertscore_f1_mean']:.4f}")
    if mean_time is not None:
        print(f"  Mean gen time  : {mean_time:.2f} s/sample")
    print(f"\nSummary  → {summary_path}")
    print(f"Per-row  → {out_dir / (stem + '_metrics_per_sample.csv')}")


if __name__ == "__main__":
    main()
