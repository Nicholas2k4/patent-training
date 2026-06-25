import glob
import json
import os

from transformers import AutoTokenizer

# CONFIGURATION
INPUT_DIR = 'test_jsons'
OUTPUT_DIR = 'patent_datasets'

TASK1_OUT = os.path.join(OUTPUT_DIR, 'task_1_test.json')
TASK2_OUT = os.path.join(OUTPUT_DIR, 'task_2_test.json')

MAX_SEQ_LENGTH = 8192

MODEL_IDS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen3-8B",
]

SAFETY_MARGIN = 5

TASK1_INSTRUCTION = (
    "Generate patent claims based on the provided abstract. "
    "The output should follow the language of the input or as requested. "
    "The requested language is ENGLISH."
)

TASK2_INSTRUCTION = (
    "Generate the remaining sections of a patent document in JSON format "
    "based on the abstract and claims provided. The output should include: "
    "title, technical_field, background_art, summary_of_invention, "
    "brief_description_of_drawing, and detailed_description. "
    "The output should follow the language of the input or as requested. "
    "The requested language is ENGLISH."
)


# TRUNCATION HELPERS 
def build_messages(model_id, instruction, input_text, output_text):
    if "mistral" in model_id.lower():
        return [
            {"role": "user", "content": instruction + "\n\n" + input_text},
            {"role": "assistant", "content": output_text},
        ]
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]


def count_chat_tokens(tokenizer, messages):
    return len(tokenizer.apply_chat_template(messages, tokenize=True))


def find_worst_case(tokenizers_info, instruction, input_text, output_text):
    worst = (0, None, None)
    for model_id, tokenizer in tokenizers_info:
        messages = build_messages(model_id, instruction, input_text, output_text)
        n_tokens = count_chat_tokens(tokenizer, messages)
        if n_tokens > worst[0]:
            worst = (n_tokens, model_id, tokenizer)
    return worst


def truncate_sample(tokenizers_info, instruction, input_text, output_json_str, max_seq_length):
    orig_worst, _, worst_tok = find_worst_case(
        tokenizers_info, instruction, input_text, output_json_str
    )

    if orig_worst <= max_seq_length:
        return output_json_str, False, orig_worst, orig_worst

    sections = json.loads(output_json_str)
    dd_text = sections.get("detailed_description", "")

    if not dd_text:
        return None, True, orig_worst, orig_worst

    excess = orig_worst - max_seq_length + SAFETY_MARGIN
    dd_tokens = worst_tok.encode(dd_text)
    target_dd_len = len(dd_tokens) - excess

    if target_dd_len <= 0:
        return None, True, orig_worst, orig_worst

    truncated_dd = worst_tok.decode(
        dd_tokens[:target_dd_len], skip_special_tokens=True
    ).rstrip()

    if truncated_dd and not truncated_dd.endswith("."):
        last_period = truncated_dd.rfind(". ")
        if last_period > len(truncated_dd) * 0.8:
            truncated_dd = truncated_dd[:last_period + 1]

    sections["detailed_description"] = truncated_dd
    new_output = json.dumps(sections, indent=2, ensure_ascii=False)

    final_worst, _, _ = find_worst_case(
        tokenizers_info, instruction, input_text, new_output
    )

    while final_worst > max_seq_length and target_dd_len > 0:
        target_dd_len -= 1
        truncated_dd = worst_tok.decode(
            dd_tokens[:target_dd_len], skip_special_tokens=True
        ).rstrip()
        sections["detailed_description"] = truncated_dd
        new_output = json.dumps(sections, indent=2, ensure_ascii=False)
        final_worst, _, _ = find_worst_case(
            tokenizers_info, instruction, input_text, new_output
        )

    if final_worst > max_seq_length:
        return None, True, orig_worst, final_worst

    return new_output, True, orig_worst, final_worst


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Directory '{INPUT_DIR}' not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    print(f"Found {len(json_files)} JSON files in {INPUT_DIR}")

    task1_data = []
    task2_data = []
    skipped = 0

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                entry = json.load(f)

            abstract = entry.get("abstract", "")
            claims_list = entry.get("claims", [])
            claims = claims_list if isinstance(claims_list, str) else "\n".join(claims_list)

            if not abstract.strip() or not claims.strip():
                skipped += 1
                print(f"Skipped {os.path.basename(file_path)}: empty abstract or claims")
                continue

            task1_data.append({
                "instruction": TASK1_INSTRUCTION,
                "input": abstract,
                "output": claims,
            })

            remaining_sections = {
                "title": entry.get("title", ""),
                "technical_field": entry.get("technical_field", ""),
                "background_art": entry.get("background_art", ""),
                "summary_of_invention": entry.get("summary_of_invention", ""),
                "brief_description_of_drawing": entry.get("brief_description_of_drawing", ""),
                "detailed_description": entry.get("detailed_description", ""),
            }
            task2_data.append({
                "instruction": TASK2_INSTRUCTION,
                "input": f"Abstract: {abstract}\n\nClaims: {claims}",
                "output": json.dumps(remaining_sections, indent=2, ensure_ascii=False),
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    with open(TASK1_OUT, 'w', encoding='utf-8') as f:
        json.dump(task1_data, f, indent=2, ensure_ascii=False)
    print(f"\nTask 1 written: {TASK1_OUT} ({len(task1_data)} samples)")

    print("\nLoading tokenizers for task 2 truncation...")
    tokenizers_info = []
    for model_id in MODEL_IDS:
        print(f"  {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tokenizers_info.append((model_id, tok))

    print(f"\nTruncating task 2 (MAX_SEQ_LENGTH={MAX_SEQ_LENGTH})...")
    task2_truncated = []
    truncated_count = 0
    over_budget_skipped = 0

    for i, entry in enumerate(task2_data):
        new_output, was_truncated, orig_tokens, final_tokens = truncate_sample(
            tokenizers_info,
            entry["instruction"],
            entry["input"],
            entry["output"],
            MAX_SEQ_LENGTH,
        )

        if new_output is None:
            over_budget_skipped += 1
            print(f"  SKIP [{i}]: {orig_tokens} tokens cannot fit")
            continue

        if was_truncated:
            truncated_count += 1
            print(f"  TRUNC [{i}]: {orig_tokens} -> {final_tokens}")

        task2_truncated.append({
            "instruction": entry["instruction"],
            "input": entry["input"],
            "output": new_output,
        })

    with open(TASK2_OUT, 'w', encoding='utf-8') as f:
        json.dump(task2_truncated, f, indent=2, ensure_ascii=False)

    # ---- Summary ------------------------------------------------------------
    print(f"{'='*60}")
    print(f"  Source        : {INPUT_DIR} ({len(json_files)} files)")
    print(f"  Skipped empty : {skipped}")
    print(f"  Task 1 out    : {TASK1_OUT} ({len(task1_data)} samples)")
    print(f"  Task 2 out    : {TASK2_OUT} ({len(task2_truncated)} samples)")
    print(f"  Task 2 trunc  : {truncated_count}")
    print(f"  Task 2 skip   : {over_budget_skipped}")


if __name__ == "__main__":
    main()
