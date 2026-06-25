import json
import os
import glob

def prepare_patent_dataset(input_dir, output_dir):
    # Make output folder if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Mencari semua file .json di dalam direktori input
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    task1_data = []
    task2_data = []

    skipped = 0
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                entry = json.load(f)
            
            abstract = entry.get("abstract", "")
            claims_list = entry.get("claims", [])
            
            # Pastikan claims adalah string
            if isinstance(claims_list, str):
                claims = claims_list
            else:
                claims = "\n".join(claims_list)
            
            # Data validation: skip jika abstract atau claims kosong
            if not abstract.strip() or not claims.strip():
                skipped += 1
                print(f"Skipped {os.path.basename(file_path)}: empty abstract or claims")
                continue
            
            # Task 1: Abstract -> Claims
            task1_data.append({
                "instruction": "Generate patent claims based on the provided abstract. The output should follow the language of the input or as requested. The requested language is ENGLISH.",
                "input": abstract,
                "output": claims
            })
            
            # Task 2: Abstract + Claims -> Remaining Sections 
            # Output hanya bagian-bagian yang perlu di-generate, bukan full document
            remaining_sections = {
                "title": entry.get("title", ""),
                "technical_field": entry.get("technical_field", ""),
                "background_art": entry.get("background_art", ""),
                "summary_of_invention": entry.get("summary_of_invention", ""),
                "brief_description_of_drawing": entry.get("brief_description_of_drawing", ""),
                "detailed_description": entry.get("detailed_description", "")
            }
            task2_data.append({
                "instruction": "Generate the remaining sections of a patent document in JSON format based on the abstract and claims provided. The output should include: title, technical_field, background_art, summary_of_invention, brief_description_of_drawing, and detailed_description. The output should follow the language of the input or as requested. The requested language is ENGLISH.",
                "input": f"Abstract: {abstract}\n\nClaims: {claims}",
                "output": json.dumps(remaining_sections, indent=2, ensure_ascii=False)
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Simpan hasil gabungan
    with open(os.path.join(output_dir, 'task_1_multilingual.json'), 'w') as f:
        json.dump(task1_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'task_2_multilingual.json'), 'w') as f:
        json.dump(task2_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully prepared datasets in {output_dir}")
    print(f"Task 1 samples: {len(task1_data)}")
    print(f"Task 2 samples: {len(task2_data)}")
    print(f"Skipped (empty data): {skipped}")

if __name__ == "__main__":
    INPUT_DIR = 'patent_jsons' 
    OUTPUT_DIR = 'patent_datasets'
    
    # Pastikan folder input ada sebelum menjalankan
    if os.path.exists(INPUT_DIR):
        prepare_patent_dataset(INPUT_DIR, OUTPUT_DIR)
    else:
        print(f"Directory '{INPUT_DIR}' not found.")
