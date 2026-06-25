import os
import glob
import json
from pydantic import BaseModel
from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv

# KONFIGURASI
load_dotenv()  # reads GEMINI_API from .env
API_KEY = os.environ["GEMINI_API"]
INPUT_FOLDER = "test_pdfs"
OUTPUT_FOLDER = "test_jsons" 
MODEL_NAME = "gemini-2.5-flash-lite" 

# JSON SCHEMA
class PatentSchema(BaseModel):
    title: str
    technical_field: str
    background_art: str
    summary_of_invention: str
    brief_description_of_drawing: str
    detailed_description: str
    claims: list[str]
    abstract: str

def main():
    client = genai.Client(api_key=API_KEY)
    
    # Buat folder output jika belum ada
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Cari semua file PDF
    pdf_files = glob.glob(os.path.join(INPUT_FOLDER, "**", "*.pdf"), recursive=True)
    pdf_files = sorted(pdf_files, key=lambda p: os.path.basename(p).lower())
    pdf_files = pdf_files[0:]
    print(f"Ditemukan {len(pdf_files)} file. Memulai proses...")

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        json_path = os.path.join(OUTPUT_FOLDER, file_name.replace(".pdf", ".json"))

        # Skip jika sudah ada
        if os.path.exists(json_path):
            print(f"Skipping {file_name} (sudah ada)")
            continue

        print(f"[{datetime.now()}] Processing: {file_name}...")

        try:
            # 1. Upload File
            uploaded_file = client.files.upload(file=pdf_path)

            # 2. Generate dengan Schema
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    uploaded_file,
                    "Extract this patent document into title, technical_field, background_art, summary_of_invention, brief_description_of_drawing, detailed_description, claim, dan abstract into json format using FULL text, DONT CUT ANYTHING"
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PatentSchema, 
                    temperature=0
                )
            )

            # 3. Simpan Hasil
            result_dict = response.parsed.model_dump()
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)
            
            print(f"Saved: {json_path}")

            # 4. Cleanup 
            client.files.delete(name=uploaded_file.name)

        except Exception as e:
            print(f"Error pada {file_name}: {e}")

if __name__ == "__main__":
    main()