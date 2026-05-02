import os
import re
import json
import gc
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- MARKER IMPORTS ---
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# --- CONFIGURATION (Safe Mode: 8-13 GB VRAM) ---
MAX_WORKERS = 2 
os.environ["INFERENCE_RAM"] = "10" 
os.environ["TORCH_DEVICE"] = "cuda"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Gunakan Absolute Path sesuai server lo
BASE_DIR = Path("/home/jovyan/SR_2/extractor_marker/data") 
PDF_INPUT = BASE_DIR / "bronze_raw"
OUTPUT_DIR = BASE_DIR / "silver"
METADATA_FILE = BASE_DIR / "metadata" / "details_cache.jsonl"
LOG_FILE = Path("extraction_progress.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def _sanitize(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "_", name.strip()) or "Lainnya"

def build_frontmatter(meta: dict, role: str) -> str:
    lines = ["---"]
    fields = ["title", "isbn", "kelas", "mata_pelajaran", "kurikulum", "jenjang", "penulis"]
    for f in fields:
        if v := meta.get(f):
            lines.append(f'{f}: "{str(v).replace(chr(34), "")}"')
    if role: lines.append(f'role: "{role.lower()}"')
    lines.append("---\n")
    return "\n".join(lines)

def process_pdf_worker(pdf_info):
    pdf_path, out_path, meta, role = pdf_info
    
    if not hasattr(process_pdf_worker, "converter"):
        process_pdf_worker.converter = PdfConverter(artifact_dict=create_model_dict())
    
    try:
        rendered = process_pdf_worker.converter(str(pdf_path))
        full_text, _, _ = text_from_rendered(rendered)
        
        content = build_frontmatter(meta, role) + full_text
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True, pdf_path.name
    except Exception as e:
        return False, f"{pdf_path.name}: {str(e)}"

def run_pdf_extraction():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    meta_index = {}
    if METADATA_FILE.exists():
        for line in METADATA_FILE.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                meta_index[_sanitize(entry.get("slug", "")).lower()] = entry
            except: continue

    all_pdfs = sorted(PDF_INPUT.rglob("*.pdf"))
    pending = []
    
    for pdf in all_pdfs:
        out_path = OUTPUT_DIR / pdf.relative_to(PDF_INPUT).with_suffix(".md")
        if out_path.exists():
            continue

        meta = meta_index.get(_sanitize(pdf.stem).lower(), {})
        role = "Siswa" if "siswa" in pdf.name.lower() else "Guru" if "guru" in pdf.name.lower() else ""
        pending.append((pdf, out_path, meta, role))

    logging.info(f"🚀 Total: {len(all_pdfs)} | Pending: {len(pending)}")
    
    if not pending:
        logging.info("✅ Selesai!")
        return

    try:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_pdf_worker, item) for item in pending]
            for future in tqdm(as_completed(futures), total=len(pending), desc="Processing"):
                success, msg = future.result()
                if not success:
                    logging.error(f"❌ {msg}")
    except KeyboardInterrupt:
        logging.info("\n🛑 Dihentikan.")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    run_pdf_extraction()