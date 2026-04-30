import os
import re
import json
import time
import gc
import torch
import logging
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import HfApi

# --- MARKER IMPORTS[cite: 1] ---
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# 1. SETUP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("marker_extraction.log"), logging.StreamHandler()]
)

# 2. CONFIGURATION & PATHS
load_dotenv()

# BASE_DIR adalah satu-satunya path yang perlu lo sesuaikan Ham
BASE_DIR = Path("./data/sr-sibi-process") 

CONFIG = {
    "pdf_dir": BASE_DIR / "raw_pdf",
    "output_dir": BASE_DIR / "extracted_md",
    "details_cache": BASE_DIR / "metadata" / "details_cache.jsonl",
    "hf_cache": BASE_DIR / "cache" / "hf_cache",
    "hf_repo_id": "AITF-SR-02/dataset_sibi_extracted_md",
    "force_ocr": False,
    "force_reextract": False,
    "batch_size": 50,
    "langs": ["id"]
}

# Setup Environment untuk Hugging Face[cite: 1]
os.environ["HF_HOME"] = str(CONFIG["hf_cache"])
os.environ["TRANSFORMERS_CACHE"] = str(CONFIG["hf_cache"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 3. UTILITIES (Logic dari Colab lo[cite: 1])
def _sanitize(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "_", name.strip()) or "Lainnya"

def _get_priority(p: Path) -> int:
    # SMA (0) → SMP (1) → SD (2) → Others (9)[cite: 1]
    PRIORITY_MAP = {"SMA_SMK_MA": 0, "SMA": 0, "SMP_MTS": 1, "SMP": 1, "SD_MI": 2, "SD": 2}
    parts = [x.upper() for x in p.parts]
    for part in parts:
        for key, prio in PRIORITY_MAP.items():
            if key.upper() in part.upper(): return prio
    return 9

def build_frontmatter(meta: dict, role: str) -> str:
    lines = ["---"]
    # Fields sesuai requirement metadata lo[cite: 1]
    fields = ["title", "isbn", "kelas", "mata_pelajaran", "kurikulum", "jenjang", "penulis"]
    for f in fields:
        if v := meta.get(f):
            lines.append(f'{f}: "{str(v).replace(chr(34), "")}"')
    if role: lines.append(f'role: "{role.lower()}"')
    lines.append("---\n")
    return "\n".join(lines)

# 4. EXTRACTION ENGINE
def run_extraction():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        os.environ["INFERENCE_RAM"] = str(int(vram_gb))
        logging.info(f"✅ GPU Mode: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)[cite: 1]")
    
    # Load Metadata Index
    meta_index = {}
    if CONFIG["details_cache"].exists():
        for line in CONFIG["details_cache"].read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                meta_index[_sanitize(entry.get("slug", "")).lower()] = entry
            except: continue
        logging.info(f"📖 Loaded {len(meta_index)} metadata entries.[cite: 1]")

    # Scan PDFs & Apply Priority[cite: 1]
    all_pdfs = sorted(CONFIG["pdf_dir"].rglob("*.pdf"))
    pending = []
    for pdf in all_pdfs:
        meta = meta_index.get(_sanitize(pdf.stem).lower(), {})
        role = "Siswa" if "siswa" in pdf.name.lower() else "Guru" if "guru" in pdf.name.lower() else ""
        out_path = CONFIG["output_dir"] / pdf.relative_to(CONFIG["pdf_dir"]).with_suffix(".md")
        
        if out_path.exists() and not CONFIG["force_reextract"]:
            continue
        pending.append((_get_priority(pdf), pdf, out_path, meta, role))

    # Sort berdasarkan priority SMA -> SMP -> SD[cite: 1]
    pending.sort(key=lambda x: (x[0], x[1].name))
    logging.info(f"📊 Total PDFs: {len(all_pdfs)} | Pending: {len(pending)}")

    # Initialize Marker[cite: 1]
    logging.info("🔄 Initializing Marker Models...")
    converter = PdfConverter(artifact_dict=create_model_dict())

    success, failed = 0, 0
    
    try:
        for priority, pdf_path, out_path, meta, role in tqdm(pending, desc="Marker Extraction", unit="pdf"):
            try:
                rendered = converter(str(pdf_path))
                full_text, _, _ = text_from_rendered(rendered)

                content = build_frontmatter(meta, role) + full_text
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")
                success += 1
            except Exception as e:
                failed += 1
                logging.error(f"❌ Gagal proses {pdf_path.name}: {str(e)[:100]}")
            
            # Periodic Memory Cleanup[cite: 1]
            if success % 10 == 0:
                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logging.info("\n🛑 Ctrl+C dideteksi! Berhenti secara rapi...")
        return success, failed

    return success, failed

def upload_to_hf():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.warning("⚠️ HF_TOKEN tidak ada di .env. Skip upload.")
        return

    api = HfApi()
    try:
        logging.info(f"🚀 Uploading ke: {CONFIG['hf_repo_id']}")
        api.create_repo(repo_id=CONFIG["hf_repo_id"], repo_type="dataset", token=hf_token, exist_ok=True)
        api.upload_folder(
            folder_path=str(CONFIG["output_dir"]),
            repo_id=CONFIG["hf_repo_id"],
            repo_type="dataset",
            token=hf_token,
            commit_message=f"Upload SIBI Clean Extraction {datetime.now().strftime('%Y-%m-%d')}"
        )
        logging.info("✅ Upload Hugging Face Berhasil!")
    except Exception as e:
        logging.error(f"❌ Gagal upload: {e}")

if __name__ == "__main__":
    s, f = run_extraction()
    logging.info(f"🏁 Selesai. Sukses: {s} | Gagal: {f}")
    
    if s > 0:
        ans = input("\nMau upload hasil ke Hugging Face sekarang? (y/n): ")
        if ans.lower() == 'y':
            upload_to_hf()