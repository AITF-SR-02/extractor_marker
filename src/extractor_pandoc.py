import os
import re
import json
import logging
import subprocess
import pypandoc
from pathlib import Path
from tqdm import tqdm

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_DIR = Path("./data/sr-sibi-process")
WORD_INPUT = BASE_DIR / "raw_word"
OUTPUT_DIR = BASE_DIR / "extracted_md"
METADATA_FILE = BASE_DIR / "metadata" / "details_cache.jsonl"

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

def convert_doc_to_docx(doc_path):
    try:
        subprocess.run(['lowriter', '--headless', '--convert-to', 'docx', '--outdir', str(doc_path.parent), str(doc_path)], check=True, stdout=subprocess.DEVNULL)
        return doc_path.with_suffix(".docx")
    except: return None

def run_word_extraction():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    meta_index = {}
    if METADATA_FILE.exists():
        for line in METADATA_FILE.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                meta_index[_sanitize(entry.get("slug", "")).lower()] = entry
            except: continue

    all_files = [f for f in WORD_INPUT.rglob("*.*") if f.suffix.lower() in [".doc", ".docx"]]
    logging.info(f"📝 Memulai Pandoc untuk {len(all_files)} file Word.")

    try:
        for file_path in tqdm(all_files, desc="Word Processing"):
            out_path = OUTPUT_DIR / file_path.relative_to(WORD_INPUT).with_suffix(".md")
            if out_path.exists(): continue

            meta = meta_index.get(_sanitize(file_path.stem).lower(), {})
            role = "Siswa" if "siswa" in file_path.name.lower() else "Guru" if "guru" in file_path.name.lower() else ""

            try:
                target_file = file_path
                if file_path.suffix.lower() == ".doc":
                    target_file = convert_doc_to_docx(file_path)
                    if not target_file: continue
                
                text = pypandoc.convert_file(str(target_file), 'commonmark', format='docx')
                content = build_frontmatter(meta, role) + text
                
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")
                
                if file_path.suffix.lower() == ".doc": os.remove(target_file)
            except Exception as e:
                logging.error(f"❌ Gagal: {file_path.name} | {e}")
    except KeyboardInterrupt:
        logging.info("🛑 Stop paksa (Ctrl+C).")

if __name__ == "__main__":
    run_word_extraction()