import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ─────────────────────────────────────────────────────────────────
# 1. KONFIGURASI REPO
# ─────────────────────────────────────────────────────────────────
# Ganti dengan nama repo tujuan lo (misal: "IlhamRafiqin/SekolahRakyat-Dataset")
REPO_ID = "AITF-SR-02/sr02-omahbse-k13-pdf" 

# Folder lokal yang mau di-push (isinya file .jsonl lo)
LOCAL_FOLDER = r".\data\dataset_k13_omahbse"

# ─────────────────────────────────────────────────────────────────
# 2. UTILS (Adopsi dari kode lo)
# ─────────────────────────────────────────────────────────────────

def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value

def load_dotenv_if_present(dotenv_path: str | os.PathLike = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), _strip_quotes(value))

def resolve_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# ─────────────────────────────────────────────────────────────────
# 3. PUSH ENGINE
# ─────────────────────────────────────────────────────────────────

def push_data_to_hf(repo_id: str, local_dir: str, token: str | None):
    api = HfApi(token=token)
    
    print(f"🚀 Memulai proses push ke: {repo_id}")
    print(f"📁 Local folder: {os.path.abspath(local_dir)}")
    
    if not token:
        print("❌ Error: Token HF tidak ditemukan! Isi .env dulu atau set environment variable.")
        return

    try:
        # 1. Cek/Buat Repo kalau belum ada
        print(f"🔍 Mengecek repository...")
        create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
        
        # 2. Upload Folder
        # path_in_repo="." berarti semua isi folder lokal bakal masuk ke root repo HF
        print(f"📤 Menunggah file ke Hugging Face Hub... (Mohon tunggu)")
        
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="dataset",
            # Opsional: kalau mau dimasukin ke subfolder tertentu di HF
            # path_in_repo="final_data", 
            commit_message="Add Gold and Silver School Dataset v14 (Smart-Cut)",
            token=token
        )
        
        print(f"✅ BERHASIL! Data lo sudah mendarat di: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"❌ Error saat push data: {str(e)}")

if __name__ == "__main__":
    # Load token dari .env
    load_dotenv_if_present(".env")
    
    # Eksekusi
    push_data_to_hf(REPO_ID, LOCAL_FOLDER, token=resolve_hf_token())