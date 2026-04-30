from huggingface_hub import snapshot_download
from huggingface_hub import HfApi, hf_hub_url
import os
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib

# --- KONFIGURASI ---
# Ganti dengan ID repository lo di Hugging Face
REPO_ID = "AITF-SR-02/dataset_sibi_extracted_md" 
# Target folder Bronze (Raw) sesuai struktur project kita
LOCAL_DIR = "./data/bronze_raw/dataset_sibi_extracted_md"

# Pattern untuk file .md baik di root maupun subfolder (nested)
ALLOW_PATTERNS = ["*.md", "*.MD", "**/*.md", "**/*.MD"]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def load_dotenv_if_present(dotenv_path: str | os.PathLike = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ (best-effort).

    Note: Python/uv doesn't auto-load .env, so we do it explicitly.
    Existing environment variables are NOT overwritten.
    """

    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value)
        if not key:
            continue

        os.environ.setdefault(key, value)


def resolve_hf_token() -> str | None:
    """Resolve HF token from env vars (after optional .env load)."""

    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _sanitize_windows_path_segment(segment: str) -> str:
    """Make a single path segment safe on Windows (no trailing spaces/dots, no reserved chars)."""

    segment = segment.strip().rstrip(". ")
    if not segment:
        segment = "_"

    invalid_chars = '<>:"/\\|?*'
    segment = "".join(("_" if ch in invalid_chars or ord(ch) < 32 else ch) for ch in segment)

    # Avoid reserved device names on Windows
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if segment.upper() in reserved:
        segment = f"{segment}_"

    return segment


def _sanitize_relpath_for_windows(rel_path: str) -> str:
    parts = []
    for part in rel_path.replace("\\", "/").split("/"):
        parts.append(_sanitize_windows_path_segment(part))
    return os.path.join(*parts)


def _download_file(url: str, dest_path: str, token: str | None) -> None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urlopen(req) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def safe_pull_data_from_hf(repo_id: str, local_dir: str, token: str | None, revision: str = "main") -> None:
    """Download files with sanitized paths to avoid Windows-invalid names in repo structure."""

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    md_files = [p for p in files if p.lower().endswith(".md")]

    abs_local_dir = os.path.abspath(local_dir)
    os.makedirs(abs_local_dir, exist_ok=True)

    print(f"🧰 Safe mode: download {len(md_files)} file .md (sanitize path untuk Windows)")

    # Pre-compute destination paths and handle collisions deterministically.
    # If multiple remote paths map to the same sanitized local path, we keep the first (sorted)
    # as-is and suffix the others with a short hash derived from the remote path.
    # Key by normcase path so Windows case-insensitive collisions are handled.
    base_key_to_items: dict[str, list[tuple[str, str]]] = {}
    for remote_path in md_files:
        safe_rel = _sanitize_relpath_for_windows(remote_path)
        base_dest = os.path.join(abs_local_dir, safe_rel)
        base_key = os.path.normcase(os.path.normpath(base_dest))
        base_key_to_items.setdefault(base_key, []).append((remote_path, base_dest))

    downloaded = 0
    skipped = 0
    total = len(md_files)

    for idx, remote_path in enumerate(md_files, start=1):
        safe_rel = _sanitize_relpath_for_windows(remote_path)
        base_dest = os.path.join(abs_local_dir, safe_rel)
        base_key = os.path.normcase(os.path.normpath(base_dest))
        items = base_key_to_items.get(base_key, [(remote_path, base_dest)])
        # Deterministic ordering by remote path
        items_sorted = sorted(items, key=lambda t: t[0])
        canonical_base_dest = items_sorted[0][1]
        remote_group = [rp for rp, _ in items_sorted]

        if len(remote_group) == 1 or remote_path == remote_group[0]:
            dest_path = canonical_base_dest
        else:
            stem, ext = os.path.splitext(canonical_base_dest)
            short = hashlib.sha1(remote_path.encode("utf-8")).hexdigest()[:8]
            dest_path = f"{stem}__{short}{ext}"

        if os.path.exists(dest_path):
            skipped += 1
        else:
            url = hf_hub_url(repo_id=repo_id, filename=remote_path, repo_type="dataset", revision=revision)
            _download_file(url, dest_path, token)
            downloaded += 1

        if idx % 25 == 0 or idx == total:
            print(f"⬇️  Progress {idx}/{total} (new: {downloaded}, skip: {skipped})")

def pull_data_from_hf(repo_id, local_dir, token=None):
    """
    Menarik data dari Hugging Face Hub sambil menjaga struktur folder asli.
    """
    print(f"🚀 Memulai sinkronisasi data dari {repo_id}...")

    abs_local_dir = os.path.abspath(local_dir)
    print(f"📁 CWD: {os.getcwd()}")
    print(f"📁 Target local_dir: {abs_local_dir}")
    print(f"🔐 HF token: {'TERDETEKSI' if token else 'TIDAK ADA'}")

    if not token:
        print(
            "⚠️ Kalau repo dataset kamu private/gated, kamu wajib autentikasi.\n"
            "   Opsi A (PowerShell, sementara):  $env:HF_TOKEN=\"<token>\"\n"
            "   Opsi B (env var standar):       $env:HUGGINGFACE_HUB_TOKEN=\"<token>\"\n"
            "   Opsi C (login CLI):             hf auth login   (atau)   huggingface-cli login\n"
        )
    
    try:
        os.makedirs(abs_local_dir, exist_ok=True)

        # snapshot_download otomatis nge-handle resume download kalau mati di tengah jalan
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=abs_local_dir,
            repo_type="dataset",
            # Kita cuma butuh file .md untuk CPT
            allow_patterns=ALLOW_PATTERNS,
            # Abaikan file cache HF atau file sampah lainnya
            ignore_patterns=[".git*", "README.md", ".cache/*"],
            token=token,
            # Menjaga symlinks agar tidak memakan storage ganda di beberapa OS
            local_dir_use_symlinks=False 
        )
        print(f"✅ Data berhasil ditarik ke: {path}")
        
        # Cek sekilas jumlah file .md yang benar-benar terunduh
        md_files = []
        for root, _, files in os.walk(abs_local_dir):
            for filename in files:
                if filename.lower().endswith(".md"):
                    md_files.append(os.path.join(root, filename))

        print(f"📊 Total file .md yang siap diproses: {len(md_files)} file")
        if len(md_files) == 0:
            print("⚠️ Tidak ada .md yang cocok dengan allow_patterns. Coba cek struktur folder di repo HF atau ubah pola filter.")
        else:
            sample = md_files[:5]
            print("🔎 Contoh file yang terunduh (maks 5):")
            for p in sample:
                print(f" - {p}")

    except Exception as e:
        msg = str(e)
        print(f"❌ Error saat narik data: {msg}")

        # Windows sering gagal kalau repo punya folder/file dengan trailing space/dot.
        # Fallback ke "safe mode" yang men-sanitize path saat menyimpan.
        if os.name == "nt" and ("No such file or directory" in msg or "Invalid argument" in msg or "Errno 2" in msg or "Errno 22" in msg):
            print("🔁 Fallback: mencoba safe mode downloader...")
            safe_pull_data_from_hf(repo_id, abs_local_dir, token=token)

if __name__ == "__main__":
    # Load token dari .env kalau ada
    load_dotenv_if_present(".env")

    os.makedirs(LOCAL_DIR, exist_ok=True)
    pull_data_from_hf(REPO_ID, LOCAL_DIR, token=resolve_hf_token())