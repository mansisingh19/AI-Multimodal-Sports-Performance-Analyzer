"""Download the public cricket-shot video dataset (Hugging Face: rokmr/cricket-shot, apache-2.0)
and put the 3 shots we need into our folders.

Save this file as:  scripts/import_hf_cricket.py
Run (from the sports_ai folder):
    pip install huggingface_hub
    python scripts/import_hf_cricket.py --download        # downloads one 2.9 GB file, then extracts

Other options:
    python scripts/import_hf_cricket.py --tar "C:\\path\\cricketshot.tar.gz"   # file downloaded by hand
    python scripts/import_hf_cricket.py --src "C:\\path\\to\\extracted\\folder" # already extracted
"""
import argparse, shutil, sys, tarfile
from pathlib import Path, PurePosixPath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

# dataset folder name -> our folder name
MAP = {"cover": "cover_drive", "pull": "pull_shot", "defense": "defensive_shot"}
REPO = "rokmr/cricket-shot"
TAR_NAME = "cricketshot.tar.gz"

ap = argparse.ArgumentParser()
ap.add_argument("--download", action="store_true", help="download the dataset file from Hugging Face")
ap.add_argument("--tar", help="path of an already downloaded cricketshot.tar.gz")
ap.add_argument("--src", help="folder where the dataset is already extracted")
args = ap.parse_args()

counts = {v: 0 for v in MAP.values()}


def target_for(parts):
    """parts = path pieces like ('cricketshot','val','cover','cover_0023.avi')."""
    if len(parts) >= 2 and parts[-2] in MAP and Path(parts[-1]).suffix.lower() in config.VIDEO_TYPES:
        split = parts[-3] if len(parts) >= 3 else "x"
        folder = config.RAW_DIR / "cricket" / MAP[parts[-2]]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{split}_{parts[-1]}", MAP[parts[-2]]
    return None, None


if args.download:
    from huggingface_hub import hf_hub_download
    print("Downloading cricketshot.tar.gz (about 2.9 GB). Please wait and keep the internet on...")
    args.tar = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=TAR_NAME)
    print("Downloaded to:", args.tar)

if args.tar:
    print("Extracting only cover, pull and defense videos...")
    with tarfile.open(args.tar, "r:gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            dest, name = target_for(PurePosixPath(m.name).parts)
            if dest is not None:
                with tar.extractfile(m) as fin, open(dest, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                counts[name] += 1
elif args.src:
    for f in Path(args.src).rglob("*"):
        dest, name = target_for(f.parts)
        if dest is not None and f.is_file():
            shutil.copy2(f, dest)
            counts[name] += 1
else:
    sys.exit("Use one of: --download, --tar <file>, --src <folder>")

print("Copied videos:", counts)
if sum(counts.values()) == 0:
    print("No videos found. Please send me the output of this command so I can check the layout:")
    print('  python -c "import tarfile,sys; t=tarfile.open(sys.argv[1]); [print(next(t).name) for _ in range(5)]" <file>')
