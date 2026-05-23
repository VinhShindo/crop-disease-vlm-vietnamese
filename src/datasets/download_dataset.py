from pathlib import Path
import zipfile
import shutil

# =========================================================
# CONFIG
# =========================================================

ZIP_FILE = Path("dataset/raw/rice_leaf_diseases.zip")

EXTRACT_PATH = Path("dataset/raw")

# =========================================================
# CHECK ZIP FILE
# =========================================================

if not ZIP_FILE.exists():

    print("=" * 60)
    print("DATASET ZIP NOT FOUND")
    print("=" * 60)

    print("\nPlease download dataset manually from Kaggle:")

    print(
        "https://www.kaggle.com/datasets/"
        "minhhuy2810/rice-diseases-image-dataset"
    )

    print("\nThen place the zip file here:")
    print(ZIP_FILE)

    exit()

# =========================================================
# EXTRACT ZIP
# =========================================================

print("=" * 60)
print("EXTRACTING DATASET")
print("=" * 60)

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("\nDataset extracted successfully.")

# =========================================================
# AUTO CLEAN STRUCTURE (OPTIONAL)
# =========================================================

print("\nChecking dataset structure...")

folders = [
    f for f in EXTRACT_PATH.iterdir()
    if f.is_dir()
]

for folder in folders:
    print("-", folder.name)

print("\nDone.")

# =========================================================
# FINAL
# =========================================================

print("\n")
print("=" * 60)
print("DATASET READY")
print("=" * 60)

print("\nExpected structure:")

print("""
dataset/
└── raw/
    ├── BrownSpot/
    ├── Healthy/
    ├── Hispa/
    └── LeafBlast/
""")