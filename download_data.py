from pathlib import Path
import os
import shutil
import zipfile
import gdown

def download(save_dir: Path):
    """Download the blender dataset."""

    # https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
    blender_file_id = "18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"

    final_path = save_dir / Path("blender")
    if os.path.exists(final_path):
        shutil.rmtree(str(final_path))
    url = f"https://drive.google.com/uc?id={blender_file_id}"
    download_path = save_dir / "blender_data.zip"
    gdown.download(url, output=str(download_path))
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(str(save_dir))
    unzip_path = save_dir / Path("nerf_synthetic")
    final_path = save_dir / Path("blender")
    unzip_path.rename(final_path)
    if download_path.exists():
        download_path.unlink()
        
if __name__ == "__main__":
    download(Path("data"))