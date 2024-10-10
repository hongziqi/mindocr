import os
import logging
import requests
from tqdm import tqdm

logger = logging.getLogger("docinsight")

def download_ckpt(url, save_dir=".docinsight/ckpt/"):
    """
    Download ckpt file from url
    """
    file_name = os.path.basename(url)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        return save_path
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        os.makedirs(save_dir, exist_ok=True)

        with open(save_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return save_path
    except Exception as e:
        logger.error(f"Download ckpt file failed: {e}")
        return None
