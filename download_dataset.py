import os
import time
from mp_api.client import MPRester
from tqdm import tqdm

API_KEY = "hsdIEBjG4dQEEfO1TNx9S9qbuD1eECjK"
SAVE_DIR = "cif_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# 每批下载数量（调小避免超时）
CHUNK_SIZE = 50
NUM_CHUNKS = 100  # 总量大致 = CHUNK_SIZE * NUM_CHUNKS

def load_already_downloaded():
    files = os.listdir(SAVE_DIR)
    downloaded = set()
    for f in files:
        if f.endswith(".cif"):
            mid = f.split("_")[0]
            downloaded.add(mid)
    return downloaded

def fetch_docs_with_retry(mpr, num_chunks, chunk_size, retries=5):
    for attempt in range(retries):
        try:
            docs = mpr.materials.summary.search(
                fields=["material_id", "structure", "symmetry"],
                num_chunks=num_chunks,
                chunk_size=chunk_size
            )
            return docs
        except Exception as e:
            print(f"请求失败，正在重试({attempt+1}/{retries})，错误: {e}")
            time.sleep(5)
    raise RuntimeError("多次请求失败，终止程序")

def main():
    downloaded = load_already_downloaded()
    print(f"已经下载了 {len(downloaded)} 个材料结构，开始下载剩余部分...")

    with MPRester(API_KEY) as mpr:
        docs = fetch_docs_with_retry(mpr, NUM_CHUNKS, CHUNK_SIZE)

        for doc in tqdm(docs, desc="下载材料结构"):
            mid = doc.material_id
            if mid in downloaded:
                continue  # 跳过已经下载的

            structure = doc.structure
            sg = doc.symmetry.number if doc.symmetry is not None else "unknown"
            filename = os.path.join(SAVE_DIR, f"{mid}_SG{sg}.cif")

            try:
                structure.to(filename)
                downloaded.add(mid)
                print(f"已保存: {filename}")
            except Exception as e:
                print(f"保存文件失败: {filename}，错误: {e}")

if __name__ == "__main__":
    main()
