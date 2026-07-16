import urllib.request
import os
import ssl
import sys

# Bypass SSL verification for downloads (sandbox may not have up-to-date certs)
ssl_ctx = ssl._create_unverified_context()

PDFS = {
    "DAEGC": "https://arxiv.org/pdf/1906.06532v1",
    "AGC":   "https://www.ijcai.org/proceedings/2019/0601.pdf",
    "DCRN":  "https://ojs.aaai.org/index.php/AAAI/article/view/20726/20485",
    "SDCN":  "https://arxiv.org/pdf/2002.01633",
    "SCAGC": "https://arxiv.org/pdf/2110.08264",
}

OUT_DIR = r"E:\Code\python code\WJ\pdfs"
os.makedirs(OUT_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

for name, url in PDFS.items():
    out_path = os.path.join(OUT_DIR, f"{name}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 50000:
        print(f"[SKIP] {name}: already downloaded ({os.path.getsize(out_path)} bytes)")
        continue
    print(f"[FETCH] {name} <- {url}")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60, context=ssl_ctx) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  -> saved {len(data)} bytes to {out_path}")
    except Exception as e:
        print(f"  !! ERROR: {e}")

print("Done downloading.")
