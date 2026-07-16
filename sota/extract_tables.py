import os
import re
import PyPDF2

PDF_DIR = r"E:\Code\python code\WJ\pdfs"
OUT_DIR = r"E:\Code\python code\WJ\pdfs\text"
os.makedirs(OUT_DIR, exist_ok=True)

# Extract text from each PDF
for name in ["DAEGC", "AGC", "DCRN", "SDCN", "SCAGC"]:
    pdf_path = os.path.join(PDF_DIR, f"{name}.pdf")
    out_path = os.path.join(OUT_DIR, f"{name}.txt")
    print(f"\n=== Extracting {name} ===")
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        print(f"  pages: {len(reader.pages)}")
        text_parts = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            text_parts.append(f"\n--- PAGE {i+1} ---\n{t}")
        full = "".join(text_parts)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full)
        print(f"  -> wrote {len(full)} chars to {out_path}")
    except Exception as e:
        print(f"  !! ERROR: {e}")

print("\nDone extracting.")
