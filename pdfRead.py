from PyPDF2 import PdfReader

reader = PdfReader("/home/ubuntu/pdf/input.pdf")

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"--- Page {i+1} ---")
    print(text)
    print()

