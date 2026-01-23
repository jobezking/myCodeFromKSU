from PyPDF2 import PdfReader

pdf = PdfReader("sample.pdf")
content = pdf.pages[0].extract_text()
print(content)