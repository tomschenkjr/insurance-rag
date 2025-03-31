from docx import Document
import os

def extract_text_from_docx(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            doc = Document(os.path.join(folder_path, filename))
            full_text = "\n".join([para.text for para in doc.paragraphs])
            documents.append((filename, full_text))
    return documents