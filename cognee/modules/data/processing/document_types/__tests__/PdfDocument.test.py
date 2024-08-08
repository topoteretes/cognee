import os
from cognee.modules.data.processing.document_types.PdfDocument import PdfDocument

if __name__ == "__main__":
    test_file_path = os.path.join(os.path.dirname(__file__), "artificial-inteligence.pdf")
    pdf_doc = PdfDocument("Test document.pdf", test_file_path, chunking_strategy="paragraph")
    reader = pdf_doc.get_reader()

    for paragraph_data in reader.read():
        print(paragraph_data["word_count"])
        print(paragraph_data["text"])
        print(paragraph_data["cut_type"])
        print("\n")
