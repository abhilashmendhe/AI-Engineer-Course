from pypdf import PdfReader
from langchain_core.documents import Document


def read_pdf_file(file_name):
    docs = []

    reader = PdfReader(file_name)
    for ind, page in enumerate(reader.pages):
        doc = Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_name, "page": ind + 1},
        )
        docs.append(doc)
    return docs


if __name__ == "__main__":
    docs = read_pdf_file("./symptoms.pdf")

    print(len(docs))
    print(docs[0])
