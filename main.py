import PyPDF2
from charactersplitter import RecursiveCharacterTextSplitter
from create_embeddings import create_embeddings
from embedding import get_relevant_chunks

def read_pdf(file_path):
    """Read text from a PDF file."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page_text = reader.pages[page_num].extract_text()
            text += page_text
    return text


# Example usage
pdf_file_path = 'data/FAQs MF.pdf'  # Replace 'example.pdf' with the path to your PDF file
pdf_text = read_pdf(pdf_file_path)

# Create Recursive Character TextSplitter instance with custom chunk size and overlap
custom_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Split the text
chunks = custom_splitter.split_text(pdf_text)

print("Chunks created successfully.")

# Create embeddings
create_embeddings(chunks)

print(get_relevant_chunks())
