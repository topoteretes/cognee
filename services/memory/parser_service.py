import os
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document
import logging

logger = logging.getLogger("parser_service")

def parse_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise ValueError(f"Failed to read PDF file: {str(e)}")

def parse_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    try:
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error parsing DOCX: {e}")
        raise ValueError(f"Failed to read DOCX file: {str(e)}")

async def parse_url(url: str) -> str:
    """Fetches a URL and extracts clean body text."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (MemoryOS)"})
            response.raise_for_status()
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        return f"Scraped content from URL ({url}):\n\n{text}"
    except Exception as e:
        logger.error(f"Error scraping URL: {e}")
        raise ValueError(f"Failed to scrape webpage: {str(e)}")

def parse_file(file_path: str) -> str:
    """Dispatches file parsing based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".txt", ".md", ".json", ".csv"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
