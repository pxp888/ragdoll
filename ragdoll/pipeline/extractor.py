from pathlib import Path
from typing import Optional
import io
from pypdf import PdfReader


def extract_text(file_path: Path, content_type: Optional[str] = None) -> str:
    """
    Extracts plain text from an uploaded file based on file type or extension.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf" or (content_type and "pdf" in content_type):
        reader = PdfReader(str(file_path))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n\n".join(text_parts)

    # For text, markdown, code, json, html, csv, etc.
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    # Try utf-8 first, fallback with replace
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")
