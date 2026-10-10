from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(directory: str | Path) -> list[dict[str, str]]:
    """Load supported documents and return their text and metadata."""
    document_path = Path(directory)

    if not document_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {document_path}"
        )

    if not document_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {document_path}"
        )

    documents: list[dict[str, str]] = []

    for path in sorted(document_path.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        else:
            text = path.read_text(encoding="utf-8")

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "source": path.name,
                "path": str(path),
                "text": text,
            }
        )

    return documents