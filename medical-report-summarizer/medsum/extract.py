"""Step 1: get plain text out of an uploaded report (text, PDF or image)."""

import io
import os

TEXT_EXT = {".txt", ".text", ".md"}
PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ALLOWED_EXT = TEXT_EXT | PDF_EXT | IMAGE_EXT


class ExtractionError(Exception):
    """Raised when text cannot be extracted from a file."""


def extract_text(filename, data):
    """Return the text inside an uploaded file given its name and raw bytes."""
    ext = os.path.splitext(filename.lower())[1]
    if ext in TEXT_EXT:
        return data.decode("utf-8", errors="ignore")
    if ext in PDF_EXT:
        return _from_pdf(data)
    if ext in IMAGE_EXT:
        return _from_image(data)
    raise ExtractionError("Unsupported file type '%s'. Use .txt, .pdf or an image." % ext)


def _from_pdf(data):
    text = ""
    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=data, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    except ImportError:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            raise ExtractionError("Install PyMuPDF or pdfplumber to read PDF files.")

    # A scanned PDF has no text layer: fall back to OCR on each page image.
    if len(text.strip()) < 20:
        text = _ocr_pdf(data)
    return text


def _ocr_pdf(data):
    try:
        import fitz
    except ImportError:
        raise ExtractionError("This PDF has no text layer and PyMuPDF is needed to OCR it.")
    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            pages.append(_from_image(pix.tobytes("png")))
    return "\n".join(pages)


def _from_image(data):
    try:
        import pytesseract
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        raise ExtractionError("Install pytesseract and Pillow (and the Tesseract program) to read images.")

    image = Image.open(io.BytesIO(data))
    image = _prepare_for_ocr(image, ImageOps, ImageFilter)
    try:
        return pytesseract.image_to_string(image, config="--psm 6")
    except pytesseract.TesseractNotFoundError:
        raise ExtractionError("Tesseract OCR is not installed. On macOS run: brew install tesseract")


def _prepare_for_ocr(image, ImageOps, ImageFilter):
    """Grayscale, upscale small scans, sharpen and binarize to help Tesseract."""
    image = ImageOps.exif_transpose(image).convert("L")
    if image.width < 1500:
        scale = 1500 / image.width
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    return image.point(lambda p: 255 if p > 150 else 0)
