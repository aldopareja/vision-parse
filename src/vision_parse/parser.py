import fitz  # PyMuPDF library for PDF processing
from pathlib import Path
from typing import Optional, List, Dict, Union, Literal, Any
from tqdm import tqdm
import base64
from pydantic import BaseModel
import asyncio
from .utils import get_device_config
from .llm import LLM
import nest_asyncio
import logging
import warnings

logger = logging.getLogger(__name__)
nest_asyncio.apply()


class PDFPageConfig(BaseModel):
    """Configuration settings for PDF page conversion."""

    dpi: int = 400  # Resolution for PDF to image conversion
    color_space: str = "RGB"  # Color mode for image output
    include_annotations: bool = True  # Include PDF annotations in conversion
    preserve_transparency: bool = False  # Control alpha channel in output


class UnsupportedFileError(BaseException):
    """Custom exception for handling unsupported file errors."""

    pass


class VisionParserError(BaseException):
    """Custom exception for handling Markdown Parser errors."""

    pass


class VisionParser:
    """Convert PDF pages to base64-encoded images and then extract text from the images in markdown format."""

    def __init__(
        self,
        page_config: Optional[PDFPageConfig] = None,
        model_name: str = "llama3.2-vision:11b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        openai_config: Optional[Dict] = None,
        custom_prompt: Optional[str] = None,
        enable_concurrency: bool = False,
        num_workers: int = 4,
        **kwargs: Any,
    ):
        """Initialize parser with PDFPageConfig and LLM configuration."""
        self.page_config = page_config or PDFPageConfig()
        self.enable_concurrency = enable_concurrency
        self.num_workers = num_workers

        self.llm = LLM(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            openai_config=openai_config,
            custom_prompt=custom_prompt,
            enable_concurrency=enable_concurrency,
            **kwargs,
        )

    def _calculate_matrix(self, page: fitz.Page) -> fitz.Matrix:
        """Calculate transformation matrix for page conversion."""
        # Calculate zoom factor based on target DPI
        zoom = self.page_config.dpi / 72
        matrix = fitz.Matrix(zoom * 2, zoom * 2)

        # Handle page rotation if present
        if page.rotation != 0:
            matrix.prerotate(page.rotation)

        return matrix

    async def _convert_page(self, page: fitz.Page, page_number: int) -> str:
        """Convert a single PDF page into base64-encoded PNG and extract markdown formatted text."""
        try:
            matrix = self._calculate_matrix(page)

            # Create high-quality image from PDF page
            pix = page.get_pixmap(
                matrix=matrix,
                alpha=self.page_config.preserve_transparency,
                colorspace=self.page_config.color_space,
                annots=self.page_config.include_annotations,
            )

            # Convert image to base64 for LLM processing
            base64_encoded = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            return await self.llm.generate_markdown(base64_encoded, pix, page_number)

        except Exception as e:
            raise VisionParserError(
                f"Failed to convert page {page_number + 1} to base64-encoded PNG: {str(e)}"
            )
        finally:
            # Clean up pixmap to free memory
            if pix is not None:
                pix = None

    async def _convert_pages_batch(self, pages: List[fitz.Page], start_idx: int):
        """Process a batch of PDF pages concurrently."""
        try:
            tasks = []
            for i, page in enumerate(pages):
                tasks.append(self._convert_page(page, start_idx + i))
            return await asyncio.gather(*tasks)
        finally:
            await asyncio.sleep(0.5)

    def convert_pdf(self, pdf_path: Union[str, Path]) -> List[str]:
        """Convert all pages in the given PDF file to markdown text."""
        pdf_path = Path(pdf_path)
        converted_pages = []

        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise UnsupportedFileError(f"File is not a PDF: {pdf_path}")

        try:
            with fitz.open(pdf_path) as pdf_document:
                total_pages = pdf_document.page_count

                with tqdm(
                    total=total_pages,
                    desc="Converting pages in PDF file into markdown format",
                ) as pbar:
                    if self.enable_concurrency:
                        # Process pages in batches based on num_workers
                        for i in range(0, total_pages, self.num_workers):
                            batch_size = min(self.num_workers, total_pages - i)
                            # Extract only required pages for the batch
                            batch_pages = [
                                pdf_document[j] for j in range(i, i + batch_size)
                            ]
                            batch_results = asyncio.run(
                                self._convert_pages_batch(batch_pages, i)
                            )
                            converted_pages.extend(batch_results)
                            pbar.update(len(batch_results))
                    else:
                        for page_number in range(total_pages):
                            # For non-concurrent processing, still need to run async code
                            text = asyncio.run(
                                self._convert_page(
                                    pdf_document[page_number], page_number
                                )
                            )
                            converted_pages.append(text)
                            pbar.update(1)

                return converted_pages

        except Exception as e:
            raise VisionParserError(
                f"Failed to convert PDF file into markdown content: {str(e)}"
            )

'''
from vision_parse import VisionParser

parser = VisionParser(
    model_name="OpenGVLab/InternVL2_5-78B",
    image_mode="url",
    api_key="your-key",
    detailed_extraction=False, # Set to True for more detailed extraction
    enable_concurrency=True,
    temperature=0.1,
    top_p=1.0,
    frequency_penalty=0.1,
    max_tokens=4096,
    openai_config={
        "OPENAI_BASE_URL": "http://0.0.0.0:23333/v1",  # Specify your custom endpoint URL here
        "OPENAI_MAX_RETRIES": 3,  # Optional: you can also configure other parameters
        "OPENAI_TIMEOUT": 480.0    # Optional: default timeout is 240.0
    },
    num_workers=4,
)

# pdf_path = "/Users/aldo/Projects/BoA_poc/boa_docs/CashPro_Remote_Deposit_Admin_Guide.pdf"
# pdf_path = "/root/CashPro_Remote_Deposit_Admin_Guide.pdf"
pdf_path = "/root/CashPro_Remote_Deposit_Admin_Guide_28_33.pdf"
# pdf_path = "./Rewards Money Market Savings .pdf"
markdown_pages = parser.convert_pdf(pdf_path)
# Concatenate all markdown pages and save to file
output_path = pdf_path.replace('.pdf', '.md')
with open(output_path, 'w') as f:
    f.write('\n\n'.join(markdown_pages))
print(f"Saved markdown to: {output_path}")

from IPython import embed; embed()
'''