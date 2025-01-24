from typing import List, Literal, Dict, Any, Union, Optional
from pydantic import BaseModel
from jinja2 import Template
import re
import fitz
import os
from tqdm import tqdm
from .utils import ImageData
from tenacity import retry, stop_after_attempt, wait_exponential
from .constants import SUPPORTED_MODELS
import logging

logger = logging.getLogger(__name__)


class ImageDescription(BaseModel):
    """Model Schema for image description."""

    text_detected: Literal["Yes", "No"]
    tables_detected: Literal["Yes", "No"]
    images_detected: Literal["Yes", "No"]
    latex_equations_detected: Literal["Yes", "No"]
    extracted_text: str
    confidence_score_text: float


class UnsupportedModelError(BaseException):
    """Custom exception for unsupported model names"""

    pass


class LLMError(BaseException):
    """Custom exception for Vision LLM errors"""

    pass
        
class LLM:
    # Load prompts at class level
    try:
        from importlib.resources import files

        _first_pass_prompt = Template(
            files("vision_parse").joinpath("image_analysis.j2").read_text()
        )
        _md_prompt_template = Template(
            files("vision_parse").joinpath("markdown_prompt.j2").read_text()
        )
        _refinement_prompt = Template(
            files("vision_parse").joinpath("refinement_prompt.j2").read_text()
        )
    except Exception as e:
        raise FileNotFoundError(f"Failed to load prompt files: {str(e)}")

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        openai_config: Optional[Dict] = None,
        custom_prompt: Optional[str] = None,
        enable_concurrency: bool = False,
        device: Optional[Literal["cuda", "mps"]] = None,
        num_workers: int = 1,
        **kwargs: Any,
    ):
        """Initialize LLM with configuration.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the model provider
            temperature: Temperature for text generation
            top_p: Top p for text generation
            openai_config: Configuration for OpenAI API
            custom_prompt: Custom prompt to use
            enable_concurrency: Whether to enable concurrent processing
            device: Device to use for processing
            num_workers: Number of workers for concurrent processing
            **kwargs: Additional arguments to pass to the model
        """
        self.model_name = model_name
        self.api_key = api_key
        self.openai_config = openai_config or {}
        self.temperature = temperature
        self.top_p = top_p
        self.custom_prompt = custom_prompt
        self.enable_concurrency = enable_concurrency
        self.device = device
        self.num_workers = num_workers
        self.kwargs = kwargs
        
        # Set default configs
        self.ollama_config = kwargs.get('ollama_config', {})
        self.gemini_config = kwargs.get('gemini_config', {})
        self.image_mode = kwargs.get('image_mode')
        self.detailed_extraction = kwargs.get('detailed_extraction', True)

        self.provider = self._get_provider_name(model_name)
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the LLM client."""
        if self.provider == "ollama":
            import ollama

            try:
                ollama.show(self.model_name)
            except ollama.ResponseError as e:
                if e.status_code == 404:
                    current_digest, bars = "", {}
                    for progress in ollama.pull(self.model_name, stream=True):
                        digest = progress.get("digest", "")
                        if digest != current_digest and current_digest in bars:
                            bars[current_digest].close()

                        if not digest:
                            logger.info(progress.get("status"))
                            continue

                        if digest not in bars and (total := progress.get("total")):
                            bars[digest] = tqdm(
                                total=total,
                                desc=f"pulling {digest[7:19]}",
                                unit="B",
                                unit_scale=True,
                            )

                        if completed := progress.get("completed"):
                            bars[digest].update(completed - bars[digest].n)

                        current_digest = digest
            except Exception as e:
                raise LLMError(
                    f"Unable to download {self.model_name} from Ollama: {str(e)}"
                )

            try:
                os.environ["OLLAMA_KEEP_ALIVE"] = str(
                    self.ollama_config.get("OLLAMA_KEEP_ALIVE", -1)
                )
                if self.enable_concurrency:
                    self.aclient = ollama.AsyncClient(
                        host=self.ollama_config.get(
                            "OLLAMA_HOST", "http://localhost:11434"
                        ),
                        timeout=self.ollama_config.get("OLLAMA_REQUEST_TIMEOUT", 240.0),
                    )
                    if self.device == "cuda":
                        os.environ["OLLAMA_NUM_GPU"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_GPU", self.num_workers // 2
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 8
                            )
                        )
                        os.environ["OLLAMA_GPU_LAYERS"] = str(
                            self.ollama_config.get("OLLAMA_GPU_LAYERS", "all")
                        )
                    elif self.device == "mps":
                        os.environ["OLLAMA_NUM_GPU"] = str(
                            self.ollama_config.get("OLLAMA_NUM_GPU", 1)
                        )
                        os.environ["OLLAMA_NUM_THREAD"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_THREAD", self.num_workers
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 8
                            )
                        )
                    else:
                        os.environ["OLLAMA_NUM_THREAD"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_THREAD", self.num_workers
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 10
                            )
                        )
                else:
                    self.client = ollama.Client(
                        host=self.ollama_config.get(
                            "OLLAMA_HOST", "http://localhost:11434"
                        ),
                        timeout=self.ollama_config.get("OLLAMA_REQUEST_TIMEOUT", 240.0),
                    )
            except Exception as e:
                raise LLMError(f"Unable to initialize Ollama client: {str(e)}")

        elif self.provider == "openai" or self.provider == "deepseek":
            #  support azure openai
            if self.provider == "openai" and self.openai_config.get(
                "AZURE_OPENAI_API_KEY"
            ):
                try:
                    import openai
                    from openai import AzureOpenAI, AsyncAzureOpenAI
                except ImportError:
                    raise ImportError(
                        "OpenAI is not installed. Please install it using pip install 'vision-parse[openai]'."
                    )

                try:
                    azure_subscription_key = self.openai_config.get(
                        "AZURE_OPENAI_API_KEY"
                    )
                    azure_endpoint_url = self.openai_config.get("AZURE_ENDPOINT_URL")

                    if not azure_endpoint_url or not azure_subscription_key:
                        raise LLMError(
                            "Set `AZURE_ENDPOINT_URL` and `AZURE_OPENAI_API_KEY` environment variables in `openai_config` parameter"
                        )

                    if self.openai_config.get("AZURE_DEPLOYMENT_NAME"):
                        self.model_name = self.openai_config.get(
                            "AZURE_DEPLOYMENT_NAME"
                        )

                    api_version = self.openai_config.get(
                        "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
                    )

                    # Initialize Azure OpenAI client with key-based authentication
                    if self.enable_concurrency:
                        self.aclient = AsyncAzureOpenAI(
                            azure_endpoint=azure_endpoint_url,
                            api_key=azure_subscription_key,
                            api_version=api_version,
                        )
                    else:
                        self.client = AzureOpenAI(
                            azure_endpoint=azure_endpoint_url,
                            api_key=azure_subscription_key,
                            api_version=api_version,
                        )

                except openai.OpenAIError as e:
                    raise LLMError(
                        f"Unable to initialize Azure OpenAI client: {str(e)}"
                    )

            else:
                try:
                    import openai
                except ImportError:
                    raise ImportError(
                        "OpenAI is not installed. Please install it using pip install 'vision-parse[openai]'."
                    )
                try:
                    if self.enable_concurrency:
                        import httpx
                        http_client = httpx.AsyncClient(
                                limits=httpx.Limits(
                                    max_connections=100,
                                    max_keepalive_connections=100,
                                )
                            )
                        self.aclient = openai.AsyncOpenAI(
                            http_client=http_client,
                            api_key=self.api_key,
                            base_url=(
                                self.openai_config.get("OPENAI_BASE_URL", None)
                                if self.provider == "openai"
                                else "https://api.deepseek.com"
                            ),
                            max_retries=self.openai_config.get("OPENAI_MAX_RETRIES", 3),
                            timeout=self.openai_config.get("OPENAI_TIMEOUT", 240.0),
                            default_headers=self.openai_config.get(
                                "OPENAI_DEFAULT_HEADERS", None
                            ),
                        )
                    else:
                        self.client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=(
                                self.openai_config.get("OPENAI_BASE_URL", None)
                                if self.provider == "openai"
                                else "https://api.deepseek.com"
                            ),
                            max_retries=self.openai_config.get("OPENAI_MAX_RETRIES", 3),
                            timeout=self.openai_config.get("OPENAI_TIMEOUT", 240.0),
                            default_headers=self.openai_config.get(
                                "OPENAI_DEFAULT_HEADERS", None
                            ),
                        )
                except openai.OpenAIError as e:
                    raise LLMError(f"Unable to initialize OpenAI client: {str(e)}")

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "Gemini is not installed. Please install it using pip install 'vision-parse[gemini]'."
                )

            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(model_name=self.model_name)
                self.generation_config = genai.GenerationConfig
            except Exception as e:
                raise LLMError(f"Unable to initialize Gemini client: {str(e)}")

    def _get_provider_name(self, model_name: str) -> str:
        """Get the provider name for a given model name."""
        try:
            return SUPPORTED_MODELS[model_name]
        except KeyError:
            supported_models = ", ".join(
                f"'{model}' from {provider}"
                for model, provider in SUPPORTED_MODELS.items()
            )
            raise UnsupportedModelError(
                f"Model '{model_name}' is not supported. "
                f"Supported models are: {supported_models}"
            )

    async def _get_response(
        self, base64_encoded: str, prompt_or_messages: Union[str, List[Dict]], structured: bool = False, is_refinement: bool = False
    ) -> Any:
        """Get response from the model."""
        if self.provider == "ollama":
            return await self._ollama(base64_encoded, prompt_or_messages, structured, is_refinement)
        elif self.provider == "openai":
            return await self._openai(base64_encoded, prompt_or_messages, structured, is_refinement)
        elif self.provider == "gemini":
            return await self._gemini(base64_encoded, prompt_or_messages, structured, is_refinement)

    async def generate_markdown(
        self, base64_encoded: str, pix: fitz.Pixmap, page_number: int
    ) -> Any:
        """Generate markdown formatted text from a base64-encoded image using appropriate model provider."""
        # First pass - Initial extraction
        first_pass_prompt = self._first_pass_prompt.render(
            custom_prompt=self.custom_prompt
        )
        first_pass_result = await self._get_response(
            base64_encoded,
            first_pass_prompt,
            structured=False
        )

        # Second pass - Refinement
        refinement_prompt = self._refinement_prompt.render(
            custom_prompt=self.custom_prompt
        )
        
        # Add the first pass result to the messages for the second pass
        if self.provider == "ollama":
            messages = [
                {
                    "role": "user",
                    "content": f"{refinement_prompt}\n\nPreviously extracted text:\n{first_pass_result}",
                    "images": [base64_encoded],
                }
            ]
        else:  # openai, gemini, etc.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{refinement_prompt}\n\nPreviously extracted text:\n{first_pass_result}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_encoded}"
                            },
                        },
                    ],
                }
            ]

        # Second pass with refinement
        markdown_content = await self._get_response(
            base64_encoded,
            messages,
            structured=False,
            is_refinement=True  # New parameter to handle refinement pass
        )

        return markdown_content

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _ollama(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Process base64-encoded image through Ollama vision models."""
        try:
            if self.enable_concurrency:
                response = await self.aclient.chat(
                    model=self.model_name,
                    format=ImageDescription.model_json_schema() if structured else None,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64_encoded],
                        }
                    ],
                    options={
                        "temperature": 0.0 if structured else self.temperature,
                        "top_p": 0.4 if structured else self.top_p,
                        **self.kwargs,
                    },
                    keep_alive=-1,
                )
            else:
                response = self.client.chat(
                    model=self.model_name,
                    format=ImageDescription.model_json_schema() if structured else None,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64_encoded],
                        }
                    ],
                    options={
                        "temperature": 0.0 if structured else self.temperature,
                        "top_p": 0.4 if structured else self.top_p,
                        **self.kwargs,
                    },
                    keep_alive=-1,
                )

            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```",
                r"\1",
                response["message"]["content"],
                flags=re.DOTALL,
            )
        except Exception as e:
            raise LLMError(f"Ollama Model processing failed: {str(e)}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _openai(
        self, base64_encoded: str, prompt_or_messages: Union[str, List[Dict]], structured: bool = False, is_refinement: bool = False
    ) -> Any:
        """Process base64-encoded image through OpenAI vision models."""
        try:
            # If prompt_or_messages is a string, construct the messages
            if isinstance(prompt_or_messages, str):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_or_messages},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_encoded}"
                                },
                            },
                        ],
                    }
                ]
            else:
                messages = prompt_or_messages

            if self.enable_concurrency:
                response = await self.aclient.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1 if is_refinement else self.temperature,
                    top_p= self.top_p,
                    stream=False,
                    **self.kwargs,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1 if is_refinement else self.temperature,
                    top_p= self.top_p,
                    stream=False,
                    **self.kwargs,
                )

            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```",
                r"\1",
                response.choices[0].message.content,
                flags=re.DOTALL,
            )
        except Exception as e:
            raise LLMError(f"OpenAI Model processing failed: {str(e)}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _gemini(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Process base64-encoded image through Gemini vision models."""
        try:
            if self.enable_concurrency:
                response = await self.client.generate_content_async(
                    [{"mime_type": "image/png", "data": base64_encoded}, prompt],
                    generation_config=self.generation_config(
                        response_mime_type="application/json" if structured else None,
                        response_schema=ImageDescription if structured else None,
                        temperature=0.0 if structured else self.temperature,
                        top_p=0.4 if structured else self.top_p,
                        **self.kwargs,
                    ),
                )
            else:
                response = self.client.generate_content(
                    [{"mime_type": "image/png", "data": base64_encoded}, prompt],
                    generation_config=self.generation_config(
                        response_mime_type="application/json" if structured else None,
                        response_schema=ImageDescription if structured else None,
                        temperature=0.0 if structured else self.temperature,
                        top_p=0.4 if structured else self.top_p,
                        **self.kwargs,
                    ),
                )

            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```", r"\1", response.text, flags=re.DOTALL
            )
        except Exception as e:
            raise LLMError(f"Gemini Model processing failed: {str(e)}")
