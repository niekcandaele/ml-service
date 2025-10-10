"""Gemini API client for multimodal AI tasks.

Wraps the google-genai SDK to provide text generation, classification, and vision capabilities
using Google's Gemini models.
"""

import asyncio
import ipaddress
import json
import logging
import socket
from urllib.parse import urlparse

import httpx
from google import genai

from ml_service.config import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Google Gemini API.

    This client provides async methods for:
    - Text classification (is question or statement)
    - OCR/Vision (extract text from images)
    - Text completion (generate text from prompts)

    Attributes:
        client: Google GenAI client instance (from google-genai SDK)
        http_client: httpx AsyncClient for downloading images
    """

    def __init__(self):
        """Initialize Gemini API client.

        Uses GOOGLE_API_KEY from environment for authentication.
        Falls back to GEMINI_API_KEY if GOOGLE_API_KEY not set.

        Raises:
            ValueError: If API key not found in environment
            Exception: Original GenAI SDK exceptions bubble naturally for debugging
        """
        # google-genai SDK looks for GEMINI_API_KEY or GOOGLE_API_KEY
        api_key = settings.google_api_key

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini API access. "
                "Set it in .env file or environment."
            )

        logger.info("Initializing Gemini API client")

        # Initialize google-genai client
        # Let exceptions bubble naturally - fail-fast philosophy
        self.client = genai.Client(api_key=api_key)

        # HTTP client for downloading images from URLs
        self.http_client = httpx.AsyncClient(timeout=30.0)

        logger.info("Gemini API client initialized successfully")

    async def classify_question(self, text: str) -> tuple[bool, float]:
        """Classify text as question or statement using Gemini.

        Args:
            text: Input text to classify (1-10000 characters)

        Returns:
            Tuple of (is_question, confidence):
            - is_question: True if text is a question, False otherwise
            - confidence: Confidence score from 0.0 to 1.0

        Raises:
            Exception: Original GenAI SDK exceptions bubble naturally for debugging

        Note:
            Uses structured output prompt to get consistent JSON response.
            Gemini 2.0 Flash is fast and cost-effective for classification.
        """
        # Prompt engineering for consistent classification
        prompt = f"""Analyze if the following text is a question or a statement.

Text: "{text}"

Respond with ONLY valid JSON in this exact format:
{{"is_question": true/false, "confidence": 0.95}}

Where:
- is_question: true if the text is asking a question, false if it's a statement
- confidence: a number between 0.0 and 1.0 representing your confidence level

JSON response:"""

        # Generate response using async API
        # genai.Client provides .aio for async operations
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=settings.classification_model,
            contents=prompt,
        )

        # Parse response text as JSON
        response_text = response.text.strip()

        # Extract JSON from response (remove markdown code blocks if present)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(response_text)
            is_question = bool(result["is_question"])
            confidence = float(result["confidence"])

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            return (is_question, confidence)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse Gemini classification response: {response_text}")
            raise RuntimeError(f"Gemini returned invalid JSON: {e}") from e

    async def extract_text_from_image(self, image: str) -> tuple[list[str], str]:
        """Extract text from image using Gemini Vision.

        Args:
            image: Either a URL (https://...) or base64 data URI (data:image/...)

        Returns:
            Tuple of (literal_texts, description):
            - literal_texts: List of exact text strings found in the image
            - description: Natural language description of the image content

        Raises:
            ValueError: If image format is invalid
            Exception: Original GenAI SDK exceptions bubble naturally for debugging

        Note:
            Gemini Vision can extract text from images and provide descriptions.
            For URLs, downloads image and sends bytes to API.
        """
        # Prepare image content for Gemini
        image_part = await self._prepare_image_part(image)

        # Prompt for OCR with structured output
        prompt = """Analyze this image and extract all visible text.
Respond with ONLY valid JSON in this exact format:

{"literal_texts": ["text1", "text2", ...], "description": "description of the image"}

Where:
- literal_texts: array of all text strings visible in the image (empty array if no text)
- description: a brief description of what the image shows

JSON response:"""

        # Generate response with image and prompt
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=settings.ocr_model,
            contents=[prompt, image_part],
        )

        # Parse response text as JSON
        response_text = response.text.strip()

        # Extract JSON from response (remove markdown code blocks if present)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(response_text)
            literal_texts = result["literal_texts"]
            description = result["description"]

            if not isinstance(literal_texts, list):
                raise ValueError("literal_texts must be a list")
            if not isinstance(description, str):
                raise ValueError("description must be a string")

            return (literal_texts, description)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse Gemini OCR response: {response_text}")
            raise RuntimeError(f"Gemini returned invalid JSON: {e}") from e

    async def generate_completion(self, prompt: str, model: str) -> str:
        """Generate text completion using Gemini.

        Args:
            prompt: Input prompt for text generation (1-10000 characters)
            model: Gemini model to use (e.g. "gemini-2.0-flash-001")

        Returns:
            Generated text response from the model

        Raises:
            Exception: Original GenAI SDK exceptions bubble naturally for debugging

        Note:
            Simple text generation - no structured output required.
            Gemini 2.0 Flash is recommended for speed and cost.
        """
        # Generate completion using async API
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=prompt,
        )

        # Return generated text
        return response.text

    async def _prepare_image_part(self, image: str) -> genai.types.Part:
        """Prepare image data for Gemini Vision API.

        Args:
            image: Either URL (https://...) or base64 data URI (data:image/...)

        Returns:
            genai.types.Part with image data

        Raises:
            ValueError: If image format is invalid
            httpx.HTTPError: If image URL cannot be downloaded
        """
        # Check if it's a URL
        if image.startswith(("http://", "https://")):
            # Validate URL to prevent SSRF attacks
            parsed = urlparse(image)
            host = parsed.hostname
            if not host:
                raise ValueError("Invalid image URL: missing hostname")

            # Resolve hostname to IP addresses
            try:
                infos = socket.getaddrinfo(host, None)
            except socket.gaierror:
                raise ValueError("Invalid image URL: cannot resolve host")

            # Check each resolved IP address
            for _, _, _, _, sockaddr in infos:
                ip_str = sockaddr[0]
                ip = ipaddress.ip_address(ip_str)

                # Block disallowed IP ranges (SSRF protection)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                ):
                    raise ValueError(
                        f"Image URL resolves to a disallowed address: {ip_str}. "
                        "Cannot access private, loopback, link-local, reserved, or multicast IPs."
                    )

            # Download image from URL
            logger.debug(f"Downloading image from URL: {image[:100]}...")
            response = await self.http_client.get(image)
            response.raise_for_status()

            # Detect content type
            content_type = response.headers.get("content-type", "image/jpeg")
            mime_type = content_type.split(";")[0].strip()

            # Create Part from bytes
            return genai.types.Part.from_bytes(data=response.content, mime_type=mime_type)

        # Check if it's base64 data URI
        if image.startswith("data:image/"):
            # Extract mime type and base64 data
            header, encoded = image.split(",", 1) if "," in image else (image, "")
            mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"

            # Decode base64
            import base64

            image_bytes = base64.b64decode(encoded)

            # Create Part from bytes
            return genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        raise ValueError(
            "Image must be either a valid URL (http://... or https://...) "
            "or a base64 data URI (data:image/...)"
        )

    async def close(self):
        """Close HTTP client connections.

        Call this when shutting down the application to clean up resources.
        """
        await self.http_client.aclose()
        logger.info("Gemini client closed")
