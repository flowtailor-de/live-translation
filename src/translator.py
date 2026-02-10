"""
Translation module supporting multiple backends.
- NLLB-200: Lightweight, fast translation
- TranslateGemma: State-of-the-art translation quality from Google
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TranslatorBase(ABC):
    """Abstract base class for all translators."""
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the translation model."""
        pass
    
    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate text from source to target language."""
        pass


class NLLBTranslator(TranslatorBase):
    """Translates text using NLLB-200 model."""
    
    # NLLB language codes
    LANGUAGE_CODES = {
        "german": "deu_Latn",
        "de": "deu_Latn",
        "farsi": "pes_Arab",
        "persian": "pes_Arab",
        "fa": "pes_Arab",
    }
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        source_lang: str = "deu_Latn",
        target_lang: str = "pes_Arab",
        device: str = "auto",
        max_length: int = 512,
    ):
        """
        Initialize translator.
        
        Args:
            model_name: HuggingFace model name
            source_lang: Source language code (NLLB format)
            target_lang: Target language code (NLLB format)
            device: Device to use (auto, cpu, cuda, mps)
            max_length: Maximum output length
        """
        self.model_name = model_name
        self.source_lang = self._resolve_lang_code(source_lang)
        self.target_lang = self._resolve_lang_code(target_lang)
        self.max_length = max_length
        
        # Determine device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self.model = None
        self.tokenizer = None
        
        logger.info(f"NLLBTranslator will use device: {device}")
    
    def _resolve_lang_code(self, code: str) -> str:
        """Resolve language code to NLLB format."""
        return self.LANGUAGE_CODES.get(code.lower(), code)
    
    def load_model(self) -> None:
        """Load the translation model."""
        if self.model is not None:
            return
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        logger.info(f"Loading NLLB model: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.source_lang,
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        
        self.model.eval()
        
        logger.info("NLLB model loaded")
    
    def translate(self, text: str) -> str:
        """
        Translate text from German to Farsi.
        
        Args:
            text: German text to translate
            
        Returns:
            Translated Farsi text
        """
        if not text or not text.strip():
            return ""
        
        if self.model is None:
            self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        # Get target language token ID
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=5,
                early_stopping=True,
            )
        
        # Decode
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.debug(f"Translated: '{text[:30]}...' -> '{translated[:30]}...'")
        
        return translated
    
    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of German texts
            
        Returns:
            List of translated Farsi texts
        """
        if not texts:
            return []
        
        if self.model is None:
            self.load_model()
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=5,
                early_stopping=True,
            )
        
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return translations


class TranslateGemmaTranslator(TranslatorBase):
    """
    Translates text using Google's TranslateGemma model.
    
    TranslateGemma is a state-of-the-art translation model from Google
    supporting 55 languages with excellent quality.
    """
    
    # ISO 639-1 language codes (TranslateGemma uses these)
    LANGUAGE_CODES = {
        "german": "de",
        "deu_Latn": "de",
        "farsi": "fa",
        "persian": "fa",
        "pes_Arab": "fa",
    }
    
    def __init__(
        self,
        model_name: str = "google/translategemma-12b-it",
        source_lang: str = "de",
        target_lang: str = "fa",
        device: str = "auto",
        max_new_tokens: int = 256,
    ):
        """
        Initialize TranslateGemma translator.
        
        Args:
            model_name: HuggingFace model name (e.g., google/translategemma-12b-it)
            source_lang: Source language code (ISO 639-1, e.g., 'de')
            target_lang: Target language code (ISO 639-1, e.g., 'fa')
            device: Device to use (auto, cpu, cuda, mps)
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.source_lang = self._resolve_lang_code(source_lang)
        self.target_lang = self._resolve_lang_code(target_lang)
        self.max_new_tokens = max_new_tokens
        
        # Determine device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self.model = None
        self.processor = None
        
        logger.info(f"TranslateGemmaTranslator will use device: {device}")
    
    def _resolve_lang_code(self, code: str) -> str:
        """Resolve language code to ISO 639-1 format."""
        return self.LANGUAGE_CODES.get(code, code)
    
    def load_model(self) -> None:
        """Load the TranslateGemma model."""
        if self.model is not None:
            return
        
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        logger.info(f"Loading TranslateGemma model: {self.model_name}...")
        logger.info("This may take a few minutes on first run...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Load model with appropriate dtype for device
        if self.device == "mps":
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto",
            )
        elif self.device == "cuda":
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            ).to(self.device)
        
        self.model.eval()
        
        logger.info("TranslateGemma model loaded successfully")
    
    def translate(self, text: str) -> str:
        """
        Translate text using TranslateGemma.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        if self.model is None:
            self.load_model()
        
        # Build the message in TranslateGemma's expected format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.source_lang,
                        "target_lang_code": self.target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Move to device
        if self.device == "mps":
            inputs = {k: v.to(self.device, dtype=torch.float32) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        input_len = inputs['input_ids'].shape[1]
        
        # Generate translation
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        # Decode only the generated tokens
        generated_tokens = outputs[0][input_len:]
        translated = self.processor.decode(generated_tokens, skip_special_tokens=True)
        
        logger.debug(f"Translated: '{text[:30]}...' -> '{translated[:30]}...'")
        
        return translated.strip()


# Keep original Translator class as alias for backwards compatibility
Translator = NLLBTranslator


class TranslateGemmaMLXTranslator(TranslatorBase):
    """
    Translates text using TranslateGemma on Apple Silicon via MLX.
    
    Uses mlx-lm for highly optimized inference on M-series chips.
    """
    
    # ISO 639-1 language codes
    LANGUAGE_CODES = {
        "german": "de",
        "deu_Latn": "de",
        "farsi": "fa",
        "persian": "fa",
        "pes_Arab": "fa",
    }
    
    def __init__(
        self,
        model_name: str = "mlx-community/translategemma-4b-it-4bit",
        source_lang: str = "de",
        target_lang: str = "fa",
        max_new_tokens: int = 256,
        **kwargs
    ):
        self.model_name = model_name
        self.source_lang = self._resolve_lang_code(source_lang)
        self.target_lang = self._resolve_lang_code(target_lang)
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.tokenizer = None
        
        # Check for Apple Silicon
        if not torch.backends.mps.is_available():
            logger.warning("MLX translator requested but MPS not available! Performance may be poor.")
            
        logger.info(f"TranslateGemmaMLXTranslator initialized for {self.model_name}")
    
    def _resolve_lang_code(self, code: str) -> str:
        return self.LANGUAGE_CODES.get(code, code)
    
    def load_model(self) -> None:
        if self.model is not None:
            return
            
        logger.info(f"Loading MLX model: {self.model_name}...")
        try:
            from mlx_lm import load
            self.model, self.tokenizer = load(self.model_name)
            logger.info("MLX model loaded successfully")
        except ImportError:
            logger.error("mlx-lm not installed! Please run: pip install mlx-lm")
            raise
    
    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
            
        if self.model is None:
            self.load_model()
        
        from mlx_lm import generate
        from src.mlx_lock import mlx_lock
        
        # Build prompt using chat template logic
        # TranslateGemma format: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        # We can use the tokenizer's apply_chat_template if available
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.source_lang,
                        "target_lang_code": self.target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback manual template if tokenizer doesn't support structured content objects
            # Note: TranslateGemma expects structured input, so simple string concatenation might fail 
            # if the tokenizer isn't the specialized one. 
            # However, mlx-community models usually ship with the correct tokenizer_config.json
            logger.warning(f"Failed to apply chat template: {e}. Trying fallback construction.")
            # Verify if simple string prompt works or if we need to manually construct the XML-like tags
            # TranslateGemma is specialized. Let's assume the tokenizer works as it's the HF one loaded by MLX.
            raise e

        # Use shared MLX lock to prevent GPU conflicts with transcriber
        with mlx_lock:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                verbose=False
            )
        
        result = response.strip()
        
        # Clean up Gemma special tokens
        clean_result = result.replace("<end_of_turn>", "").replace("<start_of_turn>", "").replace("model\n", "").strip()
        
        logger.debug(f"Translated (MLX): '{text[:30]}...' -> '{clean_result[:30]}...'")
        return clean_result


def create_translator(
    backend: str = "nllb",
    **kwargs
) -> TranslatorBase:
    """
    Factory function to create a translator based on the specified backend.
    
    Args:
        backend: Translation backend ('nllb', 'translategemma', 'translategemma-mlx')
        **kwargs: Additional arguments passed to the translator
        
    Returns:
        Translator instance
    """
    backend = backend.lower()
    
    if backend == "translategemma-mlx":
        return TranslateGemmaMLXTranslator(**kwargs)
    elif backend == "translategemma":
        return TranslateGemmaTranslator(**kwargs)
    elif backend == "nllb":
        return NLLBTranslator(**kwargs)
    else:
        logger.warning(f"Unknown backend '{backend}', falling back to NLLB")
        return NLLBTranslator(**kwargs)


if __name__ == "__main__":
    # Test translators
    import sys
    logging.basicConfig(level=logging.INFO)
    
    backend = sys.argv[1] if len(sys.argv) > 1 else "nllb"
    
    print(f"Testing {backend.upper()} Translator...")
    
    translator = create_translator(backend=backend)
    translator.load_model()
    
    test_texts = [
        "Guten Morgen, willkommen in unserer Gemeinde.",
        "Heute werden wir über Liebe und Hoffnung sprechen.",
        "Gott liebt dich.",
    ]
    
    print(f"\nTranslating German to Farsi using {backend.upper()}:")
    for text in test_texts:
        translated = translator.translate(text)
        print(f"DE: {text}")
        print(f"FA: {translated}")
        print()

