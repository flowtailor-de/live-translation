"""
Translation module using NLLB-200.
Translates German text to Farsi.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Translator:
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
        self.source_lang = source_lang
        self.target_lang = target_lang
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
        
        logger.info(f"Translator will use device: {device}")
    
    def load_model(self) -> None:
        """Load the translation model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading NLLB model: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.source_lang,
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
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


if __name__ == "__main__":
    # Test translator
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Translator...")
    translator = Translator()
    translator.load_model()
    
    test_texts = [
        "Guten Morgen, willkommen in unserer Gemeinde.",
        "Heute werden wir über Liebe und Hoffnung sprechen.",
        "Gott liebt dich.",
    ]
    
    print("\nTranslating German to Farsi:")
    for text in test_texts:
        translated = translator.translate(text)
        print(f"DE: {text}")
        print(f"FA: {translated}")
        print()
