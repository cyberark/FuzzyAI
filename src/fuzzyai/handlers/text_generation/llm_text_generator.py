from typing import Any

import torch
import torch.nn.functional as F  # For softmax function
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMTextGenerationHandler:

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = model.device

    def generate_text(self, prompt: str, max_length: int = 30, temperature: float = 0.000001, include_prompt: bool = False) -> Any:        
        """Generate text using the model given an input text."""
        # Ensure temperature is greater than 0 to avoid division by zero
        if temperature <= 0:
            raise ValueError("Temperature should be greater than 0")
        # Encode the input text
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # Initialize the generated text with the input text
        generated_ids = input_ids
        # Set the model to evaluation mode to disable dropout and enable evaluation-specific behavior in certain layers (like batch normalization) during inference time (not needed for generation)
        self.model.eval()
        # Disable gradient calculation to speed up the generation process and reduce memory consumption (not needed for inference)
        with torch.no_grad():
            # Generate tokens one by one
            outputs = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id, temperature=temperature)
            ids_to_decode = outputs[0] if include_prompt else outputs[0, input_ids.shape[-1] :]
            generated_text = self.tokenizer.decode(ids_to_decode, skip_special_tokens=True)

        return generated_text
