"""
Local LLM implementation using HuggingFace transformers.

This module provides local LLM inference using PyTorch and transformers.
"""

from typing import Any, Optional
import asyncio
import time

from torchagentic.llms.base import BaseLLM, LLMConfig


class LocalLLM(BaseLLM):
    """
    Local LLM using HuggingFace transformers and PyTorch.
    
    Supports running models locally without API calls.
    Requires the transformers library to be installed.
    
    Attributes:
        model_id: HuggingFace model ID
        device: Device to run model on (cuda, cpu, mps)
        model: Loaded model instance
        tokenizer: Tokenizer instance
    """
    
    def __init__(
        self,
        model_id: str = "microsoft/phi-2",
        device: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        load_model: bool = True,
    ):
        super().__init__(config=config or LLMConfig(model=model_id))
        
        self.model_id = model_id
        self.device = device or self._get_default_device()
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        if load_model:
            self._load_model()
    
    def _get_default_device(self) -> str:
        """Get the default available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
                low_cpu_mem_usage=True,
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._is_loaded = True
            
        except ImportError as e:
            raise ImportError(
                "transformers library is required for LocalLLM. "
                "Install with: pip install transformers torch accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {e}") from e
    
    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format messages into a prompt string."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "tool":
                formatted.append(f"Tool Result: {content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)
    
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate a response from the local model.
        
        Note: Tool calling is simulated through prompt engineering
        as local models may not support native function calling.
        """
        if not self._is_loaded:
            self._load_model()
        
        start_time = time.time()
        
        try:
            import torch
            import re
            
            # Format messages
            prompt = self._format_messages(messages)
            
            # Add tools context if provided
            if tools:
                tools_str = "Available tools:\n"
                for tool in tools:
                    func = tool.get("function", {})
                    tools_str += f"- {func.get('name', 'unknown')}: {func.get('description', '')}\n"
                prompt = tools_str + "\n" + prompt
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_tokens - 100,
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            temp = temperature or self.config.temperature
            max_new = max_tokens or self.config.max_tokens
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=temp,
                    top_p=self.config.top_p,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
            # Calculate token usage
            prompt_tokens = inputs["input_ids"].shape[1]
            completion_tokens = len(generated_ids)
            
            # Try to parse tool calls from response
            tool_calls = self._parse_tool_calls(generated_text, tools)
            
            time_taken = time.time() - start_time
            
            self._update_stats(
                tokens=prompt_tokens + completion_tokens,
                time_taken=time_taken,
            )
            
            return {
                "content": generated_text.strip(),
                "tool_calls": tool_calls,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            
        except Exception as e:
            self._update_stats(error=True)
            raise RuntimeError(f"Generation failed: {e}") from e
    
    def _parse_tool_calls(
        self,
        text: str,
        tools: Optional[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Try to parse tool calls from generated text."""
        if not tools:
            return []
        
        tool_calls = []
        
        # Look for patterns like: tool_name(arg1=value1, arg2=value2)
        for tool in tools:
            func = tool.get("function", {})
            tool_name = func.get("name", "")
            
            # Pattern: tool_name(...)
            pattern = rf"{re.escape(tool_name)}\(([^)]*)\)"
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Parse arguments (simplified)
                args = {}
                for arg in match.split(","):
                    if "=" in arg:
                        key, value = arg.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        args[key] = value
                
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "name": tool_name,
                    "arguments": args,
                })
        
        return tool_calls
    
    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ):
        """
        Generate streaming response.
        
        Yields chunks of the response as they are generated.
        """
        if not self._is_loaded:
            self._load_model()
        
        import torch
        
        # Format messages
        prompt = self._format_messages(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with streaming
        with torch.no_grad():
            generated = inputs["input_ids"]
            
            for i in range(self.config.max_tokens):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                new_token = outputs[0, -1]
                
                if new_token == self.tokenizer.eos_token_id:
                    break
                
                chunk = self.tokenizer.decode([new_token], skip_special_tokens=True)
                yield chunk
                
                # Update inputs for next iteration
                inputs = {
                    "input_ids": torch.cat([generated, new_token.unsqueeze(0).unsqueeze(0)], dim=1),
                    "attention_mask": torch.ones_like(outputs[0]).unsqueeze(0),
                }
                generated = inputs["input_ids"][0]
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using the model.
        
        Note: This uses the model's hidden states as embeddings.
        For better embeddings, use a dedicated embedding model.
        """
        if not self._is_loaded:
            self._load_model()
        
        import torch
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get last hidden state
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
                
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                sum_embeddings = token_embeddings.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1, keepdim=True)
                mean_embedding = sum_embeddings / (sum_mask + 1e-9)
                
                embeddings.append(mean_embedding[0].tolist())
        
        return embeddings
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "type": "LocalLLM",
            "model_id": self.model_id,
            "device": self.device,
            "config": self.config.to_dict(),
            "is_loaded": self._is_loaded,
            "stats": self.get_stats(),
        }
    
    def __repr__(self) -> str:
        return (
            f"LocalLLM(model={self.model_id}, "
            f"device={self.device}, loaded={self._is_loaded})"
        )


class MockLLM(BaseLLM):
    """
    Mock LLM for testing purposes.
    
    Returns predefined responses without making actual API calls.
    """
    
    def __init__(
        self,
        response: str = "This is a mock response.",
        config: Optional[LLMConfig] = None,
    ):
        super().__init__(config=config)
        self.default_response = response
        self.call_history: list[dict[str, Any]] = []
    
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Return mock response."""
        start_time = time.time()
        
        self.call_history.append({
            "messages": messages,
            "tools": tools,
            "kwargs": kwargs,
        })
        
        # Check for tool calls in messages to simulate tool response
        tool_calls = []
        if tools and len(messages) > 0:
            last_msg = messages[-1].get("content", "").lower()
            if "calculate" in last_msg or "search" in last_msg:
                tool_calls.append({
                    "id": "mock_call_1",
                    "name": tools[0].get("function", {}).get("name", "mock_tool"),
                    "arguments": {"query": last_msg},
                })
        
        time_taken = time.time() - start_time
        
        return {
            "content": self.default_response,
            "tool_calls": tool_calls,
            "token_usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(self.default_response.split()),
                "total_tokens": 100,
            },
        }
    
    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ):
        """Stream mock response."""
        words = self.default_response.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1] * 384 for _ in texts]
    
    def set_response(self, response: str) -> None:
        """Set the mock response."""
        self.default_response = response
    
    def reset(self) -> None:
        """Reset call history."""
        self.call_history = []
