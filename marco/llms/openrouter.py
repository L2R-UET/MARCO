import requests
import json
import time
from loguru import logger
from typing import Any, Dict

from marco.llms.basellm import BaseLLM

class OpenRouterLLM(BaseLLM):
    def __init__(self, model: str = 'google/gemini-2.0-flash-001', api_key: str = '', json_mode: bool = False, agent_context: str = None, *args, **kwargs):
        logger.info(f"[API] Provider: openrouter | Model: {model}")
        """Initialize the OpenRouter LLM.

        Args:
            `model` (`str`, optional): The name of the model on OpenRouter. Defaults to `mistralai/mistral-7b-instruct`.
            `api_key` (`str`): The API key for OpenRouter. If empty, will try to get from environment or config.
            `json_mode` (`bool`, optional): Whether to use JSON mode. Defaults to `False`.
            `agent_context` (`str`, optional): The context of the agent using this LLM (e.g., 'Manager', 'Analyst'). Defaults to None.
        """

        super().__init__()

        self.model = model
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.max_tokens: int = kwargs.get('max_tokens', 1024)
        self.temperature: float = kwargs.get('temperature', 0.7)
        self.top_p: float = kwargs.get('top_p', 0.95)
        
        if api_key:
            self.api_key = api_key
        else:
            import os
            self.api_key = os.getenv('OPENROUTER_API_KEY', '')
            if not self.api_key:
                try:
                    from marco.utils import read_json
                    api_config = read_json('config/api-config.json')
                    
                    # Try new multi-provider structure first
                    if 'providers' in api_config and 'openrouter' in api_config['providers']:
                        self.api_key = api_config['providers']['openrouter'].get('api_key', '')
                    
                    # Fallback to legacy structure
                    if not self.api_key:
                        if api_config.get('provider') == 'openrouter':
                            self.api_key = api_config.get('api_key', '')
                        if not self.api_key:
                            self.api_key = api_config.get('openrouter_api_key', '')
                except:
                    pass
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. Please set it in config/api-config.json or as OPENROUTER_API_KEY environment variable.")
        else:
            logger.info(f"OpenRouter API key loaded: {self.api_key[:6]}...{self.api_key[-4:]}")
        
        model_context_lengths = {
            'mistralai/mistral-7b-instruct': 32768,
            'meta-llama/llama-3.1-8b-instruct': 131072,
            'anthropic/claude-3-haiku': 200000,
            'openai/gpt-3.5-turbo': 16384,
            'openai/gpt-4': 8192,
            'openai/gpt-4-turbo': 128000,
            'openai/gpt-oss-20b:free': 8192,
            'google/gemini-pro': 32768,
            'google/gemini-2.0-flash-001': 1048576,
            'google/gemini-2.0-flash-lite-001': 1048576,
            'google/gemini-1.5-pro': 2097152,
            'google/gemini-1.5-flash': 1048576,
            'cohere/command-r': 128000,
            'microsoft/wizardlm-2-8x22b': 65536,
        }
        
        self.max_context_length = model_context_lengths.get(model, 32768)  # Default fallback
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            from marco.utils import read_json
            api_config = read_json('config/api-config.json')
            if 'providers' in api_config and 'openrouter' in api_config['providers']:
                provider_cfg = api_config['providers']['openrouter']
                if provider_cfg.get('base_url'):
                    self.base_url = provider_cfg['base_url']
                else:
                    raise ValueError("Missing 'base_url' for OpenRouter provider in config/api-config.json!")
            else:
                raise ValueError("Missing OpenRouter provider config in config/api-config.json!")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenRouter base_url from config: {e}")
        
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay_base = kwargs.get('retry_delay_base', 1)
        
        logger.info(f"Initialized OpenRouter LLM with model: {model}, max_retries: {self.max_retries}")

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length
    
    def _make_api_request(
        self,
        payload: Dict[str, Any],
        timeout: int = 60
    ) -> requests.Response:
        """Make a single API request without retry logic.
        
        This is the core request method that will be wrapped by execute_with_retry().
        
        Args:
            payload: The request payload
            timeout: Request timeout in seconds
            
        Returns:
            Response object
            
        Raises:
            requests.exceptions.RequestException: For any request errors
        """
        logger.info(f"[API CALL] Provider: openrouter | Model: {self.model}")
        return requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=timeout,
            stream=False
        )

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the OpenRouter LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The OpenRouter LLM output.
        """
        try:
            # Log the prompt being sent to the API (skip for Analyst to reduce log spam)
            if self.agent_context != "Analyst":
                logger.info(f"LLM Prompt ({self.agent_context} → {self.model}):\n{prompt}")
            
            # Log estimated token usage for the prompt
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": min(self.max_tokens, 4096),  # Cap at 4096 to prevent very long responses
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}
                messages[0]["content"] = f"{prompt}\n\nPlease respond with valid JSON only."
            
            response = self.execute_with_retry(
                self._make_api_request,
                payload=payload,
                timeout=60
            )
            
            # Log response summary for debugging
            logger.debug(f"OpenRouter API response: status={response.status_code}, length={len(response.text)} chars")
            
            # Check response size - if too large, it might be malformed
            if len(response.text) > 500000:  # 500KB limit
                logger.warning(f"Very large response from OpenRouter: {len(response.text)} chars")
            
            if response.status_code == 200:
                # Check content type before parsing JSON
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    logger.warning(f"Unexpected content-type from OpenRouter: {content_type}")
                
                # Enhanced JSON parsing with better error handling
                try:
                    result = response.json()
                except json.JSONDecodeError as json_err:
                    # Log the response details for debugging
                    response_text = response.text
                    logger.error(f"JSON decode error in OpenRouter response:")
                    logger.error(f"  Error: {json_err}")
                    logger.error(f"  Response length: {len(response_text)} chars")
                    logger.error(f"  Response preview (first 500 chars): {response_text[:500]}")
                    logger.error(f"  Response preview (last 500 chars): {response_text[-500:]}")
                    
                    # Try to extract content manually if possible
                    import re
                    content_match = re.search(r'"content":\s*"([^"]*)"', response_text)
                    if content_match:
                        logger.warning("Attempting to extract content manually from malformed JSON")
                        content = content_match.group(1)
                        
                        # Track usage with estimates since we can't parse the JSON
                        self.track_usage(
                            prompt, 
                            content, 
                            None,  # Will use estimation
                            None,  # Will use estimation
                            api_usage={'error': 'json_parse_failed'}
                        )
                        
                        return content.strip()
                    else:
                        return f"Error: Invalid JSON response (could not extract content)"
                
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    
                    input_tokens = None
                    output_tokens = None
                    total_tokens = None
                    
                    if 'usage' in result:
                        usage = result['usage']
                        input_tokens = usage.get('prompt_tokens')
                        output_tokens = usage.get('completion_tokens')
                        total_tokens = usage.get('total_tokens')
                        
                        # Log token usage summary for debugging (reduced verbosity)
                        logger.debug(f"OpenRouter API usage: {input_tokens}+{output_tokens}={total_tokens} tokens")
                        
                        # Validate token counts are consistent
                        if input_tokens and output_tokens and total_tokens:
                            expected_total = input_tokens + output_tokens
                            if total_tokens != expected_total:
                                logger.warning(f"Token count mismatch: API reported total={total_tokens}, calculated={expected_total}")
                    else:
                        logger.warning("No usage information in OpenRouter API response - falling back to estimation")
                    
                    # Track the usage
                    self.track_usage(
                        prompt, 
                        content, 
                        input_tokens, 
                        output_tokens,
                        api_usage=result.get('usage', {})  # Store full usage info
                    )
                    
                    # Log the response from the API
                    logger.info(f"LLM Response ({self.agent_context} → {self.model}):\n{content.strip()}")
                    
                    # Log actual token usage after API response
                    if input_tokens and output_tokens and total_tokens:
                        logger.info(f"📊 Token Usage ({self.agent_context}): {input_tokens} prompt + {output_tokens} completion = {total_tokens} total tokens")
                    else:
                        # Fallback to estimation if no API usage info
                        estimated_input = self.estimate_tokens(prompt)
                        estimated_output = self.estimate_tokens(content.strip())
                        estimated_total = estimated_input + estimated_output
                        logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_input} prompt + ~{estimated_output} completion = ~{estimated_total} total tokens (estimated)")
                    
                    # Log cumulative usage
                    logger.info(f"📈 Cumulative Usage ({self.agent_context}): {self.total_input_tokens + (input_tokens or 0)} total tokens across {self.api_calls + 1} calls")
                    
                    return content.strip()
                else:
                    logger.warning("No choices found in OpenRouter API response")
                    return ""
            else:
                logger.error(f"OpenRouter API request failed with status {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code}"
        
        # Use base class error handling that works for all LLM implementations
        except json.JSONDecodeError as e:
            # JSON errors are specific to response parsing, not the base request
            logger.error(f"❌ JSON decode error: {e}")
            return f"Error: INVALID_JSON - Failed to parse API response"
        
        except Exception as e:
            # Use base class error handler for consistent error handling across all LLMs
            return self.handle_api_error(e)
