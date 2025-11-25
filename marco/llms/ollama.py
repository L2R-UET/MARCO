import requests
import json
from loguru import logger
from typing import Any, Dict

from marco.llms.basellm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model: str = 'llama3.2:1b', base_url: str = 'http://localhost:11434', json_mode: bool = False, agent_context: str = None, *args, **kwargs):
        logger.info(f"[API] Provider: ollama | Model: {model}")
        """Initialize the Ollama LLM.

        Args:
            `model` (`str`, optional): The name of the model in Ollama. Defaults to `llama3.2`.
            `base_url` (`str`, optional): The base URL for the Ollama API. Defaults to `http://localhost:11434`.
            `json_mode` (`bool`, optional): Whether to use JSON mode. Defaults to `False`.
            `agent_context` (`str`, optional): The context of the agent using this LLM (e.g., 'Manager', 'Analyst'). Defaults to None.
        """
        # Call parent constructor to initialize token tracking attributes
        super().__init__()
        
        self.model = model
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.max_tokens: int = kwargs.get('max_tokens', 1024)
        self.temperature: float = kwargs.get('temperature', 0.7)
        self.top_p: float = kwargs.get('top_p', 0.9)
        self.top_k: int = kwargs.get('top_k', 40)
        self.timeout: int = kwargs.get('timeout', 300)  # 5 minutes for local models
        
        try:
            from marco.utils import read_json
            api_config = read_json('config/api-config.json')
            if 'providers' in api_config and 'ollama' in api_config['providers']:
                provider_cfg = api_config['providers']['ollama']
                if provider_cfg.get('base_url'):
                    self.base_url = provider_cfg['base_url'].rstrip('/')
                else:
                    raise ValueError("Missing 'base_url' for Ollama provider in config/api-config.json!")
            else:
                raise ValueError("Missing Ollama provider config in config/api-config.json!")
        except Exception as e:
            raise RuntimeError(f"Failed to load Ollama base_url from config: {e}")
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        
        # Default context lengths for common Ollama models
        model_context_lengths = {
            'llama3.2': 131072,
            'llama3.2:1b': 131072,
            'llama3.2:3b': 131072,
            'llama3.1': 131072,
            'llama3.1:8b': 131072,
            'llama3.1:70b': 131072,
            'llama3': 8192,
            'llama2': 4096,
            'llama2:7b': 4096,
            'llama2:13b': 4096,
            'llama2:70b': 4096,
            'codellama': 16384,
            'codellama:7b': 16384,
            'codellama:13b': 16384,
            'codellama:34b': 16384,
            'mistral': 32768,
            'mistral:7b': 32768,
            'mixtral': 32768,
            'mixtral:8x7b': 32768,
            'gemma': 8192,
            'gemma:2b': 8192,
            'gemma:7b': 8192,
            'phi': 2048,
            'phi3': 4096,
            'qwen': 32768,
            'qwen2': 32768,
            'neural-chat': 4096,
            'starling-lm': 8192,
            'yi': 4096,
            'dolphin-mixtral': 32768,
            'orca-mini': 4096,
            'vicuna': 2048,
            'wizard-vicuna': 2048,
        }
        
        self.max_context_length = model_context_lengths.get(model, 4096)  # Default fallback
        
        self.headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized Ollama LLM with model: {model} at {base_url}")

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length
    
    def _make_api_request(
        self,
        payload: Dict[str, Any],
        timeout: int = 300
    ) -> requests.Response:
        """Make a single API request without retry logic.
        
        This is the core request method that will be wrapped by execute_with_retry().
        
        Args:
            payload: The request payload
            timeout: Request timeout in seconds (default 5 minutes for local models)
            
        Returns:
            Response object
            
        Raises:
            requests.exceptions.RequestException: For any request errors
        """
        logger.info(f"[API CALL] Provider: ollama | Model: {self.model}")
        return requests.post(
            self.generate_url,
            headers=self.headers,
            json=payload,
            timeout=timeout
        )

    def _check_ollama_server(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            # Check exact match or partial match for model name
            if self.model in model_names:
                return True
            
            # Check for partial matches (e.g., 'llama3.2' matches 'llama3.2:latest')
            for model_name in model_names:
                if self.model in model_name or model_name.startswith(self.model):
                    logger.info(f"Found model match: {model_name} for requested {self.model}")
                    return True
            
            logger.warning(f"Model {self.model} not found in Ollama. Available models: {model_names}")
            logger.info(f"You can pull the model by running: ollama pull {self.model}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            logger.info("Make sure Ollama is installed and running. Visit https://ollama.ai for installation instructions.")
            return False

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the Ollama LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The Ollama LLM output.
        """
        try:
            if not self._check_ollama_server():
                error_msg = f"Ollama server not available or model {self.model} not found"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Log the prompt being sent to the API (skip for Analyst to reduce log spam)
            if self.agent_context != "Analyst":
                logger.info(f"LLM Prompt ({self.agent_context} → {self.model}):\n{prompt}")
            
            # Log estimated token usage for the prompt
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            
            # Prepare the request payload for Ollama generate API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens
                }
            }
            
            if self.json_mode:
                payload["format"] = "json"
                payload["prompt"] = f"{prompt}\n\nPlease respond with valid JSON only."
            
            # Make the API request with automatic retry for transient errors
            # Using base class retry mechanism that works for all LLM implementations
            response = self.execute_with_retry(
                self._make_api_request,
                payload=payload,
                timeout=self.timeout  # Use configurable timeout
            )
            
            # Log response summary for debugging
            logger.debug(f"Ollama API response: status={response.status_code}, length={len(response.text)} chars")
            
            # Check response size - if too large, it might be malformed
            if len(response.text) > 500000:  # 500KB limit
                logger.warning(f"Very large response from Ollama: {len(response.text)} chars")
            
            if response.status_code == 200:
                # Check content type before parsing JSON
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    logger.warning(f"Unexpected content-type from Ollama: {content_type}")
                
                # Enhanced JSON parsing with better error handling
                try:
                    result = response.json()
                except json.JSONDecodeError as json_err:
                    # Log the response details for debugging
                    response_text = response.text
                    logger.error(f"JSON decode error in Ollama response:")
                    logger.error(f"  Error: {json_err}")
                    logger.error(f"  Response length: {len(response_text)} chars")
                    logger.error(f"  Response preview (first 500 chars): {response_text[:500]}")
                    logger.error(f"  Response preview (last 500 chars): {response_text[-500:]}")
                    
                    # Try to extract content manually if possible
                    import re
                    content_match = re.search(r'"response":\s*"([^"]*)"', response_text)
                    if content_match:
                        logger.warning("Attempting to extract content manually from malformed JSON")
                        content = content_match.group(1)
                        
                        # Track usage with estimates since we can't parse the JSON
                        self.track_usage(
                            prompt, 
                            content, 
                            None,  # Will use estimation
                            None  # Will use estimation
                        )
                        
                        logger.info(f"LLM Response ({self.agent_context} → {self.model}):\n{content}")
                        return content
                    
                    return f"Error: INVALID_JSON - Failed to parse Ollama response"
                
                if 'response' in result:
                    content = result['response'].strip()
                    
                    input_tokens = None
                    output_tokens = None
                    
                    if 'prompt_eval_count' in result:
                        input_tokens = result['prompt_eval_count']
                    if 'eval_count' in result:
                        output_tokens = result['eval_count']
                    
                    # Track usage
                    self.track_usage(
                        prompt, 
                        content, 
                        input_tokens, 
                        output_tokens,
                        api_usage=result
                    )
                    
                    # Log the response from the API
                    logger.info(f"LLM Response ({self.agent_context} → {self.model}):\n{content}")
                    
                    # Log token usage after API response
                    if input_tokens and output_tokens:
                        total_tokens = input_tokens + output_tokens
                        logger.info(f"📊 Token Usage ({self.agent_context}): {input_tokens} prompt + {output_tokens} completion = {total_tokens} total tokens")
                    else:
                        # Fallback to estimation if no API usage info
                        estimated_input = self.estimate_tokens(prompt)
                        estimated_output = self.estimate_tokens(content)
                        estimated_total = estimated_input + estimated_output
                        logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_input} prompt + ~{estimated_output} completion = ~{estimated_total} total tokens (estimated)")
                    
                    # Log cumulative usage
                    logger.info(f"� Cumulative Usage ({self.agent_context}): {self.total_input_tokens + self.total_output_tokens} total tokens across {self.api_calls} calls")
                    
                    return content
                else:
                    logger.error(f"No 'response' field in Ollama response: {result}")
                    return "Error: Invalid response format from Ollama"
                    
            else:
                error_msg = f"Ollama API request failed with status {response.status_code}"
                try:
                    error_detail = response.json()
                    if 'error' in error_detail:
                        error_msg += f": {error_detail['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                
                logger.error(error_msg)
                # Use base class error handler for consistent error classification
                return self.handle_api_error(Exception(error_msg))
        
        # Use base class error handling that works for all LLM implementations
        except json.JSONDecodeError as e:
            # JSON errors are specific to response parsing, not the base request
            logger.error(f"❌ JSON decode error: {e}")
            return f"Error: INVALID_JSON - Failed to parse API response"
        
        except Exception as e:
            # Use base class error handler for consistent error handling across all LLMs
            # This handles connection errors, timeouts, etc.
            return self.handle_api_error(e)

    def list_models(self) -> list:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                logger.error(f"Failed to list Ollama models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    def pull_model(self, model: str = None) -> bool:
        if model is None:
            model = self.model
            
        try:
            payload = {"name": model}
            response = requests.post(
                f"{self.base_url}/api/pull",
                headers=self.headers,
                json=payload,
                timeout=1800  # 30 minute timeout for model pulling
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model {model}")
                return True
            else:
                logger.error(f"Failed to pull model {model}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False
