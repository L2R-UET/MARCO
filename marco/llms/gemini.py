from loguru import logger
import google.generativeai as genai
from typing import Any

from marco.llms.basellm import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model: str = 'gemini-2.0-flash-001', json_mode: bool = False, agent_context: str = None, *args, **kwargs):
        """Initialize the Gemini LLM.

        Args:
            `model` (`str`, optional): The name of the Gemini model. Defaults to `gemini-2.0-flash-001`.
            `json_mode` (`bool`, optional): Whether to use JSON mode. Defaults to `False`.
            `agent_context` (`str`, optional): The context of the agent using this LLM (e.g., 'Manager', 'Analyst'). Defaults to None.
        """
        logger.info(f"[API] Provider: gemini | Model: {model}")
        
        # Call parent constructor to initialize token tracking attributes
        super().__init__()
        
        self.model = model
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.max_tokens: int = kwargs.get('max_tokens', 2048)  # Increased from 1024 for JSON responses (recommendations need more tokens)
        self.temperature: float = kwargs.get('temperature', 0.7)  # Match OpenRouter default (was incorrectly 0)
        self.top_p: float = kwargs.get('top_p', 0.95)
        self.top_k: int = kwargs.get('top_k', 64)
        
        if 'gemini-1.5-pro' in model:
            self.max_context_length = 2097152  # 2M tokens for Gemini 1.5 Pro
        elif 'gemini-2.0-flash-001' in model:
            self.max_context_length = 1048576  # 1M tokens for Gemini 2.0 Flash
        elif 'gemini-2.0-flash-lite-001' in model:
            self.max_context_length = 1048576  # 1M tokens for Gemini 2.0 Flash Lite
        elif 'gemini-1.5-flash' in model:
            self.max_context_length = 1048576  # 1M tokens for Gemini 1.5 Flash
        else:
            self.max_context_length = 32768  # Default fallback
        
        # Configure generation parameters
        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_tokens,
        }
        
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
            logger.info("Using JSON mode for Gemini API.")
        
        # Note: Gemini API key should be configured globally via init_api() 
        # before creating any GeminiLLM instances. The init is handled in utils/init.py

        # The API key is already configured globally by init_gemini_api()
        self.client = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length
    
    def _make_api_request(self, prompt_text: str):
        """Make a single API request without retry logic.
        
        This wraps the Google SDK call so it can be used with execute_with_retry().
        
        Args:
            prompt_text: The prompt text to send
            
        Returns:
            Response object from Gemini
            
        Raises:
            Exception: For any API errors
        """
        logger.info(f"[API CALL] Provider: gemini | Model: {self.model}")
        logger.debug(f"[API CALL] Prompt type: {type(prompt_text)}, length: {len(str(prompt_text)) if prompt_text else 0}")
        
        # Ensure prompt_text is a string
        if not isinstance(prompt_text, str):
            logger.error(f"[API CALL] ERROR: prompt_text is not a string! Type: {type(prompt_text)}, Value: {prompt_text}")
            prompt_text = str(prompt_text)
        
        # Call generate_content with the prompt string
        # Gemini SDK accepts string directly or list of Content objects
        try:
            # Simply pass the string - SDK will handle the conversion
            response = self.client.generate_content(prompt_text)
            return response
        except Exception as e:
            logger.error(f"[API CALL] Gemini API error details:")
            logger.error(f"  - Error type: {type(e).__name__}")
            logger.error(f"  - Error message: {str(e)}")
            logger.error(f"  - Model: {self.model}")
            logger.error(f"  - Prompt length: {len(prompt_text) if isinstance(prompt_text, str) else 'N/A'}")
            raise
    
    def _get_retriable_errors(self) -> tuple:
        """Override to include Google API specific errors.
        
        Returns:
            Tuple of exception types that should trigger retry
        """
        base_errors = super()._get_retriable_errors()
        
        try:
            from google.api_core import exceptions as google_exceptions
            return base_errors + (
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.TooManyRequests,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ResourceExhausted,
            )
        except ImportError:
            # If google.api_core not available, just use base errors
            return base_errors
    
    def _should_retry_response(self, response: Any) -> bool:
        """Check if a Gemini response indicates a retriable error.
        
        Args:
            response: The Gemini response object
            
        Returns:
            True if should retry, False otherwise
        """
        # Gemini responses don't have status codes, but may have finish_reason indicating retryable state
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                # finish_reason: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
                finish_reason = getattr(candidate, 'finish_reason', None)
                # MAX_TOKENS and SAFETY might be transient, but not retriable at this level
                # Only system errors should trigger retries
                if finish_reason == 5:  # OTHER - potentially transient
                    return True
        
        return False

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the Gemini LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The Gemini LLM output.
        """
        try:
            # Log the prompt being sent to the API (skip for Analyst to reduce log spam)
            logger.info(f"LLM Prompt ({self.agent_context} → {self.model}):\n{prompt}")
            
            # Log estimated token usage for the prompt
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            logger.info(f"📊 Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare prompt based on JSON mode
            if self.json_mode:
                # For JSON mode, add instruction to the prompt
                messages[0]["content"] = f"{prompt}\n\nPlease respond with valid JSON only."
                actual_prompt = messages[0]["content"]
            else:
                actual_prompt = prompt
            
            # Make the API request with automatic retry for transient errors
            # Using base class retry mechanism that works for all LLM implementations
            response = self.execute_with_retry(
                self._make_api_request,
                prompt_text=actual_prompt
            )
            
            content = ""
            finish_reason_num = None
            
            # Check finish reason first
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    finish_reason_num = getattr(candidate, 'finish_reason', None)
                    if finish_reason_num == 2:  # MAX_TOKENS
                        logger.warning(f"Response truncated: finish_reason=2 (MAX_TOKENS). Current limit: {self.max_tokens} tokens.")
            
            # Try multiple extraction methods
            try:
                # Method 1: Try the quick accessor (works when response has valid parts)
                if hasattr(response, 'text') and response.text:
                    try:
                        content = response.text.strip()  # Only strip whitespace, preserve newlines
                        logger.debug(f"Successfully extracted {len(content)} chars using response.text")
                    except (ValueError, RuntimeError, AttributeError):
                        # .text accessor failed - will try other methods below
                        pass
                
                # Method 2: Extract from candidates/parts if quick accessor didn't work
                if not content and hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts_text = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    parts_text.append(part.text)
                            if parts_text:
                                content = ' '.join(parts_text).strip()
                                if content:
                                    logger.debug(f"Extracted {len(content)} chars from candidate.content.parts")
                                    break
                
                # Method 3: Try direct parts attribute
                if not content and hasattr(response, 'parts') and response.parts:
                    parts_text = []
                    for part in response.parts:
                        if hasattr(part, 'text') and part.text:
                            parts_text.append(part.text)
                    if parts_text:
                        content = ' '.join(parts_text).strip()
                        logger.debug(f"Extracted {len(content)} chars from response.parts")
                
                # Method 4: Try to extract from first candidate's text directly
                if not content and hasattr(response, 'candidates') and response.candidates:
                    try:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'text'):
                                candidate_text = candidate.text.strip()
                                if candidate_text:
                                    content = candidate_text
                                    logger.debug(f"Extracted {len(content)} chars from candidate.text")
                                    break
                    except (ValueError, RuntimeError, AttributeError):
                        pass
            
            except Exception as e:
                logger.debug(f"Error during text extraction methods: {type(e).__name__}: {str(e)[:100]}")
            
            # Log response status if no content extracted
            if not content:
                logger.warning(f"Could not extract text from Gemini response.")
                if hasattr(response, 'prompt_feedback'):
                    logger.warning(f"Prompt feedback: {response.prompt_feedback}")
                if hasattr(response, 'candidates') and response.candidates:
                    # Log finish reasons
                    reason_map = {1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
                    for i, candidate in enumerate(response.candidates):
                        reason = getattr(candidate, 'finish_reason', None)
                        reason_name = reason_map.get(reason, f"UNKNOWN({reason})")
                        logger.warning(f"Candidate {i}: finish_reason={reason_name}")
                        if reason == 2:
                            logger.error(f"⚠️ Response truncated due to MAX_TOKENS limit ({self.max_tokens}). Try increasing max_tokens or reducing prompt size.")
            
            if content:
                
                # Track token usage (extract from response metadata if available)
                input_tokens = None
                output_tokens = None
                
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                
                # Track the usage
                self.track_usage(
                    prompt,  # Use original prompt for consistency
                    content, 
                    input_tokens, 
                    output_tokens,
                    api_usage=({'prompt_tokens': input_tokens, 'completion_tokens': output_tokens} if input_tokens or output_tokens else {})
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
                logger.info(f"📈 Cumulative Usage ({self.agent_context}): {self.total_input_tokens + (input_tokens or 0)} total tokens across {self.api_calls + 1} calls")
                
                return content
            else:
                logger.warning("Empty response from Gemini API")
                return ""
        
        # Use base class error handling that works for all LLM implementations
        except Exception as e:
            # Use base class error handler for consistent error handling across all LLMs
            # This handles connection errors, timeouts, Google API errors, etc.
            return self.handle_api_error(e)
