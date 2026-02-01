import torch
import logging
import platform

# Configure isolated logger
logger = logging.getLogger("ai_engine.config")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class AIConfig:
    """
    Central configuration and hardware verification for the AI Engine.
    Ensures safe execution on available hardware (CPU/GPU).
    """
    
    def __init__(self):
        self.device = self._detect_device()
        self.max_seq_length = 512
        self.embedding_model = "all-MiniLM-L6-v2"
        self.llm_model_path = "./models/llama-2-7b-chat-quantized.gguf" # Placeholder
        
    def _detect_device(self) -> str:
        """
        Safely identifies the best available hardware accelerator.
        """
        try:
            # Check availability safely
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                logger.info(f"GPU Detected: {props.name}")
                return "cuda"
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple Silicon (MPS) Detected.")
                return "mps"
        except Exception as e:
            logger.warning(f"Device detection error: {e}")
            
        logger.info("Defaulting to CPU Mode.")
        return "cpu"

    def get_llm_config(self):
        """
        Returns the optimized configuration for loading the LLM.
        Use this to prevent OOM errors and lagging.
        """
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Check if we can use quantization (requires bitsandbytes installed)
        try:
            from transformers import BitsAndBytesConfig
            import torch
            
            if self.device == "cuda":
                logger.info("Configuring 4-bit quantization for GPU efficiency...")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                logger.warning("GPU not available for quantization. Loading in standard mode (might be slow).")
        except ImportError:
            logger.warning("bitsandbytes not installed. Quantization disabled.")
        except Exception as e:
            logger.warning(f"Could not configure quantization: {e}")
            
        return model_kwargs

    def log_capabilities(self):
        """
        Public method to log system readiness without starting workloads.
        """
        logger.info(f"AI Engine Configured for: {self.device.upper()}")
        logger.info(f"OS: {platform.system()} {platform.release()}")
        try:
            logger.info(f"PyTorch Version: {torch.__version__}")
        except:
            logger.info("PyTorch Version: Not Available (Running in Compatibility Mode)")

# Singleton instance
config = AIConfig()

if __name__ == "__main__":
    config.log_capabilities()
