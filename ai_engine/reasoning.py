from typing import List, Dict, Optional
import time

class ReasoningCore:
    """
    Orchestrates the 'Chain of Thought' process for the chatbot.
    Separates context retrieval, reasoning, and final answer generation.
    """

    def __init__(self, config=None):
        self.config = config if config else AIConfig()
        self.history = []
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """
        Lazy loads the CUSTOM USER MODEL ('RepairGPT').
        Strictly prevents usage of any pre-trained external models.
        """
        if self.model:
            return

        print("[Reasoning] Initializing YOUR Custom AI Model ('RepairGPT')...")
        
        try:
            # PURE CUSTOM ENGINE ONLY
            try:
                from .chat_scratch import NumpyChat
            except ImportError:
                from chat_scratch import NumpyChat
            
            import os
            
            model_dir = "./models/token_model"
            weights_file = os.path.join(model_dir, "weights.npz")
            
            if os.path.exists(weights_file):
                self.model = NumpyChat(model_dir=model_dir)
                print(f"[Reasoning] Custom Token-AI 'RepairGPT' loaded from {weights_file}")
            else:
                print(f"[Reasoning] No trained model found at {weights_file}.")
                raise FileNotFoundError("Your custom Token AI is not trained yet! Please run 'train_token_gru.py'")
                
        except Exception as e:
            print(f"[Reasoning] FATAL ERROR: {e}")
            raise e

    def unload(self):
        """
        Unloads model to free VRAM.
        """
        if self.model:
            import gc
            import torch
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[Reasoning] Unloaded model to free VRAM.")

    def plan_response(self, query: str, context_docs: List[str]) -> Dict[str, str]:
        """
        Phase 1: Generates a plan/reasoning trace before answering.
        """
        # For now, we still use a heuristic plan, but we could use the LLM to generate this too.
        # Generate a dynamic plan based on the query
        steps = ["Analyze user intent"]
        
        if "image" in query.lower() or "picture" in query.lower():
            steps.append("Process visual data")
            
        if "code" in query.lower() or "error" in query.lower() or "function" in query.lower():
            steps.append("Review technical context")
            steps.append("Synthesize coding solution")
        else:
            steps.append("Retrieve knowledge base")
            steps.append("Formulate explanation")
            
        steps.append("Check for clarity and formatting")
        
        plan = {
            "intent": "dynamic_analysis",
            "steps": steps
        }
        return plan

    def generate_answer(self, query: str, reasoning_plan: Dict) -> str:
        """
        Phase 2: Synthesizes the final answer using the LLM.
        """
        self._load_model()
        
        # 1. Custom Numpy RNN
        if hasattr(self.model, 'generate') and not type(self.model).__name__.startswith('LLM'): 
            try:
                # Prepare input prompt
                prompt = f"User: {query}\nAI:"
                # Generate directly (NumpyChat handles tokenization/detokenization)
                response_full = self.model.generate(prompt, length=150)
                # Extract just the AI part
                if "AI:" in response_full:
                    response = response_full.split("AI:", 1)[1]
                else:
                    response = response_full
                return response.replace("User:", "").strip()
            except Exception as e:
                return f"Error: {e}"

        # 2. CTransformers (Professional GGUF)
        if self.model.__module__.startswith('ctransformers'):
             try:
                system_header = "You are a helpful AI assistant."
                prompt = f"<|user|>\n{query}</s>\n<|assistant|>\n"
                
                # CTransformers call is simpler:
                response = self.model(
                    prompt, 
                    max_new_tokens=256, 
                    temperature=0.7, 
                    top_p=0.9,
                    stop=["</s>", "<|user|>"]
                )
                return response.strip()
             except Exception as e:
                return f"GGUF Error: {e}"

        # 3. Standard PyTorch (Fallback)
        try: 
            context_str = "\n".join([f"- {doc}" for doc in reasoning_plan.get("context", [])])
            system_header = "Instruct: You are an expert AI Assistant specialized in Computer Science and Artificial Intelligence."
            
            prompt = (
                f"{system_header}\n"
                f"Context: {context_str}\n"
                f"User: {query}\n"
                f"Output:"
            )
                    
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
            
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=250,
                do_sample=True, 
                temperature=0.3,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
             return f"Error: {e}"

    def run_pipeline(self, query: str, context_docs: List[str]) -> Dict:
        """
        Executes the full reasoning loop.
        """
        start_time = time.time()
        
        # 0. Web Search Check (Heuristic or Explicit)
        web_context = ""
        if "search" in query.lower() or "internet" in query.lower() or "latest" in query.lower() or "price" in query.lower():
            print("[Reasoning] Web Search Triggered.")
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                
                web_context = "\n\n**Web Search Results:**\n"
                for r in results:
                    web_context += f"- [{r['title']}]({r['href']}): {r['body']}\n"
                
                print(f"[Reasoning] Found {len(results)} web results.")
            except Exception as e:
                print(f"[Reasoning] Web Search Failed: {e}")
                web_context = "\n[Web Search Failed]"

        # 1. Thought
        reasoning_trace = self.plan_response(query, context_docs)
        
        # Merge Web Context into reasoning context
        final_context = context_docs + [web_context] if web_context else context_docs
        reasoning_trace["context"] = final_context
        
        # 2. Action (Generation)
        answer = self.generate_answer(query, reasoning_trace)
        
        # 3. Update History
        self.history.append((query, answer))
        # Keep only last 5 turns to prevent overflowing context window
        if len(self.history) > 5:
            self.history.pop(0)
        
        # 3. Reflection (Logging)
        duration = time.time() - start_time
        
        return {
            "answer": answer,
            "reasoning": reasoning_trace,
            "meta": {"latency_ms": round(duration * 1000, 2), "web_search": bool(web_context)}
        }
    
if __name__ == "__main__":
    from model_config import AIConfig
    core = ReasoningCore()
    # Mock context
    result = core.run_pipeline("What is the capital of France?", ["France is a country in Europe.", "Paris is the capital of France."])
    print(result)
