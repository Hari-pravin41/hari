import os
import torch
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

class TrainingPipeline:
    """
    Modular training pipeline for fine-tuning LLMs on consumer hardware (RTX 3050 4GB).
    Supports JSONL datasets with 'instruction' and 'response' fields.
    """
    
    def __init__(self, base_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", output_dir="./models/my_lora"):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # LoRA Config: Optimized for 4GB VRAM
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16,           # Increased rank slightly for better learning
            lora_alpha=32, 
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Target all attention blocks
        )
        
    def format_prompt(self, example):
        """
        Formats instruction/response pairs into TinyLlama chat format.
        """
        instruction = example.get('instruction', '')
        response = example.get('response', '')
        
        # PROMPT FORMAT: <|user|>...</s><|assistant|>...</s>
        text = f"<|user|>\n{instruction}</s>\n<|assistant|>\n{response}</s>"
        return {"text": text}

    def train(self, dataset_path: str = "data/train.jsonl", epochs: int = 3):
        """
        Starts the fine-tuning process.
        """
        if not os.path.exists(dataset_path):
             print(f"[Error] Dataset not found at {dataset_path}")
             return

        print(f"[Training] Loading Base Model: {self.base_model_path}...")
        
        # Load Model in FP16 (approx 2.2GB VRAM)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False # Training requires cache=False
        )
        
        # IMPORTANT: Gradient Checkpointing enables training on 4GB VRAM
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        # Apply LoRA
        model = get_peft_model(model, self.peft_config)
        model.print_trainable_parameters()
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        tokenizer.pad_token = tokenizer.unk_token # Use unk as pad for Llama
        tokenizer.padding_side = "right" # Trainer expects right padding usually
        
        # Load & Preprocess Data
        print(f"[Training] Loading dataset: {dataset_path}")
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        formatted_dataset = dataset.map(self.format_prompt)
        
        def tokenize_function(examples):
            # Max length limited to 512 to save VRAM
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
        tokenized_datasets = formatted_dataset.map(tokenize_function, batched=True)
        
        # Trainer Config
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1, # Must be 1 for 4GB VRAM
            gradient_accumulation_steps=8, # Simulate batch size of 8
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            num_train_epochs=epochs,
            save_strategy="epoch",
            optim="adamw_torch",
            report_to=["none"] # Disable wandb/mlflow
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        print("[Training] Starting Training Loop...")
        print("Note: This may take a while. Monitor GPU temperatures.")
        try:
            trainer.train()
            print(f"[Training] Success! Saving adapters to {self.output_dir}")
            model.save_pretrained(self.output_dir)
        except Exception as e:
            print(f"[Training] Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "active":
        pipeline = TrainingPipeline()
        pipeline.train()
    else:
        print("Training module loaded. Run with 'python -m ai_engine.training active' to start.")
