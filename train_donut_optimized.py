# train_donut_optimized.py - Memory Optimized for Azure VM
import os
import torch
import gc
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, DonutProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image

# -------------------
# 1. Config - Optimized for Low Memory
# -------------------
DATASET_DIR = "dataset_donut"
MODEL_NAME = "naver-clova-ix/donut-base"  # Consider using "donut-small" if still having issues
OUTPUT_DIR = "./donut-finetuned-optimized"

# Simple memory monitoring
def print_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
    except ImportError:
        print("Memory monitoring not available (psutil not installed)")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print_memory_usage()

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Running on CPU - using ultra-low memory settings")

# -------------------
# 2. Load datasets with memory optimization
# -------------------
print("Loading datasets...")
train_dataset = load_dataset("json", data_files=os.path.join(DATASET_DIR, "train/metadata.jsonl"))["train"]
val_dataset = load_dataset("json", data_files=os.path.join(DATASET_DIR, "val/metadata.jsonl"))["train"]

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print_memory_usage()

# -------------------
# 3. Load model & processor with memory optimization
# -------------------
print("Loading model and processor...")

# Force garbage collection before loading model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Load processor first
processor = DonutProcessor.from_pretrained(MODEL_NAME, use_fast=False)

# Load model with memory optimizations
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use float32 for CPU
    low_cpu_mem_usage=True,
    device_map="auto" if torch.cuda.is_available() else None
)

# Fix Donut model configuration
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Move model to device
model = model.to(device)
print(f"Model moved to {device}")
print(f"Decoder start token ID: {model.config.decoder_start_token_id}")
print(f"Pad token ID: {model.config.pad_token_id}")
print(f"EOS token ID: {model.config.eos_token_id}")
print_memory_usage()

# -------------------
# 4. Ultra Memory-Efficient Dataset
# -------------------
class UltraMemoryEfficientDonutDataset:
    def __init__(self, dataset, base_path, processor, device, max_length=256):
        self.dataset = dataset
        self.base_path = base_path
        self.processor = processor
        self.device = device
        self.max_length = max_length  # Reduced from 512
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        try:
            # Load image with error handling
            img_path = os.path.join(self.base_path, example["image"])
            image = Image.open(img_path).convert("RGB")
            
            # Resize image to reduce memory usage (optional)
            # image = image.resize((800, 600))  # Smaller size for memory efficiency
            
            # Process image with processor
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

            # Tokenize text label with shorter max length
            text = example["text"]
            input_ids = self.processor.tokenizer(
                text,
                max_length=self.max_length,  # Reduced max length
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()
            
            # Clear image from memory
            del image
            
            return {
                "pixel_values": pixel_values,
                "labels": input_ids
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample if there's an error
            dummy_pixel_values = torch.zeros((3, 1024, 1024))  # Standard Donut input size
            dummy_labels = torch.zeros(self.max_length, dtype=torch.long)
            return {
                "pixel_values": dummy_pixel_values,
                "labels": dummy_labels
            }

# Create ultra memory-efficient datasets
train_dataset = UltraMemoryEfficientDonutDataset(
    train_dataset, 
    os.path.join(DATASET_DIR, "train"), 
    processor, 
    device,
    max_length=256  # Reduced max length
)
val_dataset = UltraMemoryEfficientDonutDataset(
    val_dataset, 
    os.path.join(DATASET_DIR, "val"), 
    processor, 
    device,
    max_length=256  # Reduced max length
)

# Memory-efficient collator
def memory_efficient_collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Clear individual items from memory
    for item in batch:
        del item['pixel_values']
        del item['labels']
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

# -------------------
# 5. Ultra-Low Memory Training Configuration
# -------------------
# Ultra-conservative settings for Azure VM
batch_size = 1
gradient_accumulation_steps = 8  # Reduced from 16
max_length = 256  # Reduced from 512

print(f"Using batch size: {batch_size}, gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Max sequence length: {max_length}")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=100,  # More frequent saves
    eval_steps=100,
    logging_steps=25,  # More frequent logging
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=3,  # Reduced epochs
    learning_rate=2e-5,  # Lower learning rate
    save_total_limit=1,  # Keep only 1 checkpoint
    predict_with_generate=True,
    remove_unused_columns=False,
    fp16=False,  # Disable mixed precision for CPU
    dataloader_pin_memory=False,
    dataloader_num_workers=0,  # No multiprocessing
    report_to="none",
    # Memory optimizations
    gradient_checkpointing=True,
    optim="adamw_torch",
    warmup_steps=25,  # Reduced warmup
    # CPU specific optimizations
    dataloader_prefetch_factor=None,
    dataloader_drop_last=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Additional memory optimizations
    max_grad_norm=1.0,
    weight_decay=0.01,
    logging_dir="./logs",
    # Disable features that use extra memory
    ddp_find_unused_parameters=False,
    dataloader_drop_last=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    data_collator=memory_efficient_collate_fn,
)

# -------------------
# 6. Train with Memory Monitoring
# -------------------
if __name__ == "__main__":
    print("Starting training...")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print("Training with ultra-low memory settings!")
    print_memory_usage()
    
    try:
        # Clear all caches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer.train()
        
        # Save model
        print("Saving model...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print("✅ Training complete! Model saved at:", OUTPUT_DIR)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Try reducing the model size or increasing VM memory")
        print_memory_usage()
