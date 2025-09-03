# train_donut_ultra_minimal.py - Ultra Minimal for Azure VM
import os
import torch
import gc
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, DonutProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image

# -------------------
# 1. Config - Ultra Minimal
# -------------------
DATASET_DIR = "dataset_donut"
MODEL_NAME = "naver-clova-ix/donut-base"  # We'll try to use a smaller subset
OUTPUT_DIR = "./donut-finetuned-ultra-minimal"

# Simple memory monitoring
def print_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
    except ImportError:
        print("Memory monitoring not available")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print_memory_usage()

# -------------------
# 2. Load datasets - Use only a subset for faster training
# -------------------
print("Loading datasets...")
train_dataset = load_dataset("json", data_files=os.path.join(DATASET_DIR, "train/metadata.jsonl"))["train"]
val_dataset = load_dataset("json", data_files=os.path.join(DATASET_DIR, "val/metadata.jsonl"))["train"]

# Use only first 50 samples for faster training
train_dataset = train_dataset.select(range(min(50, len(train_dataset))))
val_dataset = val_dataset.select(range(min(10, len(val_dataset))))

print(f"Train samples: {len(train_dataset)} (subset)")
print(f"Val samples: {len(val_dataset)} (subset)")
print_memory_usage()

# -------------------
# 3. Load model & processor
# -------------------
print("Loading model and processor...")

# Force garbage collection
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    # Load processor first
    print("Loading processor...")
    processor = DonutProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    print("✅ Processor loaded successfully")
    
    # Load model with minimal settings
    print("Loading model...")
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
    )
    print("✅ Model loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please check your internet connection")
    exit(1)

# Fix Donut model configuration
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Move model to device
model = model.to(device)
print(f"Model moved to {device}")
print_memory_usage()

# -------------------
# 4. Ultra Minimal Dataset
# -------------------
class UltraMinimalDonutDataset:
    def __init__(self, dataset, base_path, processor, device, max_length=64):  # Very short max_length
        self.dataset = dataset
        self.base_path = base_path
        self.processor = processor
        self.device = device
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        try:
            # Load image
            img_path = os.path.join(self.base_path, example["image"])
            image = Image.open(img_path).convert("RGB")
            
            # Resize image to smaller size for faster processing
            image = image.resize((512, 512))  # Much smaller size
            
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

            # Tokenize text with very short max length
            text = example["text"][:100]  # Limit text length
            input_ids = self.processor.tokenizer(
                text,
                max_length=self.max_length,
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
            # Return a dummy sample
            dummy_pixel_values = torch.zeros((3, 512, 512))  # Smaller size
            dummy_labels = torch.zeros(self.max_length, dtype=torch.long)
            return {
                "pixel_values": dummy_pixel_values,
                "labels": dummy_labels
            }

# Create ultra minimal datasets
train_dataset = UltraMinimalDonutDataset(
    train_dataset, 
    os.path.join(DATASET_DIR, "train"), 
    processor, 
    device,
    max_length=64  # Very short max length
)
val_dataset = UltraMinimalDonutDataset(
    val_dataset, 
    os.path.join(DATASET_DIR, "val"), 
    processor, 
    device,
    max_length=64  # Very short max length
)

# Simple collator
def simple_collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

# -------------------
# 5. Ultra Minimal Training Configuration
# -------------------
batch_size = 1
gradient_accumulation_steps = 2  # Very small
max_length = 64

print(f"Using batch size: {batch_size}, gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Max sequence length: {max_length}")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=10,  # Very frequent saves
    eval_steps=10,
    logging_steps=5,  # Very frequent logging
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=1,  # Only 1 epoch
    learning_rate=5e-6,  # Very low learning rate
    save_total_limit=1,
    predict_with_generate=True,
    remove_unused_columns=False,
    fp16=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch",
    warmup_steps=5,  # Very small warmup
    dataloader_prefetch_factor=None,
    dataloader_drop_last=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_grad_norm=1.0,
    weight_decay=0.01,
    logging_dir="./logs",
    ddp_find_unused_parameters=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    data_collator=simple_collate_fn,
)

# -------------------
# 6. Train
# -------------------
if __name__ == "__main__":
    print("Starting training...")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print("Training with ULTRA minimal settings!")
    print("This should complete much faster and avoid being killed.")
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
        print("This ultra-minimal version should work on your VM")
        print_memory_usage()
