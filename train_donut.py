# train_donut.py
import os
import torch
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, DonutProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image

# -------------------
# 1. Config
# -------------------
DATASET_DIR = "dataset_donut"
MODEL_NAME = "naver-clova-ix/donut-base"   # or "donut-small" if GPU is limited
OUTPUT_DIR = "./donut-finetuned"

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Running on CPU - optimizing for Azure VM")

# -------------------
# 2. Load datasets
# -------------------
print("Loading datasets...")
train_dataset = load_dataset("json", data_files=os.path.join(DATASET_DIR, "train/metadata.jsonl"))["train"]
val_dataset = load_dataset("json", data_files=os.path.join(DATASET_DIR, "val/metadata.jsonl"))["train"]

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# -------------------
# 3. Load model & processor
# -------------------
print("Loading model and processor...")
# Force use of slow tokenizer to avoid sentencepiece dependency
processor = DonutProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Fix Donut model configuration
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Move model to device
model = model.to(device)
print(f"Model moved to {device}")
print(f"Decoder start token ID: {model.config.decoder_start_token_id}")
print(f"Pad token ID: {model.config.pad_token_id}")
print(f"EOS token ID: {model.config.eos_token_id}")

# -------------------
# 4. Preprocessing - Memory Efficient Approach
# -------------------
class DonutDataset:
    def __init__(self, dataset, base_path, processor, device):
        self.dataset = dataset
        self.base_path = base_path
        self.processor = processor
        self.device = device
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Load image
        img_path = os.path.join(self.base_path, example["image"])
        image = Image.open(img_path).convert("RGB")
        
        # Process image with processor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Tokenize text label
        text = example["text"]
        input_ids = self.processor.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        return {
            "pixel_values": pixel_values,
            "labels": input_ids
        }

# Create memory-efficient datasets
train_dataset = DonutDataset(train_dataset, os.path.join(DATASET_DIR, "train"), processor, device)
val_dataset = DonutDataset(val_dataset, os.path.join(DATASET_DIR, "val"), processor, device)

# Custom data collator for our dataset
def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

# -------------------
# 5. Training setup - Optimized for CPU/Azure VM
# -------------------
# Calculate optimal batch size based on device
if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb >= 24:  # 24GB+ GPU (RTX 4090, A100, etc.)
        batch_size = 4
        gradient_accumulation_steps = 4
    elif gpu_memory_gb >= 16:  # 16GB+ GPU (RTX 4080, etc.)
        batch_size = 2
        gradient_accumulation_steps = 8
    elif gpu_memory_gb >= 12:  # 12GB+ GPU (RTX 3080, etc.)
        batch_size = 1
        gradient_accumulation_steps = 16
    else:  # 8GB GPU (RTX 3070, etc.)
        batch_size = 1
        gradient_accumulation_steps = 32
else:
    # CPU settings for Azure VM
    batch_size = 1
    gradient_accumulation_steps = 16  # Smaller for CPU

print(f"Using batch size: {batch_size}, gradient accumulation steps: {gradient_accumulation_steps}")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=200,  # More frequent saves for CPU training
    eval_steps=200,
    logging_steps=50,  # More frequent logging
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=5,  # Reduced epochs for CPU training
    learning_rate=3e-5,  # Slightly lower learning rate for CPU
    save_total_limit=2,
    predict_with_generate=True,
    remove_unused_columns=False,
    fp16=False,  # Disable mixed precision for CPU
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    report_to="none",
    # Memory optimization
    gradient_checkpointing=True,
    optim="adamw_torch",
    warmup_steps=50,  # Reduced warmup for CPU
    # CPU specific optimizations
    dataloader_prefetch_factor=None,
    # Additional CPU optimizations
    dataloader_drop_last=True,  # Drop incomplete batches
    load_best_model_at_end=True,  # Load best model at end
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    data_collator=collate_fn,
)

# -------------------
# 6. Train
# -------------------
if __name__ == "__main__":
    print("Starting training...")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print("Training on CPU - this will take longer but will work!")
    
    # Clear cache if GPU available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Training complete! Model saved at:", OUTPUT_DIR)
