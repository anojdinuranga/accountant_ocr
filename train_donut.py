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
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Fix Donut model configuration
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Move model to GPU
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
# 5. Training setup - Optimized for GPU
# -------------------
# Calculate optimal batch size based on GPU memory
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
    batch_size = 1
    gradient_accumulation_steps = 32

print(f"Using batch size: {batch_size}, gradient accumulation steps: {gradient_accumulation_steps}")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=10,
    learning_rate=5e-5,
    save_total_limit=2,
    predict_with_generate=True,
    remove_unused_columns=False,
    fp16=True,  # Mixed precision for speed and memory efficiency
    dataloader_pin_memory=False,  # Disable pin memory to reduce CPU RAM usage
    dataloader_num_workers=0,  # Single worker to avoid memory issues
    report_to="none",  # disable wandb/hf logging unless you want it
    # Memory optimization
    gradient_checkpointing=True,  # Trade compute for memory
    optim="adamw_torch",  # Use PyTorch optimizer for better memory efficiency
    warmup_steps=100,  # Gradual warmup to avoid memory spikes
    # GPU specific optimizations
    dataloader_prefetch_factor=None,  # Disable prefetching to save memory
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
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Training complete! Model saved at:", OUTPUT_DIR)
