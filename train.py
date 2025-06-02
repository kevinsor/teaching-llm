# train.py
# Fine-tunes a small GPT-2 model on your writing
# Run this file once your dataset.txt is ready

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# ✅ TODO 1: Set your model name (we’ll use distilgpt2 for fast training)
model_name = "distilgpt2"

# ✅ TODO 2: Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ TODO 3: Load your dataset
def load_dataset(tokenizer, file_path="dataset.txt", block_size=64):
    # Use the TextDataset helper to tokenize your dataset
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset(tokenizer)

# ✅ TODO 4: Prepare data collator (this helps mask tokens for training)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# ✅ TODO 5: Set your training arguments
training_args = TrainingArguments(
    output_dir="./models/your-model",
    overwrite_output_dir=True,
    num_train_epochs=3,  # You can change this!
    per_device_train_batch_size=2,
    save_steps=50,
    save_total_limit=2,
    logging_dir="./logs"
)

# ✅ TODO 6: Create a Trainer object and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# ✅ TODO 7: Train the model!
trainer.train()

# ✅ TODO 8: Save your model for future use
trainer.save_model("./models/your-model")
tokenizer.save_pretrained("./models/your-model")

print("✅ Training complete. Model saved in ./models/your-model")
