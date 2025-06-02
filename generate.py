# generate.py
# Load your trained model and generate new text!

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ✅ TODO 1: Point to the directory of your trained model
model_path = "./models/your-model"

# ✅ TODO 2: Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# ✅ TODO 3: Set up a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✅ TODO 4: Ask the user for a prompt
prompt = input("Enter your starting prompt: ")

# ✅ TODO 5: Generate a continuation using your fine-tuned model
output = generator(prompt, max_length=50, num_return_sequences=1)

print("📝 Generated Text:")
print(output[0]['generated_text'])
