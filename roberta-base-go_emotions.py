from transformers import AutoTokenizer, AutoModel
model_name = "SamLowe/roberta-base-go_emotions"

# Download and cache the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save locally
local_dir = "models/roberta-base-go_emotions"
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)
