#pip install transformers datasets


from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

#use tokenizer from pre-trained model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

ds = load_dataset("open-web-math/open-web-math")

def tokenize_function(examples):
    #calll that pre-trained tokenizer
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")


#start getting ready for pretraining

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)


#train
trainer.train()

#5-generate responses from each mdodel and compare

base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs, skip_special_tokens=True)

prompt = "I am struggling to understand volume of revolutions topic in my calculus class"
base_response = generate_response(base_model, base_tokenizer, prompt)

 # from finetuned model
finetuned_response = generate_response(model, tokenizer, prompt)






