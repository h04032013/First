#pip install transformers datasets

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("open-web-math/open-web-math")

train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

#use tokenizer from pre-trained model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct")

def tokenize_function(examples):
    #calll that pre-trained tokenizer
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train_data = train_data.map(tokenize_function, batched=True)
tokenized_test_data = test_data.map(tokenize_function, batched=True)

tokenized_train_data = tokenized_train_data.remove_columns(["text"])
tokenized_test_data = tokenized_test_data.remove_columns(["text"])

tokenized_train_data.set_format("torch")
tokenized_test_data.set_format("torch")


#start getting ready for pretraining

training_args = TrainingArguments(
    output_dir="./phi3.5-mini-finetuned",
    evaluation_strategy="epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data["train"],
    eval_dataset=tokenized_test_data["test"],
)


#training
trainer.train()

#5-generate responses from each mdodel and compare

base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
base_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

#def generate_response(model, tokenizer, prompt):
 #   inputs = tokenizer(prompt, return_tensors="pt")
  #  outputs = model.generate(**inputs)
   # return tokenizer.decode(outputs, skip_special_tokens=True)


prompts = [
    "im in 7th grade explain a function to me", 
    "If A=2+i, O=-4, P=-i, and S=2+4i, find A-O+P+S", 
    "The perimeter of a rectangle is 24 inches. What is the number of square inches in the maximum possible area for this rectangle?",
    "Megan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a seven-digit number that meets these conditions, what is the probability that she dials Fatima's correct number? Express your answer as a common fraction.", 
    "Square ABCD has its center at (8,-8) and has an area of 4 square units. The top side of the square is horizontal. The square is then dilated with the dilation center at (0,0) and a scale factor of 2. What are the coordinates of the vertex of the image of square ABCD that is farthest from the origin? Give your answer as an ordered pair",
    "At the MP Donut Hole Factory, Niraek, Theo, and Akshaj are coating spherical donut holes in powdered sugar.  Niraek's donut holes have radius 6 mm, Theo's donut holes have radius 8 mm, and Akshaj's donut holes have radius 10 mm.  All three workers coat the surface of the donut holes at the same rate and start at the same time.  Assuming that the powdered sugar coating has negligible thickness and is distributed equally on all donut holes, how many donut holes will Niraek have covered by the first time all three workers finish their current donut hole at the same time?",
    "I'm having trouble understanding how to convert between radians and degrees. If I have an angle of 5π665pπ radians, how do I convert that to degrees?",
    "Can you explain how to solve this equation using logarithms? I'm stuck on 5x=2005x=200. What steps do I take?",
    "I need help finding the vertical and horizontal asymptotes for the function f(x)=(2x^2)+3x-1/(x^2-4). How do I figure them out?",
    "Could you explain how to find the sum of a geometric series? For example, how do I find the sum of the series 3+6+12+24+… if it goes on forever?",
    "How do I prove that Au(B∩C)=(AuB)∩(AuC)? I'm confused about how to use set identities to show this.",
    "Can you help me figure out how many different ways I can arrange the letters in the word 'MATHEMATICS'? I'm not sure how to handle the repeated letters.",
    "How do I determine if a graph has an Eulerian path? What conditions should I check in a graph like this one, where some vertices have odd degrees?",
    "I'm working on writing a proof by induction, but I'm stuck. How would I prove that the sum of the first nn odd numbers is n2n2?",
    "I'm having trouble solving recurrence relations. Can you show me how to solve the recurrence relation T(n)=2T(n/2)+nT(n)=2T(n/2)+n using the master theorem?",
    "Can you explain how to multiply two matrices? For example, how would I compute the product of the matrices A=[[2,3],[1,4]]and B = [[5,6],[7,8]]?",
    "How do I find the determinant of a 3x3 matrix? I'm working on [[2,0,1],[3,-1,4],[5,2,6]], but I'm not sure what steps to take.",
    "Can you walk me through how to find the eigenvalues and eigenvectors of a matrix? For example, how would I find them for the matrix A=[[4,1],[2,3]]?",
    "I'm having trouble understanding the concept of a basis for a vector space. Could you explain how to determine if the set of vectors {(1, 2, 3), (2, 1, 0), (0, 3, 1)} forms a basis for R^3?",
    "How do I find the matrix representation of a linear transformation? For example, if I have the transformation T: R^2 to R^2 defined by ( T(x, y) = (2x + 3y, 4x - y)), how do I represent T as a matrix?",
    "Introduce the topics of integration by substitution and partial integration",
    "I am struggling understanding volumes of a revolution",
    "I'm working on finding the area between curves. How do I find the area between y=x^2 and y=2xy from x=0to x=2?",
    "I'm struggling with u-substitution. For the integral ∫(2x+1)ex(^2)+xdx, how do I decide what substitution to make?",
    "How do I use the disk method to find the volume of a solid generated by rotating the curve y=x around the x-axis from x=0 to x=4?",
    "Explain reimann sums"
    # ... 27 prompts in total
]

def generate_responses(model, tokenizer, prompts):
    responses = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses[prompt] = response
    return responses


# Generate responses from the base model
base_responses = generate_responses(base_model, base_tokenizer, prompts)

# Generate responses from the fine-tuned model
finetuned_responses = generate_responses(model, tokenizer, prompts)

# Write responses to an output file
output_file = "responses.txt"

with open(output_file, "w") as f:
    for prompt in prompts:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Base Model Response: {base_responses[prompt]}\n")
        f.write(f"Fine-Tuned Model Response: {finetuned_responses[prompt]}\n")
        f.write("\n")
#testing for git 12/0222
#.conda/pkgs/cache/09cdf8bf.json
# .conda/pkgs/cache/497deca9.json


