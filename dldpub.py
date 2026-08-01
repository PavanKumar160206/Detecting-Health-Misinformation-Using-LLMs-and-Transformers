from datasets import load_dataset

# Load PUBHEALTH dataset
dataset = load_dataset("ImperialCollegeLondon/health_fact")

# See the available splits
print(dataset)

# Look at the first training example
print(dataset["train"][0])
