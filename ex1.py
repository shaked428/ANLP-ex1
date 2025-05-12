import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, HfArgumentParser
from dataclasses import dataclass, field
import numpy as np
import evaluate
import os


@dataclass
class TrainingArgumentsExt:
    max_train_samples: int = field(
        default=100, metadata={"help": "Maximum number of training samples"}
    )
    max_eval_samples: int = field(
        default=100, metadata={"help": "Maximum number of evaluation samples"}
    )
    max_predict_samples: int = field(
        default=100, metadata={"help": "Maximum number of predict samples"}
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs"}
    )
    lr: float = field(
        default=2e-5, metadata={"help": "Learning rate for training"}
    )
    batch_size: int = field(
        default=8, metadata={"help": "Train batch size"}
    )
    do_train: bool = field(
        default=False, metadata={"help": "Whether to run training"}
    )
    do_predict: bool = field(
        default=False, metadata={"help": "Whether to run prediction"}
    )
    model_path: str = field(
        default="./output", metadata={"help": "Where to save/load the model"}
    )

parser = HfArgumentParser(TrainingArgumentsExt)
train_args_cli = parser.parse_args_into_dataclasses()[0]

# Load the MRPC dataset
dataset = load_dataset("glue", "mrpc")

model_name = "bert-base-uncased"
wandb_name = f"{model_name}_lr_{train_args_cli.lr}_epoch_{train_args_cli.num_train_epochs}_batch_size_{train_args_cli.batch_size}"

#wandb.login()
#wandb.init(project="bert-mrpc", name=wandb_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Define the metric
metric = evaluate.load("glue", "mrpc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args_object = TrainingArguments(
    output_dir=train_args_cli.model_path,
    learning_rate=train_args_cli.lr,
    num_train_epochs=train_args_cli.num_train_epochs,
    per_device_train_batch_size=train_args_cli.batch_size,
    per_device_eval_batch_size=train_args_cli.batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    run_name="bert-mrpc-run-1",
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    #report_to="wandb",  # This enables wandb logging
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["validation"]
test_dataset = encoded_dataset["test"]

if train_args_cli.max_train_samples != -1:
    print(f"using only {train_args_cli.max_train_samples} samples from the training set")
    train_dataset = train_dataset.select(range(train_args_cli.max_train_samples))

if train_args_cli.max_eval_samples != -1:
    print(f"using only {train_args_cli.max_eval_samples} samples from the eval set")
    eval_dataset = eval_dataset.select(range(train_args_cli.max_eval_samples))

if train_args_cli.max_predict_samples != -1:
    print(f"using only {train_args_cli.max_predict_samples} samples from the test set")
    test_dataset = test_dataset.select(range(train_args_cli.max_predict_samples))


# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args_object,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Train the model
if (train_args_cli.do_train):
    trainer.train()
    trainer.evaluate()

    # saves model + tokenizer
    trainer.save_model(train_args_cli.model_path)  
    tokenizer.save_pretrained(train_args_cli.model_path)




if (train_args_cli.do_predict):
    model = AutoModelForSequenceClassification.from_pretrained(train_args_cli.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(train_args_cli.model_path)
    test_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # test_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=False)
    test_trainer = Trainer(
        model=model,
        args=training_args_object,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=test_data_collator,
        compute_metrics=compute_metrics,
    )
    predictions = test_trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(axis=-1)
    print("create prediction.txt")
    # create the prediction.txt file 
    with open("prediction.txt", "w", encoding="utf-8") as f:
        for i in range(len(test_dataset)):
            sentence1 = dataset["test"]["sentence1"][i]
            sentence2 = dataset["test"]["sentence2"][i]
            label = predicted_labels[i]

            line = f"{sentence1}###{sentence2}###{label}\n"
            f.write(line)

#wandb.finish()

