import copy
import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Adafactor, AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")

# Prompt engineering:
# https://community.openai.com/t/prompt-engineering-for-rag/621495
#
# Fine-tuning is based on the following:
# - A base model
# - A base tokenizer
# - A set of desired (input, output) pairs
# - Some nuance with how the chat template is applied to the input/output pairs
# - A fixed system prompt and a fixed Yes/No output format

cache_dir = "../assets/models"
output_dir = "../assets/models/llama-1b-claim-ft"
checkpoint_dir = f"{output_dir}-checkpoints"
model_path = "meta-llama/Llama-3.2-1B-Instruct"


class BinaryClassificationTuner:
    def __init__(self, model, tokenizer, train_dataset, messages):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.messages = messages

    def train(self, epochs, batch_size=1):
        print("Preparing optimizer")
        optimizer = Adafactor(self.model.parameters(), weight_decay=0.01)
        print("Preparing training data")
        train_dataset = self._prepare_train_data()
        print(f"Prepared {len(train_dataset)} training instances")
        global_step = 0
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}")
            np.random.shuffle(train_dataset)
            num_batches = (len(train_dataset) + batch_size - 1) // batch_size
            for batch_index in range(num_batches):
                batch_start = batch_index * batch_size
                batch_end = min(batch_start + batch_size, len(train_dataset))
                batch_samples = train_dataset[batch_start:batch_end]
                batch = self._collate_batch(batch_samples)
                logits = self.model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )["logits"]
                loss = self._calculate_loss(
                    logits,
                    batch["label_input_ids"],
                ).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                step = batch_index + 1
                if step == 1 or step % 100 == 0 or step == num_batches:
                    print(
                        f"epoch {epoch + 1}/{epochs} | "
                        f"batch {step}/{num_batches} | "
                        f"examples {batch_end}/{len(train_dataset)} | "
                        f"loss {loss.item():.6f}"
                    )

                if global_step % 500 == 0:
                    self._save_checkpoint(global_step)

    def _prepare_train_data(self):
        """
        Return the train dataset as a list of dictionaries.
        """
        train_dataset = self.train_dataset.to_dict(orient="records")
        train_dataset_prepared = []
        total_instances = len(train_dataset)

        for idx, train_instance in enumerate(train_dataset, start=1):
            messages = copy.deepcopy(self.messages)
            for message in messages:
                if message["role"] == "user":
                    message["content"] = message["content"].replace(
                        "__SENTENCE__",
                        train_instance["sentence"],
                    )
                    break

            chat_template_input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                continue_final_message=True,
                add_generation_prompt=False,
                return_tensors="pt",
            ).input_ids[:, :-1]

            label_input_ids = self.tokenizer(
                train_instance["label"],
                add_special_tokens=False,
                return_tensors="pt",
                padding="max_length",
                max_length=chat_template_input_ids.shape[1],
            )["input_ids"]
            label_input_ids = torch.where(
                label_input_ids != self.tokenizer.pad_token_id,
                label_input_ids,
                -100,
            )

            train_dataset_prepared.append(
                {
                    "chat_template_input_ids": chat_template_input_ids,
                    "label_input_ids": label_input_ids,
                }
            )

            if idx == 1 or idx % 500 == 0 or idx == total_instances:
                print(f"Prepared training example {idx}/{total_instances}")

        return train_dataset_prepared

    def _collate_batch(self, batch_samples):
        input_ids = pad_sequence(
            [sample["chat_template_input_ids"].squeeze(0) for sample in batch_samples],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        label_input_ids = pad_sequence(
            [sample["label_input_ids"].squeeze(0) for sample in batch_samples],
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_input_ids": label_input_ids,
        }

    def _save_checkpoint(self, global_step):
        ckpt_path = os.path.join(checkpoint_dir, f"step-{global_step}")
        os.makedirs(ckpt_path, exist_ok=True)
        print(f"Saving checkpoint to {ckpt_path}")
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)

    def _calculate_loss(self, logits, labels):
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        cross_entropy_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return cross_entropy_loss


def load_model_and_tokenizer():
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        use_safetensors=True,
    )
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        use_safetensors=True,
        padding_side="left",
        fix_mistral_regex=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_training_data():
    print("Loading training data from ../data/ours/train.csv")
    train_df = pd.read_csv("../data/ours/train.csv")
    train_df = train_df.rename(columns={"text": "sentence"})
    train_df["label"] = train_df["label"].map({1: "Yes", 0: "No"})
    train_df = train_df.filter(items=["sentence", "label"])
    train_df = train_df.dropna(subset=["sentence", "label"]).copy()
    train_df["sentence"] = train_df["sentence"].astype(str)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_df


def calculate_loss(logits, labels):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    cross_entropy_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return cross_entropy_loss


if __name__ == "__main__":
    print("Starting finetuning script")
    model, tokenizer = load_model_and_tokenizer()
    train_df = load_training_data()

    print(f"Loaded {len(train_df)} rows")
    print(train_df["label"].value_counts())

    # Demo
    print("Running BinaryClassificationTuner demo")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI agent used to determine whether or not a "
                "sentence is a factual claim. Only respond with Yes or No"
            ),
        },
        {
            "role": "user",
            "content": "Is the following sentence a factual claim? __SENTENCE__",
        },
        {"role": "assistant", "content": ""},
    ]
    bct = BinaryClassificationTuner(model, tokenizer, train_df, messages)
    bct.train(epochs=1, batch_size=10)

    print("Running single-example optimization demo")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a yes/no answering bot. Only respond to questions "
                "with Yes or No"
            ),
        },
        {
            "role": "user",
            "content": "Is the capital of New York state New York City?",
        },
        {"role": "assistant", "content": ""},
    ]
    answer = "Yes"
    chat_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
    )
    chat_template_input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        continue_final_message=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).input_ids[:, :-1]

    label_tokenized = tokenizer(
        [answer],
        add_special_tokens=False,
        return_tensors="pt",
        padding="max_length",
        max_length=chat_template_input_ids.shape[1],
    )["input_ids"]

    # -100 comes from the Llama documentation recommendation for loss masking.
    label_tokenized_fixed = torch.where(
        label_tokenized != tokenizer.pad_token_id,
        label_tokenized,
        -100,
    )

    # Test a before response.
    print("Generating pre-training response")
    print(tokenizer.batch_decode(model.generate(chat_template_input_ids, max_new_tokens=1))[0])

    print("Preparing optimizer for single-example demo")
    optimizer = Adafactor(model.parameters(), weight_decay=0.01)

    for step in range(3):
        logits = model(chat_template_input_ids, use_cache=False)["logits"]
        loss = calculate_loss(logits, label_tokenized_fixed).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"single-example step {step + 1}/3 | loss {loss.item():.6f}")

    print("Generating post-training response")
    print(tokenizer.batch_decode(model.generate(chat_template_input_ids, max_new_tokens=1))[0])

    print(f"Saving model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done")
