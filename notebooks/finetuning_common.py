import copy
import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Adafactor, AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")

NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(NOTEBOOKS_DIR)
BASE_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
CACHE_DIR = os.path.join(PROJECT_DIR, "assets", "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "assets", "models", "llama-1b-claim-ft")
CHECKPOINT_DIR = f"{OUTPUT_DIR}-checkpoints"
TRAIN_DATA_PATH = os.path.join(PROJECT_DIR, "data", "ours", "train.csv")
TEST_DATA_PATH = os.path.join(PROJECT_DIR, "data", "ours", "test.csv")
DEFAULT_RESULTS_PATH = os.path.join(PROJECT_DIR, "results", "llama_1b_ft_eval.csv")

TRAIN_MESSAGES = [
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
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"step-{global_step}")
        os.makedirs(ckpt_path, exist_ok=True)
        print(f"Saving checkpoint to {ckpt_path}")
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)

    def _calculate_loss(self, logits, labels):
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


def load_base_model_and_tokenizer():
    print(f"Loading base model from {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        cache_dir=CACHE_DIR,
        use_safetensors=True,
    )
    print("Loading base tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        cache_dir=CACHE_DIR,
        use_safetensors=True,
        padding_side="left",
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_finetuned_model_and_tokenizer(model_dir):
    print(f"Loading fine-tuned model from {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        use_safetensors=True,
    )
    print("Loading fine-tuned tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_split_dataframe(path):
    print(f"Loading data from {path}")
    dataset = pd.read_csv(path)
    dataset = dataset.rename(columns={"text": "sentence"})
    dataset = dataset.dropna(subset=["sentence", "label"]).copy()
    dataset["sentence"] = dataset["sentence"].astype(str)
    return dataset


def load_training_data():
    train_df = load_split_dataframe(TRAIN_DATA_PATH)
    train_df["label"] = train_df["label"].map({1: "Yes", 0: "No"})
    train_df = train_df.filter(items=["sentence", "label"])
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_df


def load_test_data():
    test_df = load_split_dataframe(TEST_DATA_PATH)
    test_df["label"] = test_df["label"].astype(int)
    return test_df.filter(items=["sentence", "label"]).reset_index(drop=True)


def build_inference_messages(sentence):
    return [
        {
            "role": "system",
            "content": (
                "You are an AI agent used to determine whether or not a "
                "sentence is a factual claim. Only respond with Yes or No"
            ),
        },
        {
            "role": "user",
            "content": f"Is the following sentence a factual claim? {sentence}",
        },
        {"role": "assistant", "content": ""},
    ]


def label_token_ids(tokenizer, labels):
    token_ids = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.append(ids[0])
    if not token_ids:
        raise ValueError(f"Could not tokenize labels: {labels}")
    return sorted(set(token_ids))


def predict_claim_probability(model, tokenizer, sentence, device):
    messages = build_inference_messages(sentence)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        continue_final_message=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).input_ids[:, :-1].to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    yes_token_ids = label_token_ids(tokenizer, ("Yes", " yes", "YES"))
    no_token_ids = label_token_ids(tokenizer, ("No", " no", "NO"))

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        next_token_logits = outputs.logits[0, -1]
        no_logit = torch.logsumexp(next_token_logits[no_token_ids], dim=0)
        yes_logit = torch.logsumexp(next_token_logits[yes_token_ids], dim=0)
        probs = torch.softmax(torch.stack((no_logit, yes_logit)), dim=-1)

    claim_probability = probs[1].item()
    return {
        "claim_probability": claim_probability,
        "pred": int(claim_probability >= 0.5),
    }
