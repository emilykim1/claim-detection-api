from finetuning_common import (
    OUTPUT_DIR,
    TRAIN_MESSAGES,
    BinaryClassificationTuner,
    load_base_model_and_tokenizer,
    load_training_data,
)


if __name__ == "__main__":
    print("Starting finetuning script")
    model, tokenizer = load_base_model_and_tokenizer()
    train_df = load_training_data()

    print(f"Loaded {len(train_df)} training rows")
    print(train_df["label"].value_counts())

    tuner = BinaryClassificationTuner(model, tokenizer, train_df, TRAIN_MESSAGES)
    tuner.train(epochs=1, batch_size=10)

    print(f"Saving final model and tokenizer to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done")
