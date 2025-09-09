#!/usr/bin/env python3

import torch
from datasets import load_dataset, Audio
from transformers import (
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    Trainer,
    TrainingArguments,
)


class VoxtralDataCollator:
    """Data collator for Voxtral STT training - processes audio and text."""
    
    def __init__(self, processor, model_id):
        self.processor = processor
        self.model_id = model_id
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        """
        Each feature should have:
          - "audio": raw audio (whatever your processor expects)
          - "text":  transcription string
        """
        texts  = [f["text"] for f in features]
        audios = [f["audio"]["array"] for f in features]

        # 1) Build the PROMPT part: [AUDIO]…[AUDIO] <transcribe>
        prompt = self.processor.apply_transcription_request(  # (same method you used)
            language="en",
            model_id=self.model_id if hasattr(self, "model_id") else None,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )
        # prompt["input_ids"]: shape [B, L_prompt]
        # keep any extra fields (e.g., audio features) to pass through to the model
        passthrough = {k: v for k, v in prompt.items()
                       if k not in ("input_ids", "attention_mask")}

        prompt_ids = prompt["input_ids"]           # [B, Lp]
        prompt_attn = prompt["attention_mask"]     # [B, Lp]
        B = prompt_ids.size(0)

        tok = self.processor.tokenizer
        # 2) Tokenize transcriptions WITHOUT padding; we'll pad after concatenation
        text_tok = tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
        text_ids_list = text_tok["input_ids"]

        # 3) Concatenate: input_ids = [PROMPT] + [TEXT]
        input_ids, attention_mask, labels = [], [], []
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            ids  = p_ids + t_ids + [tok.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1) 
            # labels: mask prompt tokens, learn only on text tokens
            lab  = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]


            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # 4) Pad to max length in batch
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids      = [pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0,      max_len) for x in attention_mask]
        labels         = [pad_to(x, -100,   max_len) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # 5) Include processor outputs needed by the model (e.g., audio features)
        for k, v in passthrough.items():
            batch[k] = v

        return batch

def load_and_prepare_dataset():
    """Load and prepare dataset for training."""
    dataset_name = "hf-audio/esb-datasets-test-only-sorted"
    dataset_config = "voxpopuli"
    
    print(f"Loading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    
    # Cast audio to 16kHz (required for Voxtral)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    train_dataset = dataset.select(range(100))
    eval_dataset = dataset.select(range(100, 150))
    
    return train_dataset, eval_dataset


def main():
    # Configuration
    model_checkpoint = "mistralai/Voxtral-Mini-3B-2507"
    output_dir = "./voxtral-finetuned"
    
    # Set device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")
    
    # Load processor and model
    print("Loading processor and model...")
    processor = VoxtralProcessor.from_pretrained(model_checkpoint)

    model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    # Setup data collator
    data_collator = VoxtralDataCollator(processor, model_checkpoint)
    
    # Simple training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
        eval_steps=50 if eval_dataset else None,
        save_steps=50,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=1,
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()

    
    # Save model and processor
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    # Final evaluation
    if eval_dataset:
        results = trainer.evaluate()
        print(f"Final evaluation results: {results}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
