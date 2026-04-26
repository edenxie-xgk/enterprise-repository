from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _enable_lora(model, *, rank: int, alpha: int, dropout: float, target_modules: list[str]):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA training requires the `peft` package. Install it with `pip install peft` "
            "or from requirements-train.txt."
        ) from exc

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
    return get_peft_model(model, peft_config)


def _render_plain_chat(messages: list[dict], *, add_generation_prompt: bool) -> str:
    blocks = []
    for message in messages:
        role = str(message.get("role") or "user").strip().upper()
        content = str(message.get("content") or "").strip()
        blocks.append(f"{role}:\n{content}")
    if add_generation_prompt:
        blocks.append("ASSISTANT:\n")
    return "\n\n".join(blocks)


def _apply_chat_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return _render_plain_chat(messages, add_generation_prompt=add_generation_prompt)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class FinancialFactLoRADataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        messages = list(row.get("messages") or [])
        if len(messages) < 2:
            raise ValueError("Each training example must contain at least system/user/assistant messages")

        prompt_messages = messages[:-1]
        full_text = _apply_chat_template(self.tokenizer, messages, add_generation_prompt=False)
        prompt_text = _apply_chat_template(self.tokenizer, prompt_messages, add_generation_prompt=True)

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        prompt_tokens = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        input_ids = torch.tensor(full_tokens["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full_tokens["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _build_collator(tokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


def _resolve_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for the financial fact extraction model.")
    parser.add_argument("--train-file", default="data/financial_fact_lora_from_hf.jsonl", help="Training JSONL file")
    parser.add_argument("--model-name", required=True, help="Base Hugging Face causal LM")
    parser.add_argument("--output-dir", default="outputs/financial_fact_extractor_lora", help="Adapter output directory")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum tokenized sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-step batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--num-train-epochs", type=int, default=1, help="Epoch count")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimization steps")
    parser.add_argument("--log-every", type=int, default=10, help="Print loss every N optimizer steps")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16", help="Model weight dtype")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote model code")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(train_path)
    if not rows:
        raise RuntimeError(f"No training rows found in {train_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise RuntimeError("Tokenizer must provide either pad_token, eos_token, or unk_token")

    torch_dtype = _resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model = _enable_lora(
        model,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
    )

    dataset = FinancialFactLoRADataset(rows, tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        collate_fn=_build_collator(tokenizer),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    use_mixed_precision = device.type == "cuda" and args.dtype in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    optimizer_step = 0
    running_loss = 0.0

    for epoch in range(max(1, args.num_train_epochs)):
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_mixed_precision):
                outputs = model(**batch)
                loss = outputs.loss / max(1, args.gradient_accumulation_steps)

            loss.backward()
            running_loss += float(loss.item())

            if step % max(1, args.gradient_accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if optimizer_step % max(1, args.log_every) == 0:
                    print(json.dumps({"optimizer_step": optimizer_step, "loss": running_loss}, ensure_ascii=False))
                    running_loss = 0.0

                if args.max_steps > 0 and optimizer_step >= args.max_steps:
                    break

        if args.max_steps > 0 and optimizer_step >= args.max_steps:
            break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "train_file": str(train_path),
        "model_name": args.model_name,
        "rows": len(rows),
        "optimizer_steps": optimizer_step,
        "epochs": args.num_train_epochs,
        "training_mode": "lora",
        "device": str(device),
        "dtype": args.dtype,
        "output_dir": str(output_dir),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
