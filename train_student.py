"""
train_student.py
================
DeepSeekの教師出力を使って、Hugging Faceの小さな学生モデルを
知識蒸留で訓練するスクリプト。

損失関数:
    L = alpha * KLDiv(teacher_logits, student_logits) + (1-alpha) * CE(labels, student_logits)

使い方:
    python train_student.py --config config.yaml
"""

import os
import json
import math
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# データセット
# ─────────────────────────────────────────────

@dataclass
class DistillationSample:
    prompt: str
    response: str
    system_prompt: Optional[str] = None


class DistillationDataset(Dataset):
    """
    蒸留用データセット。
    教師の出力テキストを正解ラベルとして学生モデルを訓練する。
    """

    def __init__(self, data_file: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load(data_file)
        logger.info(f"Dataset loaded: {len(self.samples)} samples")

    def _load(self, filepath: str) -> list[DistillationSample]:
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if not data.get("response"):
                    continue
                samples.append(DistillationSample(
                    prompt=data["prompt"],
                    response=data["response"],
                    system_prompt=data.get("system_prompt"),
                ))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # プロンプト + 応答 を結合してラベルを作成
        if sample.system_prompt:
            full_text = f"System: {sample.system_prompt}\nUser: {sample.prompt}\nAssistant: {sample.response}"
        else:
            full_text = f"User: {sample.prompt}\nAssistant: {sample.response}"

        prompt_text = f"User: {sample.prompt}\nAssistant: "
        if sample.system_prompt:
            prompt_text = f"System: {sample.system_prompt}\n" + prompt_text

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # ラベル: プロンプト部分は -100 でマスク（応答部分のみ学習）
        prompt_enc = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # プロンプト部分を無視
        labels[attention_mask == 0] = -100  # パディング部分を無視

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ─────────────────────────────────────────────
# 蒸留損失
# ─────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    知識蒸留損失:
        L = alpha * KLDiv(student || teacher) + (1-alpha) * CE(labels, student)

    注意: logprobs APIからの教師ラベルがある場合はそれを使用し、
    ない場合は CE のみで訓練（ソフトラベルなし蒸留）。
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, student_logits, labels, teacher_logits=None):
        # Cross Entropy Loss（ハードラベル）
        ce = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        if teacher_logits is not None:
            # KL Divergence Loss（ソフトラベル）
            student_soft = F.log_softmax(student_logits / self.T, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)
            kl = self.kl_loss(student_soft.view(-1, student_soft.size(-1)),
                              teacher_soft.view(-1, teacher_soft.size(-1)))
            kl = kl * (self.T ** 2)  # 温度スケール補正
            loss = self.alpha * kl + (1 - self.alpha) * ce
        else:
            loss = ce

        return loss, ce


# ─────────────────────────────────────────────
# トレーナー
# ─────────────────────────────────────────────

class DistillationTrainer:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        set_seed(42)

        # トークナイザ・学生モデルの読み込み
        model_name = cfg["student"]["model_name"]
        logger.info(f"Loading student model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # パディングトークンがない場合は設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)

        self.loss_fn = DistillationLoss(
            temperature=cfg["distillation"]["temperature"],
            alpha=cfg["distillation"]["alpha"],
        )

    def prepare_data(self) -> tuple[DataLoader, DataLoader]:
        dataset = DistillationDataset(
            data_file=self.cfg["dataset"]["output_file"],
            tokenizer=self.tokenizer,
            max_length=self.cfg["student"]["max_length"],
        )

        val_size = max(1, int(len(dataset) * 0.1))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        bs = self.cfg["distillation"]["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

        logger.info(f"Train: {train_size} | Val: {val_size}")
        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self.prepare_data()

        dcfg = self.cfg["distillation"]
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=dcfg["learning_rate"],
            weight_decay=0.01,
        )

        total_steps = len(train_loader) * dcfg["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=dcfg["warmup_steps"],
            num_training_steps=total_steps,
        )

        output_dir = Path(self.cfg["student"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(dcfg["num_epochs"]):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{dcfg['num_epochs']}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = outputs.logits

                # 教師ロジット（今回はソフトラベルなしモード）
                # ※ ソフトラベルを使う場合は別途 teacher_logits を渡す
                loss, ce_loss = self.loss_fn(student_logits, labels, teacher_logits=None)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

                # チェックポイント保存
                if global_step % dcfg["save_steps"] == 0:
                    ckpt_path = output_dir / f"checkpoint-{global_step}"
                    self.model.save_pretrained(ckpt_path)
                    self.tokenizer.save_pretrained(ckpt_path)
                    logger.info(f"Checkpoint saved: {ckpt_path}")

            # バリデーション
            val_loss = self.evaluate(val_loader)
            avg_train_loss = epoch_loss / len(train_loader)
            logger.info(
                f"Epoch {epoch+1} | train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | perplexity={math.exp(val_loss):.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best_model"
                self.model.save_pretrained(best_path)
                self.tokenizer.save_pretrained(best_path)
                logger.info(f"Best model updated: val_loss={val_loss:.4f}")

        # 最終モデル保存
        final_path = output_dir / "final_model"
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Training complete! Model saved to: {final_path}")

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss, _ = self.loss_fn(outputs.logits, labels)
            total_loss += loss.item()

        return total_loss / len(val_loader)


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    trainer = DistillationTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
