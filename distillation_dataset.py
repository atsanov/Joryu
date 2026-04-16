"""
distillation_dataset.py
=======================
DeepSeek API（教師モデル）からソフトラベル・出力テキストを収集し、
蒸留用データセットを生成するスクリプト。

使い方:
    python distillation_dataset.py --config config.yaml
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
import openai  # DeepSeek は OpenAI互換API
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DeepSeek API クライアント
# ─────────────────────────────────────────────

class DeepSeekTeacher:
    """DeepSeek APIを教師モデルとして使うクラス"""

    def __init__(self, api_key: str, base_url: str, model: str,
                 temperature: float = 1.0, max_tokens: int = 512):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 retries: int = 3) -> dict:
        """
        プロンプトに対してDeepSeekの応答を生成する。

        Returns:
            {
                "prompt": str,
                "response": str,
                "logprobs": list | None,  # ソフトラベルとして使用
                "model": str,
            }
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    logprobs=True,          # トークンレベルのlogprobsを要求
                    top_logprobs=5,         # 上位5トークンの確率を取得
                )

                choice = response.choices[0]
                content = choice.message.content

                # logprobs の抽出
                logprobs_data = None
                if choice.logprobs and choice.logprobs.content:
                    logprobs_data = [
                        {
                            "token": lp.token,
                            "logprob": lp.logprob,
                            "top_logprobs": [
                                {"token": tlp.token, "logprob": tlp.logprob}
                                for tlp in (lp.top_logprobs or [])
                            ]
                        }
                        for lp in choice.logprobs.content
                    ]

                return {
                    "prompt": prompt,
                    "response": content,
                    "logprobs": logprobs_data,
                    "model": self.model,
                    "system_prompt": system_prompt,
                }

            except openai.RateLimitError:
                wait = 2 ** attempt * 5
                logger.warning(f"Rate limit hit. Waiting {wait}s... (attempt {attempt+1}/{retries})")
                time.sleep(wait)

            except openai.APIError as e:
                logger.error(f"API error: {e} (attempt {attempt+1}/{retries})")
                if attempt == retries - 1:
                    raise
                time.sleep(2)

        raise RuntimeError(f"Failed after {retries} retries for prompt: {prompt[:50]}...")


# ─────────────────────────────────────────────
# データ収集
# ─────────────────────────────────────────────

def load_prompts(filepath: str) -> list[dict]:
    """
    プロンプトファイルを読み込む。
    jsonl形式: {"prompt": "...", "system": "..."(optional)}
    """
    prompts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    logger.info(f"Loaded {len(prompts)} prompts from {filepath}")
    return prompts


def collect_teacher_outputs(
    teacher: DeepSeekTeacher,
    prompts: list[dict],
    output_file: str,
    max_samples: Optional[int] = None,
    request_delay: float = 0.5,
) -> None:
    """
    教師モデルの出力を収集してjsonlに保存する。
    既存ファイルがある場合は再開（resume）対応。
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 既に処理済みのプロンプトを確認（resume対応）
    processed_prompts = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_prompts.add(data["prompt"])
        logger.info(f"Resuming: {len(processed_prompts)} samples already collected.")

    if max_samples:
        prompts = prompts[:max_samples]

    pending = [p for p in prompts if p["prompt"] not in processed_prompts]
    logger.info(f"Collecting {len(pending)} remaining samples...")

    with open(output_path, "a", encoding="utf-8") as f:
        for item in tqdm(pending, desc="Collecting teacher outputs"):
            prompt = item["prompt"]
            system = item.get("system", None)

            result = teacher.generate(prompt, system_prompt=system)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            time.sleep(request_delay)  # API負荷軽減

    logger.info(f"Done! Data saved to: {output_file}")


# ─────────────────────────────────────────────
# サンプルプロンプト生成
# ─────────────────────────────────────────────

def create_sample_prompts(output_file: str) -> None:
    """動作確認用のサンプルプロンプトを生成"""
    samples = [
        {"prompt": "機械学習とは何ですか？簡潔に説明してください。"},
        {"prompt": "Pythonでリストを逆順にする方法を教えてください。"},
        {"prompt": "知識蒸留（Knowledge Distillation）の利点を3つ挙げてください。"},
        {"prompt": "自然言語処理の主なタスクにはどのようなものがありますか？"},
        {"prompt": "ニューラルネットワークの過学習を防ぐ方法は？"},
    ]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Sample prompts created: {output_file}")


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect teacher outputs from DeepSeek API")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--create-sample", action="store_true",
                        help="サンプルプロンプトファイルを生成して終了")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.create_sample:
        create_sample_prompts(cfg["dataset"]["input_file"])
        return

    # 教師モデル初期化
    api_key = cfg["deepseek"]["api_key"] or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY":
        raise ValueError("DeepSeek APIキーを config.yaml または 環境変数 DEEPSEEK_API_KEY に設定してください")

    teacher = DeepSeekTeacher(
        api_key=api_key,
        base_url=cfg["deepseek"]["base_url"],
        model=cfg["deepseek"]["model"],
        temperature=cfg["deepseek"]["temperature"],
        max_tokens=cfg["deepseek"]["max_tokens"],
    )

    prompts = load_prompts(cfg["dataset"]["input_file"])

    collect_teacher_outputs(
        teacher=teacher,
        prompts=prompts,
        output_file=cfg["dataset"]["output_file"],
        max_samples=cfg["dataset"]["max_samples"],
    )


if __name__ == "__main__":
    main()
