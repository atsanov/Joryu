import json
import os
import yaml
from openai import OpenAI
from tqdm import tqdm

# --- 設定 ---
COUNT_TO_GENERATE = 5000  # 生成したい合計件数
BATCH_SIZE = 50           # 1回のリクエストで生成させる件数（効率化のため）

def generate_prompts():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    client = OpenAI(api_key=cfg['deepseek']['api_key'], base_url=cfg['deepseek']['base_url'])
    output_file = "data/prompts.jsonl"
    os.makedirs("data", exist_ok=True)

    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_count = sum(1 for _ in f)

    pbar = tqdm(total=COUNT_TO_GENERATE, initial=existing_count, desc="プロンプト生成中")

    while existing_count < COUNT_TO_GENERATE:
        try:
            # DeepSeekに「問い」だけを大量に作らせる
            response = client.chat.completions.create(
                model=cfg['deepseek']['model'],
                messages=[
                    {"role": "system", "content": "あなたは歴史の専門家です。"},
                    {"role": "user", "content": f"日本史および世界史に関する、具体的で多様な質問（プロンプト）を{BATCH_SIZE}個、1行に1つずつ出力してください。解説は不要です。問いのみを出力してください。"}
                ],
                max_tokens=2000,
                temperature=0.9 # 多様性を出すために高め
            )
            
            new_prompts = response.choices[0].message.content.strip().split('\n')
            
            with open(output_file, "a", encoding="utf-8") as f:
                for p in new_prompts:
                    p = p.strip().lstrip('0123456789. ') # 番号などが付いている場合に削除
                    if p and existing_count < COUNT_TO_GENERATE:
                        f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
                        existing_count += 1
                        pbar.update(1)
        
        except Exception as e:
            print(f"エラー発生。再試行します: {e}")
            continue

    pbar.close()
    print(f"完了！ {output_file} に合計 {existing_count} 件のプロンプトが保存されました。")

generate_prompts()
