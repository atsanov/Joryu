import json
import os
import yaml
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

# --- 設定 ---
TOTAL_PROMPTS = 5000   # 最終的に欲しい件数
BATCH_SIZE = 50        # 1回のリクエストでDeepSeekに作らせる件数
MAX_WORKERS = 5        # 同時にリクエストを投げる数（並列度）

def fetch_batch(client, model):
    """DeepSeekにプロンプトをまとめて生成させる"""
    prompt_text = (
        f"日本史および世界史に関する、具体的で多様な質問を{BATCH_SIZE}個出力してください。\n"
        "ルール:\n"
        "1. 1行に1つ、問いのみを出力すること。\n"
        "2. 解説や番号、前置きは一切不要。\n"
        "3. 可能な限り時代や地域を分散させること。"
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "あなたは歴史の専門家です。簡潔に問いのみをリストアップします。"},
            {"role": "user", "content": prompt_text}
        ],
        max_tokens=2000,
        temperature=1.0 # 多様性を出す
    )
    
    # 行ごとに分割してクリーンアップ
    lines = response.choices[0].message.content.strip().split('\n')
    clean_lines = [l.strip().lstrip('0123456789. ') for l in lines if l.strip()]
    return clean_lines

def fast_generate_prompts():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    client = OpenAI(api_key=cfg['deepseek']['api_key'], base_url=cfg['deepseek']['base_url'])
    output_file = "data/prompts.jsonl"
    os.makedirs("data", exist_ok=True)

    # 既存の件数確認
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_count = sum(1 for _ in f)

    pbar = tqdm(total=TOTAL_PROMPTS, initial=existing_count, desc="プロンプト爆速生成中")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        # 必要な回数分、非同期タスクを投入
        needed_requests = (TOTAL_PROMPTS - existing_count) // BATCH_SIZE + 1
        for _ in range(needed_requests):
            futures.append(executor.submit(fetch_batch, client, cfg['deepseek']['model']))

        with open(output_file, "a", encoding="utf-8") as f:
            for future in concurrent.futures.as_completed(futures):
                if existing_count >= TOTAL_PROMPTS:
                    break
                
                try:
                    new_prompts = future.result()
                    for p in new_prompts:
                        if existing_count < TOTAL_PROMPTS:
                            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
                            existing_count += 1
                            pbar.update(1)
                    f.flush()
                except Exception as e:
                    print(f"\nエラー発生（スキップします）: {e}")

    pbar.close()
    print(f"完了！ {output_file} に合計 {existing_count} 件保存されました。")

fast_generate_prompts()
