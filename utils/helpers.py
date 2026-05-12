# utils/helpers.py
import time
import json
from datetime import datetime

def stream_formatter(text, delay=0.02):
    """
    逐字产生文本。增加了一点点延迟确保肉眼可见。
    """
    for char in text:
        yield char
        time.sleep(delay)

def save_analysis_result(user_input, final_result, filename="data/history.jsonl"):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "analysis": final_result
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")