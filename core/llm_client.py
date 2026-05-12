# core/llm_client.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# core/llm_client.py 的初始化部分

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=45.0,     # 🌟 关键：如果45秒没反应，强制重试，绝对不准卡6分钟！
            max_retries=3     # 🌟 关键：断网自动重连3次
        )

    def call_stream(self, model, messages):
        """核心：确保 stream=True 并且直接 yield 内容"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                stream=True # 必须开启
            )
            for chunk in response:
                # 只要有一丁点内容，就立刻吐出去
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"⚠️ 网络请求异常: {str(e)}"