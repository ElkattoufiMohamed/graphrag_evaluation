from src.qwen_llm import QwenLLM
from dotenv import load_dotenv
load_dotenv()


llm = QwenLLM(model="qwen-plus")
print(llm.generate("Say 'ok' only.", temperature=0.0))
