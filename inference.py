# inference.py
from model import Qwen3VLModule
import torch

def main():
    # 1. Cấu hình
    MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
    PROMPT = "Describe this image."

    print("Loading model... (this may take a while)")
    # Khởi tạo model module
    qwen_module = Qwen3VLModule(model_name=MODEL_NAME)
    
    # Chuyển sang chế độ eval (tắt dropout, v.v.)
    qwen_module.eval()

    print(f"Processing image: {IMAGE_URL}")
    print(f"Prompt: {PROMPT}")
    print("-" * 50)

    # 2. Gọi hàm suy luận
    response = qwen_module.predict_answer(
        image_source=IMAGE_URL,
        prompt_text=PROMPT,
        max_new_tokens=128
    )

    # 3. In kết quả
    print("Model Output:")
    print(response)

if __name__ == "__main__":
    main()