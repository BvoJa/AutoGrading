# model.py
import torch
import pytorch_lightning as pl
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class Qwen3VLModule(pl.LightningModule):
    def __init__(self, model_name="Qwen/Qwen3-VL-4B-Instruct"):
        super().__init__()
        # Lưu hyperparameters để tái sử dụng
        self.save_hyperparameters()
        
        # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load Model
        # Lưu ý: device_map="auto" sẽ tự động chia model lên GPU/CPU
        # Nếu dùng Lightning Trainer để train thì nên bỏ device_map và để Trainer quản lý
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto" 
        )

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        # Hàm forward chuẩn nếu bạn muốn mở rộng sau này
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )

    def predict_answer(self, image_source: str, prompt_text: str, max_new_tokens=128):
        """
        Hàm helper để thực hiện Inference nhanh:
        Input: 
            - image_source: URL hoặc đường dẫn file ảnh
            - prompt_text: Câu hỏi hoặc yêu cầu cho model
        Output:
            - Chuỗi văn bản trả lời
        """
        # 1. Tạo cấu trúc messages chuẩn cho Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_source,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # 2. Preprocessing bằng template chat của model
        # apply_chat_template sẽ tự động xử lý ảnh và text thành tensors
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 3. Đưa dữ liệu vào đúng thiết bị mà model đang nằm (GPU/CPU)
        inputs = inputs.to(self.model.device)

        # 4. Generate
        # Dùng torch.no_grad() để tiết kiệm bộ nhớ khi inference
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # 5. Post-processing (Cắt bỏ phần prompt đầu vào, chỉ lấy phần mới sinh ra)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] # Trả về chuỗi kết quả đầu tiên