import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import json

class MiniCPMFineTuner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['model_config']['minicpm']
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.bfloat16
        )
        self._add_lora_adapters()

    def _add_lora_adapters(self):
        peft_config = LoraConfig(
            r=self.config['lora_config']['r'],
            lora_alpha=self.config['lora_config']['lora_alpha'],
            target_modules=self.config['lora_config']['target_modules'],
            lora_dropout=self.config['lora_config']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, peft_config)

    def load_finetune_data(self, data_path: str):
        def format_example(example):
            return {
                "text": f"段落：{example['paragraph']}\n风险点：{example['risk_point']}\n标签：{example['label']}"
            }
        
        dataset = load_dataset('json', data_files=data_path)['train']
        return dataset.map(format_example, batched=False)

    def train(self, train_data, output_dir: str):
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            max_steps=500,
            logging_steps=50,
            fp16=True,
            optim="adamw_torch",
            report_to="none"
        )

        self.model.train()
        self.model.print_trainable_parameters()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            data_collator=lambda data: {
                "input_ids": torch.stack([self.tokenizer.encode(d["text"], return_tensors="pt").squeeze() for d in data])
            }
        )

        trainer.train()
        self.model.save_pretrained(output_dir)

    def predict(self, input_text: str):
        prompt = f"分析段落风险：{input_text}\n最终结论："
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)