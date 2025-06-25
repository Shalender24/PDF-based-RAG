from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Generator:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_id = model_id

    def load_local_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True,
            top_p=0.95,
        )

        return HuggingFacePipeline(pipeline=pipe)
