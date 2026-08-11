from transformers import pipeline


class ExtractiveReader:
    # small QA model, answer span extraction

    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad", device: str = "cpu"):
        device_index = -1 if device == "cpu" else 0
        self.pipeline = pipeline("question-answering", model=model_name, device=device_index)

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return ""
        result = self.pipeline(question=question, context=context)
        return result["answer"]
