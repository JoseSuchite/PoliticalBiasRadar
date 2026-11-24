import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "google-bert/bert-base-uncased"

class BERTEncoder:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

    def encode(self, article):
        return self.encode_batch([article])

    def encode_batch(self, articles):
        inputs = self.tokenizer(
            articles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * mask).sum(dim=1)
        embeddings = embeddings / mask.sum(dim=1).clamp_min(1)

        return embeddings.to("cpu", copy=True)
