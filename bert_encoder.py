import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


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
            max_length=384,
            return_overflowing_tokens=True
        )

        article_mapping = inputs.pop("overflow_to_sample_mapping")

        article_embeddings = torch.zeros(
            len(articles),
            self.model.config.hidden_size,
            device=self.device
        )

        token_counts = torch.zeros(
            len(articles), 1, device=self.device
        )

        chunk_batch_size = 8

        with torch.no_grad():
            for start in range(0, len(article_mapping), chunk_batch_size):
                end = start + chunk_batch_size

                batch = {
                    key: value[start:end].to(self.device)
                    for key, value in inputs.items()
                }

                outputs = self.model(**batch)

                # Mean pooling with attention masking
                mask = batch["attention_mask"].unsqueeze(-1)

                chunk_embeddings = (
                    outputs.last_hidden_state * mask
                ).sum(dim=1)

                chunk_counts = mask.sum(dim=1).float()

                mapping = article_mapping[start:end].to(self.device)

                # Combine chunks from the same article
                article_embeddings.index_add_(
                    0, mapping, chunk_embeddings
                )

                token_counts.index_add_(
                    0, mapping, chunk_counts
                )

        # Weighted mean across article chunks
        embeddings = article_embeddings / token_counts.clamp_min(1)

        # Normalize the final article embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.to("cpu", copy=True)
