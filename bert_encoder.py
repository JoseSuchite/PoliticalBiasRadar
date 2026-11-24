import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


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
            max_length=512,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True
        )

        # Track which article each chunk belongs to
        article_mapping = inputs.pop("overflow_to_sample_mapping")
        special_mask = inputs.pop("special_tokens_mask")

        article_embeddings = torch.zeros(
            len(articles), 768, device=self.device
        )

        token_counts = torch.zeros(
            len(articles), 1, device=self.device
        )

        # Process chunks in small batches to limit GPU memory usage
        chunk_batch_size = 8

        with torch.no_grad():
            for start in range(0, len(article_mapping), chunk_batch_size):
                end = start + chunk_batch_size

                batch = {
                    key: value[start:end].to(self.device)
                    for key, value in inputs.items()
                }

                outputs = self.model(**batch)

                # Exclude padding and special tokens
                mask = batch["attention_mask"] * (
                    1 - special_mask[start:end].to(self.device)
                )
                mask = mask.unsqueeze(-1)

                # Sum token embeddings within each chunk
                chunk_embeddings = (
                    outputs.last_hidden_state * mask
                ).sum(dim=1)

                chunk_counts = mask.sum(dim=1).float()

                mapping = article_mapping[start:end].to(self.device)

                # Combine chunks belonging to the same article
                article_embeddings.index_add_(
                    0, mapping, chunk_embeddings
                )

                token_counts.index_add_(
                    0, mapping, chunk_counts
                )

        # Average across all content tokens in each article
        embeddings = article_embeddings / token_counts.clamp_min(1)

        return embeddings.to("cpu", copy=True)
