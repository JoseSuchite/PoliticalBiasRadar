import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ENCODER_MODELS = {
    1: ("google-bert/bert-base-uncased", 512),
    2: ("sentence-transformers/all-mpnet-base-v2", 384),
}


class Encoder:
    def __init__(self, model_name, max_length, max_chunks=None):
        if max_chunks is not None and max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = max_chunks
        self.max_length = max_length
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

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
            max_length=self.max_length,
            return_overflowing_tokens=True
        )

        article_mapping = inputs.pop("overflow_to_sample_mapping")
        slots_per_article = self.max_chunks if self.max_chunks is not None else 1
        embedding_rows = len(articles) * slots_per_article
        if self.max_chunks is not None:
            # The tokenizer emits each article's chunks consecutively in order.
            chunks_per_article = torch.bincount(article_mapping, minlength=len(articles))
            first_chunk = chunks_per_article.cumsum(0) - chunks_per_article
            slot = torch.arange(len(article_mapping)) - first_chunk[article_mapping]
            keep = slot < self.max_chunks
            inputs = {key: value[keep] for key, value in inputs.items()}
            article_mapping = article_mapping[keep] * slots_per_article + slot[keep]

        embedding_sums = torch.zeros(
            embedding_rows,
            self.model.config.hidden_size,
            device=self.device
        )

        token_counts = torch.zeros(
            embedding_rows, 1, device=self.device
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

                # Add to the article row, or to an individual chunk slot.
                embedding_sums.index_add_(0, mapping, chunk_embeddings)

                token_counts.index_add_(
                    0, mapping, chunk_counts
                )

        # Average the pooled article or each chunk slot, then flatten to one row per article.
        embeddings = embedding_sums / token_counts.clamp_min(1)
        embeddings = embeddings.reshape(len(articles), -1)

        # Normalize the final article embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.to("cpu", copy=True)
