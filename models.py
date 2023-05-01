import torch.nn as nn
import torch

def bigrams(some_list: list):
    return zip(some_list, some_list[1:])

class DANClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_layer_sizes: list[int],
        batch_size: int,
        lr: int,
        num_folds: int,
        p_dropout: float = 0.1,
        pretrained_vectors=None
    ):
        super(DANClassifier, self).__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.p_dropout = p_dropout
        self.batch_size = batch_size
        self.lr = lr
        self.num_folds = num_folds
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size, embedding_dim=self.embedding_dim, mode="mean"
        )
        self.fc = nn.Sequential()
        in_out_dims = bigrams([self.embedding_dim] + hidden_layer_sizes)

        for idx, (in_dim, out_dim) in enumerate(in_out_dims):
            self.fc.add_module(
                name=f"{idx}_in{in_dim}_out{out_dim}", module=nn.Linear(in_dim, out_dim)
            )
        self.proj = nn.Linear(hidden_layer_sizes[-1], self.num_classes)
        self.dropout = nn.Dropout(p=self.p_dropout)

        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)

    def forward(self, token_indices: torch.Tensor, *args, **kwargs):
        avg_emb = self.embedding(token_indices)
        out = self.fc(avg_emb)
        out = self.dropout(out)
        out = self.proj(out)
        return out