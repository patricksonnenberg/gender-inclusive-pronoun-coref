from nltk.tag import pos_tag
from csv import DictReader
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchtext as tt
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

import nltk

nltk.download("averaged_perceptron_tagger")

TAB = "\t"


class CorefRelationExtractionDataset(Dataset):
    TSV_COLUMNS = [
        "occupation(0)",
        "other-participant(1)",
        "answer",
        "sentence",
        "gender",
        "answer_person",
        "pronoun",
        "e1_idx",
        "e2_idx",
        "e3_idx",
        "answer_tuple",
    ]

    def __init__(
        self,
        file_path,
        tokenize_fn: Optional[Callable] = None,
        add_entity_tags: bool = False,
        truncate: bool = False,
        add_positional_encoding: bool = False,
        add_pos_tags: bool = False,
        pad_token: str = "<pad>",
        train_dataset: Optional["CorefRelationExtractionDataset"] = None,
    ):
        if not tokenize_fn:

            def tokenize(sentence: str) -> list[str]:
                return sentence.split()

        # Makes the input arguments instance variables
        self.file_path = file_path
        self.column_names = self.TSV_COLUMNS
        self.tokenize = tokenize_fn or tokenize
        self.sentence_column = self.TSV_COLUMNS[3]
        self.relation_column = self.TSV_COLUMNS[10]  # E.g. "('1', 'masculine')"

        self.add_entity_tags = add_entity_tags
        self.truncate = truncate
        self.add_positional_encoding = add_positional_encoding
        self.add_pos_tags = add_pos_tags

        self.e1_idx_column = self.TSV_COLUMNS[7]
        self.e2_idx_column = self.TSV_COLUMNS[8]
        self.e3_idx_column = self.TSV_COLUMNS[9]

        # Tag constants
        self.e1_tag = "<e1>"
        self.e1_tag_close = "</e1>"
        self.e2_tag = "<e2>"
        self.e2_tag_close = "</e2>"
        self.e3_tag = "<e3>"
        self.e3_tag_close = "</e3>"

        self.pad_token = pad_token
        self.unk_token = "<unk>"

        self.data = list(iter(self))

        # If train dataset was passed, taking note of it for vocab purposes
        self.train_dataset = train_dataset

        if not self.train_dataset:
            self.vocab = build_vocab_from_iterator(
                self.tokens, specials=[self.pad_token, self.unk_token]
            )
            self.vocab.set_default_index(self.vocab[self.unk_token])
            self.label_vocab = {rel: idx for idx, rel in enumerate(set(self.relations))}
        else:
            self.vocab = self.train_dataset.vocab
            self.label_vocab = self.train_dataset.label_vocab

    def get_tsv_reader(self, file_in) -> DictReader:
        """
        Reads in TSV file, returns each row as a dict, where keys are the
        column names. It skips the first line, as those are column names.
        """
        next(file_in)  # To skip first line (column names)
        return DictReader(f=file_in, fieldnames=self.column_names, delimiter=TAB)

    def truncate_tokens(
        self, tokens: list[str], e1_idx: int, e3_idx: int
    ) -> tuple[list[str], int, int]:
        """
        Removes tokens that appear before token index of e1 and after token
        index of e3.
        """
        tokens = tokens[e1_idx : (e3_idx + 1)]
        e1_idx, e3_idx = 0, len(tokens) - 1
        return tokens, e1_idx, e3_idx

    def surround_entities_with_tags(
        self, tokens: list[str], e1_idx: int, e2_idx: int, e3_idx: int
    ) -> list[str]:
        """
        Adds tags around each of the three entities with their appropriate
        tags.
        """
        tokens = (
            tokens[:e1_idx]
            + [self.e1_tag, tokens[e1_idx], self.e1_tag_close]
            + tokens[(e1_idx + 1) : e2_idx]
            + [self.e2_tag, tokens[e2_idx], self.e2_tag_close]
            + tokens[(e2_idx + 1) : e3_idx]
            + [self.e3_tag, tokens[e3_idx], self.e3_tag_close]
            + tokens[(e3_idx + 1) :]
        )
        return tokens

    def add_positional_encoding_fn(
        self, tokens: list[str], e1_idx: int, e2_idx: int, e3_idx: int
    ) -> list[str]:
        """
        Adds positional encoding as strings.
        """
        tokens = [str(e1_idx)] + [str(e2_idx)] + [str(e3_idx)] + tokens[:]
        return tokens

    def add_pos(self, tokens: list[str]) -> list[str]:
        """
        Adds POS tags using NLTK.
        """
        tagged = pos_tag(tokens, tagset=None, lang="eng")
        new_tokens = []
        for tok in tagged:  # NLTK puts them in a tuple, e.g. ('word', POS)
            new_tokens.append(tok[0])
            new_tokens.append(tok[1])
        return new_tokens

    def __iter__(self):
        """
        Defining iterator for the TSV file, and for each row it tokenizes
        and handles the text appropriately, adding modified row back to the
        iterator.
        """
        with open(self.file_path) as file_in:
            for d in self.get_tsv_reader(file_in):
                # Handles the entity id conversion
                d[self.e1_idx_column] = int(d[self.e1_idx_column])
                d[self.e2_idx_column] = int(d[self.e2_idx_column])
                d[self.e3_idx_column] = int(d[self.e3_idx_column])
                e1_idx, e2_idx, e3_idx = (
                    d[self.e1_idx_column],
                    d[self.e2_idx_column],
                    d[self.e3_idx_column],
                )

                # Handles tokenization
                sentence = d[self.sentence_column]
                tokens = self.tokenize(sentence)
                del d[self.sentence_column]

                # Indicates the three entities for clarity
                d["entity1"] = tokens[e1_idx]
                d["entity2"] = tokens[e2_idx]
                d["entity3"] = tokens[e3_idx]

                if self.truncate:
                    tokens, e1_idx, e3_idx = self.truncate_tokens(
                        tokens, e1_idx, e3_idx
                    )
                if self.add_entity_tags:
                    tokens = self.surround_entities_with_tags(
                        tokens, e1_idx, e2_idx, e3_idx
                    )
                if self.add_positional_encoding:
                    tokens = self.add_positional_encoding_fn(
                        tokens, e1_idx, e2_idx, e3_idx
                    )
                if self.add_pos_tags:
                    tokens = self.add_pos(tokens)

                d["tokens"] = tokens
                yield d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def relations(self):
        """
        Returns iterator object that generates values of relation column
        attribute from each dict yielded by the iterator
        """
        it = iter(self)
        for d in it:
            yield d[self.relation_column]

    @property
    def tokens(self):
        """
        Returns iterator object that generates values of the tokens key
        from each dict yielded by the iterator.
        """
        it = iter(self)
        for d in it:
            yield d["tokens"]


class CollateCallable:
    """
    An alternative to a collate_fn to make it more modular.
    """

    def __init__(
        self,
        vocab: tt.vocab.Vocab,  # Maps tokens to integers
        label_vocab: dict,  # Maps labels to integers
        pad_value: int = 0,
    ):
        self.vocab = vocab
        self.label_vocab = label_vocab
        self.pad_value = pad_value

    @staticmethod
    def pad(token_indices):
        """
        Pads token indices for each example in the batch using zeros
        """
        return nn.utils.rnn.pad_sequence(
            [torch.tensor(indices) for indices in token_indices], batch_first=True
        )

    def __call__(self, examples):
        """
        Used when instance of class is invoked. Returns two tensors:
        1) token_indices - pads token indices and passes that through vocab
        object to convert to a tensor
        2) labels - maps the relation label to corresponding int value and
        converts list to a tensor
        """
        token_indices = self.pad(
            [self.vocab(example["tokens"]) for example in examples]
        )
        labels = torch.tensor(
            [
                self.label_vocab.get(example["answer_tuple"], self.pad_value)
                for example in examples
            ]
        )
        return token_indices, labels
