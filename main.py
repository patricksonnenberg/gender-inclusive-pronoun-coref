import argparse
import itertools as it
import json
import add_sentences
import test_stanford_corenlp
# import test_coreferee_spacy

import pandas as pd
from rich import print as pprint
import click
import skorch as sk
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from skorch.callbacks import EpochScoring
from skorch.dataset import ValidSplit
from skorch.helper import predefined_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from torchtext.vocab import build_vocab_from_iterator

from transformers import AutoTokenizer

from torchtext.vocab import pretrained_aliases

import process_dataset as ds
import models as m

import numpy as np

def flatten(nested):
    """
    Takes nested list and turns it into a single list
    """
    return list(it.chain.from_iterable(nested))

@click.command()
@click.option("--model-name", type=click.Choice(["dan", "lstm", "cnn"]))
@click.option("--batch-size", default=60)
@click.option("--max-epochs", default=1)
@click.option("--truncate", is_flag=True)
@click.option("--add-entity-tags", is_flag=True)
@click.option("--add-positional-encoding", is_flag=True)
@click.option("--add-pos-tags", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--train-file", type=click.Path(readable=True), required=True)
@click.option("--dev-file", type=click.Path(readable=True), required=False)
@click.option("--test-file", type=click.Path(readable=True), required=False)
@click.option("--hidden-layer-sizes", default="200,100")
@click.option("--embedding-dim", default=300)
@click.option("--lr", default=5e-3)
@click.option("--device", type=click.Choice(["cpu", "cuda"]))
@click.option("--model-args", type=json.loads, default="{}")
@click.option("--cross-validation-args", type=json.loads, default="{}")
@click.option("--grid-search", is_flag=True)
@click.option("--random-search", is_flag=True)
@click.option("--num-folds", default=10, type=int)
@click.option("--num-parallel-jobs", default=None, type=int)
@click.option("--embedding-source", default=None, type=str)

def main(
    model_name,
    batch_size,
    max_epochs,
    truncate,
    add_entity_tags,
    add_positional_encoding,
    add_pos_tags,
    debug,
    train_file,
    dev_file,
    test_file,
    hidden_layer_sizes,
    embedding_dim,
    lr,
    device,
    model_args,
    cross_validation_args,
    grid_search,
    random_search,
    num_folds,
    num_parallel_jobs,
    embedding_source,
):

    model_args = {
        (k if k.startswith("module__") else f"module__{k}"): v
        if k != "hidden_layer_sizes"
        else [int(hl) for hl in v.split(",")]
        for k, v in model_args.items()
    }

    cross_validation_args = {
        (k if k.startswith("module__") else f"module__{k}"): v
        if k != "hidden_layer_sizes"
        else [[int(x) for x in sz.split(",")] for sz in v]
        for k, v in cross_validation_args.items()
    }

    # We'll use the custom dataset class since it has
    # extra functionality for truncation, adding tags etc.
    train_set = ds.CorefRelationExtractionDataset(
        file_path=train_file,
        truncate=truncate,
        add_entity_tags=add_entity_tags,
        add_positional_encoding=add_positional_encoding,
        add_pos_tags=add_pos_tags,
    )
    dev_set = ds.CorefRelationExtractionDataset(
        file_path=dev_file,
        truncate=truncate,
        add_entity_tags=add_entity_tags,
        add_positional_encoding=add_positional_encoding,
        add_pos_tags=add_pos_tags,
    )
    if test_file:
        test_set = ds.CorefRelationExtractionDataset(
            file_path=test_file,
            truncate=truncate,
            add_entity_tags=add_entity_tags,
            add_positional_encoding=add_positional_encoding,
            add_pos_tags=add_pos_tags,
        )

    pretrained = None
    if embedding_source:  # To handle Glove embeddings
      embedding = pretrained_aliases[embedding_source]()

      # These embedding objects have a .get_vecs_by_tokens method which gets
      # embeddings. (vocab.get_itos() gives a int -> str map)
      pretrained = embedding.get_vecs_by_tokens(train_set.vocab.get_itos())

      # In case loading pretrained embeddings, we'll get the dimensionality
      # from the vectors themselves
      embedding_size = pretrained.size()[-1]

    # Only need one callable to have ability to pad sequences
    collate_callable = ds.CollateCallable(
        vocab=train_set.vocab, label_vocab=train_set.label_vocab
    )

    # Wraps everything in a DataLoader and gets the data
    loader_train = DataLoader(
        dataset=train_set, batch_size=batch_size, collate_fn=collate_callable
    )
    loader_dev = DataLoader(
        dataset=dev_set, batch_size=batch_size, collate_fn=collate_callable
    )
    if test_file:
        loader_test = DataLoader(
            dataset=test_set, batch_size=batch_size, collate_fn=collate_callable
        )

    # All vectorized data handled
    data = [(X, y) for (X, y) in loader_train]
    X = collate_callable.pad(flatten([d[0] for d in data]))
    y = torch.stack(flatten([d[1] for d in data]))

    data_dev = [(X_dev, y_dev) for (X_dev, y_dev) in loader_dev]
    X_dev = collate_callable.pad(flatten([d[0] for d in data_dev]))
    print("X DEV")
    print(len(data_dev))
    y_dev = torch.stack(flatten([d[1] for d in data_dev]))
    if test_file:
        data_test = [(X_test, y_test) for (X_test, y_test) in loader_test]
        X_test = collate_callable.pad(flatten([d[0] for d in data_test]))
        y_test = torch.stack(flatten([d[1] for d in data_test]))


    if model_name=="dan":
        # fill model_args with default values
        for arg, val in [
            ("module__hidden_layer_sizes", hidden_layer_sizes),
            ("module__embedding_dim", embedding_dim),
        ]:
            if arg not in model_args and arg not in cross_validation_args:
                model_args[arg] = (
                    val
                    if arg != "module__hidden_layer_sizes"
                    else [int(hl) for hl in val.split(",")]
                )
        hidden_layer_sizes = [int(hl) for hl in hidden_layer_sizes.split(",")]

        net = sk.NeuralNetClassifier(
            module=m.DANClassifier,
            module__num_classes=len(train_set.label_vocab),
            module__vocab_size=len(train_set.vocab),
            module__batch_size=batch_size,
            optimizer=torch.optim.AdamW,
            module__lr=lr,
            module__num_folds=num_folds,
            max_epochs=max_epochs,
            criterion=nn.CrossEntropyLoss,
            batch_size=batch_size,
            iterator_train__shuffle=False,
            device=device,
            module__pretrained_vectors=pretrained,
            **model_args,
        )








    # Rewrote grid search below, initializes cv object
    if grid_search:
        cv = GridSearchCV(
            estimator=net,
            param_grid=cross_validation_args,
            refit=True,
            cv=num_folds,
            n_jobs=num_parallel_jobs,
        )
        cv_results = cv.fit(X=X, y=y)
        best_model = cv.best_estimator_
        dev_score = best_model.score(X=X_dev, y=y_dev)
        test_score = best_model.score(X=X_test, y=y_test)  # To report the test score
        print("The Best Hyperparamters: ", cv.best_params_)
    elif random_search:  # Did not really utilize since grid search was more effective
        cv = RandomizedSearchCV(
            estimator=net,
            n_iter=10,
            param_distributions=cross_validation_args,
            refit=True,
            cv=num_folds,
            n_jobs=num_parallel_jobs,
        )
        cv_results = cv.fit(X=X, y=y)
        best_params = cv.best_params_
        best_model = net.set_params(**best_params) # create a new model object with the best hyperparameters
        dev_score = best_model.score(X=X_dev, y=y_dev)
        test_score = best_model.score(X=X_test, y=y_test)
        print("The Best Hyperparamters: ", cv.best_params_)
    else:
        # Here we just want a CV score
        cv_results = cross_val_score(estimator=net, X=X, y=y, scoring="accuracy")
        cv_results_summary = pd.Series(cv_results).describe()
        pprint(cv_results_summary)
        net.fit(X=X, y=y)
        best_model = net
        dev_score = best_model.score(X=X_dev, y=y_dev)
        dev_pred = best_model.predict(X=X_dev)
        print("Dev gold labels:")
        print(y_dev)
        print("Dev predicted labels:")
        print(dev_pred)
        if test_file:
            test_score = best_model.score(X=X_test, y=y_test)

    pprint(cv_results)

    print(f"Dev accuracy: {dev_score}")
    if test_file:
        print(f"Test accuracy: {test_score}")






if __name__ == "__main__":
    # if __name__ == '__main__':
    #     add_sentences
    #     test_stanford_corenlp
    #     test_coreferee_spacy
    main()