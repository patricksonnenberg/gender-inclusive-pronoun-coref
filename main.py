import itertools as it
import json
import pandas as pd
from rich import print as pprint
import click
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

import skorch as sk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from gensim.models.fasttext import load_facebook_model

import process_dataset as ds
import models as m

# fasttext.util.download_model('en', if_exists='ignore')  # Need to download


def flatten(nested):
    """
    Takes nested list and turns it into a single list
    """
    return list(it.chain.from_iterable(nested))


@click.command()
@click.option(
    "--model-name",
    type=click.Choice(["lstm", "cnn", "rnn"], case_sensitive=False),
    multiple=True,
)
@click.option("--batch-size", default=60)
@click.option("--max-epochs", default=1)
@click.option("--truncate", is_flag=True)
@click.option("--add-entity-tags", is_flag=True)
@click.option("--add-positional-encoding", is_flag=True)
@click.option("--add-pos-tags", is_flag=True)
@click.option("--bidirectional", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--train-file", type=click.Path(readable=True), required=True)
@click.option("--dev-file", type=click.Path(readable=True), required=False)
@click.option("--test-file", type=click.Path(readable=True), required=False)
@click.option("--lr", default=5e-3)
@click.option("--num-layers", default=2)
@click.option("--device", type=click.Choice(["cpu", "cuda"]))
@click.option("--model-args", type=json.loads, default="{}")
@click.option("--num-folds", default=10, type=int)
@click.option("--embedding-source", default=None, type=str)
def main(
    model_name,
    batch_size,
    max_epochs,
    truncate,
    add_entity_tags,
    add_positional_encoding,
    add_pos_tags,
    bidirectional,
    debug,
    train_file,
    dev_file,
    test_file,
    lr,
    num_layers,
    device,
    model_args,
    num_folds,
    embedding_source,
):
    model_args = {
        (k if k.startswith("module__") else f"module__{k}"): v
        if k != "hidden_layer_sizes"
        else [int(hl) for hl in v.split(",")]
        for k, v in model_args.items()
    }
    embedding_dim = 100

    # Using the custom dataset class since it has extra
    # functionality for truncation, adding tags etc.
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
        if "glove" in embedding_source:
            glove_dims = {
                "glove.6B.50d": 50,
                "glove.6B.100d": 100,
                "glove.6B.200d": 200,
                "glove.6B.300d": 300,
            }
            embedding_dim = glove_dims[embedding_source]
            embedding = GloVe(name="6B", dim=embedding_dim, cache="glove/")
            pretrained = embedding.get_vecs_by_tokens(train_set.vocab.get_itos())

        elif "fasttext" in embedding_source:
            embedding = load_facebook_model("cc.en.300.bin")
            itos = train_set.vocab.get_itos()
            pretrained_list = [embedding.wv.get_vector(token) for token in itos]
            pretrained = torch.tensor(pretrained_list)
            embedding_dim = 300

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
    y_dev = torch.stack(flatten([d[1] for d in data_dev]))

    if test_file:
        data_test = [(X_test, y_test) for (X_test, y_test) in loader_test]
        X_test = collate_callable.pad(flatten([d[0] for d in data_test]))
        y_test = torch.stack(flatten([d[1] for d in data_test]))

    list_of_nets = []  # Tracks all models that are called

    if "lstm" in model_name:
        net = sk.NeuralNetClassifier(
            module=m.LSTMTextClassifier,
            module__num_classes=len(train_set.label_vocab),
            module__emb_input_dim=len(train_set.vocab),
            module__emb_output_dim=embedding_dim,
            module__hidden_size=100,
            optimizer=torch.optim.AdamW,
            module__lr=lr,
            module__num_layers=num_layers,
            module__num_folds=num_folds,
            module__bidirectional=bidirectional,
            max_epochs=max_epochs,
            criterion=nn.CrossEntropyLoss,
            module__batch_size=batch_size,
            iterator_train__shuffle=False,
            device=device,
            module__pretrained_vectors=pretrained,
            **model_args,
        )
        if len(model_name) == 1:
            list_of_nets.append(net)
        else:
            lstm_net = net
            list_of_nets.append(("lstm", lstm_net))

    if "rnn" in model_name:
        net = sk.NeuralNetClassifier(
            module=m.RNNTextClassifier,
            module__num_classes=len(train_set.label_vocab),
            module__emb_input_dim=len(train_set.vocab),
            module__emb_output_dim=embedding_dim,
            module__hidden_size=100,
            optimizer=torch.optim.AdamW,
            module__lr=lr,
            module__num_layers=num_layers,
            module__num_folds=num_folds,
            max_epochs=max_epochs,
            criterion=nn.CrossEntropyLoss,
            module__batch_size=batch_size,
            iterator_train__shuffle=False,
            device=device,
            module__pretrained_vectors=pretrained,
            **model_args,
        )
        if len(model_name) == 1:
            list_of_nets.append(net)
        else:
            rnn_net = net
            list_of_nets.append(("rnn", rnn_net))

    if "cnn" in model_name:
        net = sk.NeuralNetClassifier(
            module=m.CNNTextClassifier,
            module__num_classes=len(train_set.label_vocab),
            module__emb_input_dim=len(train_set.vocab),
            module__emb_output_dim=embedding_dim,
            optimizer=torch.optim.AdamW,
            module__lr=lr,
            module__num_folds=num_folds,
            max_epochs=max_epochs,
            criterion=nn.CrossEntropyLoss,
            module__batch_size=batch_size,
            iterator_train__shuffle=False,
            device=device,
            module__pretrained_vectors=pretrained,
            **model_args,
        )
        if len(model_name) == 1:
            list_of_nets.append(net)
        else:
            cnn_net = net
            list_of_nets.append(("cnn", cnn_net))

    if len(list_of_nets) == 1:
        net = list_of_nets[0]
        cv_results = cross_val_score(estimator=net, X=X, y=y, scoring="accuracy")
        cv_results_summary = pd.Series(cv_results).describe()
        pprint(cv_results_summary)
        net.fit(X=X, y=y)
        best_model = net
        dev_score = best_model.score(X=X_dev, y=y_dev)
        dev_pred = best_model.predict(X=X_dev)
        get_percent_correct(y_dev, dev_pred, train_set.label_vocab)
        print(f"Dev accuracy: {dev_score}")
        if test_file:
            test_score = best_model.score(X=X_test, y=y_test)
            print(f"Test accuracy: {test_score}")
        pprint(cv_results)
    else:
        # Training an ensemble of models
        print(f"Training {len(list_of_nets)} models...")
        #  Initializes ensemble model with list of models
        #  Hard voting makes final prediction based on class with most votes
        #  Soft voting makes prediction choosing class with highest probability
        ensemble_net = VotingClassifier(estimators=list_of_nets, voting="soft")
        #  Perform cross validation on the model with training data
        cv_results = cross_val_score(
            estimator=ensemble_net, X=X, y=y, scoring="accuracy"
        )
        cv_results_summary = pd.Series(cv_results).describe()
        pprint(cv_results_summary)
        #  Sets voting strategy
        ensemble_net.set_params(voting="hard")
        #  Fit ensemble model on training data
        ensemble_net.fit(X, y)
        best_model = ensemble_net
        dev_score = best_model.score(X=X_dev, y=y_dev)
        #  Makes predictions on dev data
        dev_pred = best_model.predict(X=X_dev)
        get_percent_correct(y_dev, dev_pred, train_set.label_vocab)
        print(f"Dev accuracy: {dev_score}")
        if test_file:
            test_score = best_model.score(X=X_test, y=y_test)
            print(f"Test accuracy: {test_score}")


def get_percent_correct(y_dev, dev_pred, label_vocab):
    """
    This method calculates the percentage of correct predictions by
    pronoun/gender. However, if 'masculine' is correctly predicted but the
    wrong entity (1 instead of 0) is predicted, that will not be counted as
    correct for that pronoun/gender.
    Parameters
    ----------
    y_dev: The gold labels
    dev_pred: The predictions
    label_vocab: To get the labels/classes

    Returns dict displaying percentage for each pronoun/gender
    -------
    """
    flipped_label_dict = {value: key for key, value in label_vocab.items()}
    y_dev_numpy = y_dev.numpy()  # Tensor -> numpy array
    dict_total_counts = {
        "masculine": 0,
        "feminine": 0,
        "neutral_they": 0,
        "neutral_ze": 0,
        "neutral_xe": 0,
        "neutral_ey": 0,
    }
    dict_correct_counts = {
        "masculine": 0,
        "feminine": 0,
        "neutral_they": 0,
        "neutral_ze": 0,
        "neutral_xe": 0,
        "neutral_ey": 0,
    }
    for i in range(len(dev_pred)):
        gender_gold = (
            (flipped_label_dict[y_dev_numpy[i]]).split("," "")[1].strip()[1:-2]
        )
        dict_total_counts[gender_gold] += 1
        if dev_pred[i] == y_dev_numpy[i]:  # Only if correct entity too
            dict_correct_counts[gender_gold] += 1
    dict_percentages = {}
    for key, value in dict_total_counts.items():
        dict_percentages[key] = round(
            dict_correct_counts[key] / dict_total_counts[key], 4
        )
    print("Percent Correct for Each Pronoun: ", dict_percentages)


if __name__ == "__main__":
    main()
