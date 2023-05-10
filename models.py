import torch.nn as nn
import torch


def bigrams(some_list: list):
    return zip(some_list, some_list[1:])


class LSTMTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    hidden_size : int
        Dimension size for hidden states within the LSTM
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    batch_size : int, default 64
        Batch size
    lr : float, default 0.005
        Learning rate
    num_layers : int, default 2
        Number of layers
    num_folds : int, default 10
        Number of folds
    bidirectional : boolean, default False
        Specifies if bidirectional or unidirectional LSTM
    pretrained_vectors : default None
        Specifies if pretrained vectors
    """

    def __init__(
        self,
        emb_input_dim=0,
        emb_output_dim=0,
        hidden_size=100,
        num_classes=2,
        dr=0.2,
        batch_size=64,
        lr=0.005,
        num_layers=2,
        num_folds=10,
        bidirectional=False,
        pretrained_vectors=None,
    ):
        super(LSTMTextClassifier, self).__init__()

        self.num_layers = num_layers
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        self.my_lil_lstm = nn.LSTM(
            input_size=emb_output_dim,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
        )
        self.maxpool_layer = torch.nn.AdaptiveMaxPool1d(1)
        self.dropout_layer = nn.Dropout(dr)
        self.batch_size = batch_size
        self.lr = lr
        self.num_folds = num_folds
        if bidirectional:  # Size is doubled due to concatenated vectors
            self.projection_layer = torch.nn.Linear(2 * hidden_size, num_classes)
        else:
            self.projection_layer = torch.nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        self.apply(self._init_weights)

        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for i in range(self.num_layers):
                torch.nn.init.xavier_uniform_(module.all_weights[i][0])
                torch.nn.init.xavier_uniform_(module.all_weights[i][1])
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded):
        embedding = torch.permute(embedded, (1, 0, 2))  # E.g. [100, 32, 128]
        embedding, _ = self.my_lil_lstm(embedding)  #  [32, 128, 100]
        embedding = torch.permute(embedding, (1, 2, 0))  # [32, 100, 128]
        embedding = self.maxpool_layer(embedding)  # [32, 100, 1]
        embedding = embedding.squeeze(2)  # [32, 100]
        embedding = self.dropout_layer(embedding)
        embedding = self.projection_layer(embedding)
        return embedding

    def forward(self, data):
        embedded = self.embedding(data)
        return self.from_embedding(embedded)


class CNNTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 100
        Number of filters for each width
    num_conv_layers : int, default = 3
        Number of convolutional layers (conv + pool)
    batch_size : int, default = 64
        Batch size
    lr : float, default = 0.0005
        Learning rate
    num_folds : int, default = 10
        Number of folds
    pretrained_vectors : default None
        Specifies if pretrained vectors
    intermediate_pool_size: int, default = 3
    """

    def __init__(
        self,
        emb_input_dim=0,
        emb_output_dim=0,
        num_classes=2,
        dr=0.2,
        filter_widths=[3, 4],
        num_filters=100,
        num_conv_layers=3,
        batch_size=64,
        lr=0.0005,
        num_folds=10,
        pretrained_vectors=None,
        intermediate_pool_size=3,
        **kwargs
    ):
        super(CNNTextClassifier, self).__init__(**kwargs)
        self.emb_input_dim = emb_input_dim

        self.emb_output_dim = emb_output_dim
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)

        self.num_classes = num_classes
        self.dr = dr
        self.batch_size = batch_size
        self.lr = lr
        self.num_folds = num_folds
        self.filter_widths = filter_widths
        self.num_filters = num_filters
        self.num_conv_layers = num_conv_layers
        self.intermediate_pool_size = intermediate_pool_size

        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)

        list_of_sequentials = []
        for each_width in self.filter_widths:
            this_sequence = nn.Sequential()  # adding sequential for each width
            for each_layer in range(self.num_conv_layers):
                this_sequence.append(nn.Dropout(self.dr))
                if each_layer == 0:  # size differs for very first layer
                    this_sequence.append(
                        nn.Conv1d(self.emb_output_dim, self.num_filters, each_width)
                    )
                else:
                    this_sequence.append(
                        nn.Conv1d(self.num_filters, self.num_filters, each_width)
                    )
                this_sequence.append(nn.ReLU())
                # use adaptive max pool if the last layer, otherwise max pool
                if each_layer == self.num_conv_layers - 1:
                    this_sequence.append(nn.AdaptiveMaxPool1d(1))
                else:
                    this_sequence.append(
                        nn.MaxPool1d(self.intermediate_pool_size, stride=1)
                    )
            # Append to the list:
            list_of_sequentials.append(
                this_sequence
            )  # add it to the list of sequentials
        self.module_list = nn.ModuleList(list_of_sequentials)
        self.projection_layer = nn.Linear(
            (num_filters * len(filter_widths)), self.num_classes
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded):
        embedded = torch.permute(embedded, (0, 2, 1))  # E.g. [32, 50, 128]
        results_from_module_list = []
        for each_layer in self.module_list:
            results_from_module_list.append(each_layer(embedded))
        concatted_list = torch.cat(results_from_module_list, dim=1)  # [32, 200, 1]
        concatted_list = concatted_list.squeeze()  # [32, 200]
        final_projection = self.projection_layer(concatted_list)  # [32, 4]
        return final_projection  # batch size by num classes

    def forward(self, data):
        embedded = self.embedding(data)
        return self.from_embedding(embedded)


class RNNTextClassifier(nn.Module):
    """
    Parameters
    ----------
    num_classes : int
        Number of categories in classifier output
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    hidden_size : int
        Dimension size for hidden states
    lr : float
        Learning rate
    num_layers : int
        Number of layyers
    num_folds : int
        Number of folds
    batch_size : int
        Batch size
    pretrained_vectors : boolean
        Specifies if pretrained vectors
    """

    def __init__(
        self,
        num_classes,
        emb_input_dim,
        emb_output_dim,
        hidden_size,
        lr,
        num_layers,
        num_folds,
        batch_size,
        pretrained_vectors,
    ):
        super(RNNTextClassifier, self).__init__()

        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        self.embedding.weight.requires_grad = False
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lr = lr
        self.num_folds = num_folds
        self.batch_size = batch_size

        self.rnn = nn.RNN(emb_output_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)

    def forward(self, x):
        x = self.embedding(x)
        # Initialize hidden state for RNN
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        x, _ = self.rnn(x, hidden)
        # Use the output state of the last time step for classification
        x = x[:, -1]
        x = self.fc(x)
        return x
