## Gender-Inclusive Pronoun Coreference Resolution

This project focuses on the incorporation of singular gender-neutral 
they/them and neopronouns in coreference resolution, and broadly does the 
following: 
1. Defines gender and examines gender in coreference resolution 
   research (found in the PDF)
2. Adapts sentence templates to create a new dataset
3. Tests performance of two existing coreference models (CoreNLP and Coreferee)
   on the new dataset
4. Builds four new models - LSTM, BiLSTM, CNN, RNN
5. Adds additional features and tunable hyperparameters, such as adding 
   entity tags, adding positional encoding, adding parts of speech tags, 
   truncating, batch size,
   learning rate
6. Uses randomly initialized, GloVe, and FastText word embeddings
7. Performs experiments to find the best performing model
8. Evaluates the results


<br />



### 🧑‍💻 Running the Code 🧑‍💻
Simply run the code from the terminal. All flags are found in `main.py`. At 
minimum, it's required to provide a train file, dev file, and a model name. 
However, one can also specify other parameters such as batch size, learning 
rate, adding entity tags, etc. 

Note: I do not include the training and development files, as they are 
originally influenced by sentence templates from [Rudinger et 
al., 2018](https://github.com/rudinger/winogender-schemas), but I explain 
more how to create these files in Step 2. 

A generic command would be:

`python3 main.py --train-file data_split/train_all_six.tsv  --dev-file 
data_split/dev_all_six.tsv --max-epochs 10 --model-name lstm
`

The command that produced the highest accuracies for me was:

`python3 main.py --train-file data_split/train_all_six.tsv  --dev-file 
data_split/dev_all_six.tsv --max-epochs 20 --model-name lstm 
--bidirectional --embedding-source glove.6B.100d --truncate --batch-size 32`


<br />

### Step 1: Examining Gender
`coreference_resolution_paper.pdf` defines gender and examines gender in 
coreference resolution 
research. It also describes the entire study and contains all results. 

<br />

### Step 2: Create Dataset
Note: to reproduce this step, one must download the templates from the link 
below.

The dataset is constructed from Winograd schema templates from [Rudinger et 
al., 2018](https://github.com/rudinger/winogender-schemas). Their data 
contained sentences with two entities and one pronoun that refers to one of 
the entities. For example, 
1. **The nurse** notified the patient that...
   1. **her** shift would be ending in an hour.
   2. **his** shift would be ending in an hour.
   3. **their** shift would be ending in an hour.

where the two entities are "nurse" and "patient." I create six sentences 
from each template using six different pronouns:

Pronoun | Nominative | Accusative | Possessive | 
--- | --- | --- | --- |
masculine | he | him | his | --- | --- | --- | --- |
feminine | she | her | her | --- | --- | --- | --- |
neutral they | they | them | their | --- | --- | --- | --- |
neutral ze | ze | zir | zir | --- | --- | --- | --- |
neutral xe | xe | xem | xir | --- | --- | --- | --- |
neutral ey | ey | em | eir | 

`data/templates.tsv` is the file with the templates. I do not include it 
here, but it can be downloaded from the link above. 

`add_sentences.py` reads in the templates and produces the new sentences. 

`new_sentences_updated.py` is written by the previous file. This new file 
also contains useful metadeta in each column, including the 
entities. It also adds gold labels in the format ('1', 'neutral_they'). The 
number refers to entity 0 or entity 1. 

`new_sentences_small.tsv` is included as an example of the file that was 
created with metadata. 

`data_split/` folder contains splits that I manually created for training 
and development sets, and there was not enough data to create a test set as 
well. I do not include them here, but the files are in the exact same 
format as `new_sentences_updated.py`. There are four train files: one that 
only contains binary pronouns; one that includes binary and they/them; one 
that includes binary, they/them, and ze; and one that includes all six 
pronouns. There are two development files: one that includes binary and 
they/them pronouns; one that includes all six pronouns. No sentences from 
the training data are found in the development data. I can provide the 
files upon request. 

<br />

### Step 3: Test performance on existing coreference models
`test_stanford_corenlp.py` runs each sentence through the CoreNLP server to 
obtain the coreferences, which are returned. The code obtains the output 
and compares it to the gold label. 

`model_predictions/corenlp_predictions_and_gold.tsv` will be created when 
the above code is run. There is a column for the prediction, one for the 
gold label, and one for the sentence itself. 

`test_coreferee_spacy.py` runs each sentence through Coreferee to obtain 
the coreferences, which are returned. The code compares the output to the 
gold label. Coreferee uses spaCy, so the code can be changed to use small, 
large, or transformer pipeline. 

`model_predictions/coreferee_sm_predictions_and_gold.tsv` will be created 
when the above code is run. There is a column for the prediction, one for the 
gold label, and one for the sentence itself. 


<br />


### Step 4: Build new models
`models.py` builds three models: LSTM, CNN, and RNN. The LSTM class also 
handles bidirectional LSTMs (BiLSTM), which I consider to be a separate 
model. These classes are called from `main.py`.

`main.py` calls on both `models.py` and `process_dataset.py` to 
appropriately process the train and dev sets, handle embeddings, initialize 
the model(s), and train and test the models. The results are printed to the 
terminal, as is a dictionary that contains accuracy scores for each 
individual pronoun:

Percent Correct for Each Pronoun:  {'masculine': 0.5417, 'feminine': 0.5833, 'neutral_they': 0.5417, 'neutral_ze': 0.5833, 'neutral_xe': 0.5833, 'neutral_ey': 0.5417}


<br />


### Step 5: Add features and hyperparameters
`process_dataset.py` reads in and iterates over the data and processes it 
appropriately, and it can handle truncating the tokens, surrounding the 
entities with tags, adding positional encoding, and adding POS tags. 

<br />


### Step 6: Add word embeddings
Note: the embeddings will need to be downloaded separately. The line to 
download FastText is found at the top of `main.py`. 

GloVe embeddings can be downloaded [here](https://nlp.stanford.edu/projects/glove/).

`glove/{embeddings}` is where the GloVe embeddings should be saved. The 
code can handle 50d, 100d, 200d, and 300d embeddings. 

`cc.en.300.bin` is created when FastText embeddings are downloaded. The 
command to do so is found at the top of `main.py`.


<br />

### Step 7: Perform experiments
`test_gpt.py` was used to try and prompt OpenAI's GPT with this task, but 
it performed too poorly, and often only contained garbage output. 

There are five main experiments: 
1. The first tests each model against each 
set of training data. 
2. The next tests each model against the additional 
features to see if any improve performance. 
3. The third tests each model 
against various embedding types. 
4. The fourth tests each model being 
combined with other models to make the final predictions. 
5. Finally, the 
fifth tests the highest scoring models being combined with all features 
that led to increased accuracies. 

All experiments and trainings are run 
for 20 epochs and using GPU and Google Colab. All scores reported are 
accuracy scores on the dev set. 

<br />


### Step 8: Evaluate results
`coreference_resolution_paper.pdf` contains the results. 

The experiments show that more inclusive training data leads to better 
results on test data that are less likely to erase certain gender 
identities. The existing coreference models I tested could not even  detect 
neopronouns. The pretrained embeddings  performed especially poorly when 
only trained on binary pronouns, showing that those embeddings are also erasing certain identities. The (Bi)LSTM consistently scored the highest, likely because they are designed to handle long-term dependencies across sequential data
