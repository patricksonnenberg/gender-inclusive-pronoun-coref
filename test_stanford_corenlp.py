import requests
import json
import csv
import pandas as pd

class Stanford():
    def get_corefs(self, sentence):
        """
        This method runs a sentence through the CoreNLP server to obtain the coreferences present in
        each sentence. A list of these coreferences is returned.
        """
        url = 'https://corenlp.run/'  # URL for the CoreNLP server
        payload = {'outputFormat': 'json', 'annotators': 'coref', 'input': sentence}  # Defines POST request
        response = requests.post(url, data=payload)  # Sends POST request to CoreNLP server
        corenlp_dict = json.loads(response.content)
        corefs_dict = corenlp_dict["corefs"]  # Access "corefs" key in the dictionary

        corefs_list = []  # Loops over keys in the corefs dict to access the coref chain
        for key in corefs_dict.keys():
            coref_chain = corefs_dict[key]
            for coref_mention in coref_chain:
                corefs_list.append(coref_mention["text"].lower().replace("the ", ""))
        return corefs_list

    def compare_results(self, input_file):
        """
        This method gets the predicted coreferents for each sentence and compares them
        to the actual gold coreferents. They are both added to a new file as well, and
        this method counts the correct number of predictions for each gender/pronoun.
        """
        sentence_count = 0
        total_correct_count = 0
        correct_predictions = {"masculine": 0, "feminine": 0, "neutral_they": 0, "neutral_ze": 0, "neutral_xe": 0, "neutral_ey": 0}
        with open(input_file, 'r') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            df = pd.DataFrame(columns=['prediction_coref', 'gold_corefs', 'sentence'])
            next(tsvreader)
            for row in tsvreader:
                gold_list = []
                sentence = row[0]
                correct_referent = row[5]
                correct_pronoun = row[6]
                gold_list.append(correct_referent)
                gold_list.append(correct_pronoun)
                predicted_list = self.get_corefs(sentence)
                row_to_add = {'prediction_coref': str(predicted_list),
                              'gold_corefs': str(gold_list),
                              'sentence': sentence}
                df = pd.concat([df, pd.DataFrame(row_to_add, index=[0])], ignore_index=True)
                sentence_count += 1
                set_correct = set(gold_list)  # Convert to sets for comparison
                set_predict = set(predicted_list)
                if len(set_predict) == 2:
                    gender = row[4]
                    if set_predict == set_correct:  # Comparing predicted and gold
                        total_correct_count += 1
                        correct_predictions[gender] += 1
            df.to_csv('predictions_and_gold.csv', index=False)
            df.to_csv('predictions_and_gold.tsv', index=False, sep='\t')

if __name__ == "__main__":
    compare = Stanford().compare_results('new_sentences.tsv')