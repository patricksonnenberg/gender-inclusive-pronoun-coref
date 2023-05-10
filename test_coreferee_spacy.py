import coreferee, spacy
import re
import csv
import pandas as pd

#  Note: will need to run:
#!pip install spacy-transformers
#!python3 -m pip install coreferee
#!python3 -m coreferee install en
#!python -m spacy download en_core_web_lg
#!python -m spacy download en_core_web_trf

nlp = spacy.load("en_core_web_sm")  # Change for lg or trf
nlp.add_pipe("coreferee")


class Coreferee:
    def get_corefs(self, sentence):
        """
        This method runs a sentence through the Coreferee to obtain the
        coreferences present in each sentence. A list of these predicted
        coreferences is returned.
        """
        doc = nlp(sentence)
        doc._.coref_chains.print()
        if len(doc._.coref_chains) > 0:
            result = doc._.coref_chains[0].pretty_representation
            #  Cleans up predictions to meet my format
            split_item = re.split(":|\(|\)", result)
            stripped_output = [item.strip() for item in split_item if item.strip()]
            entity_list = []
            for item in stripped_output:
                item = item.strip(", ").strip()
                if item and len(item) > 1:
                    entity_list.append(item)
            return entity_list
        return

    def compare_results(self, input_file):
        """
        This method gets the predicted coreferents for each sentence and compares them
        to the actual gold coreferents. They are both added to a new file as well, and
        this method counts the correct number of predictions for each gender/pronoun.
        """
        sentence_count = 0
        total_correct_count = 0
        correct_predictions = {
            "masculine": 0,
            "feminine": 0,
            "neutral_they": 0,
            "neutral_ze": 0,
            "neutral_xe": 0,
            "neutral_ey": 0,
        }
        with open(input_file, "r") as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            df = pd.DataFrame(columns=["prediction_coref", "gold_corefs", "sentence"])
            next(tsvreader)
            for row in tsvreader:
                gold_list = []
                sentence = row[0]
                correct_referent = row[5]
                correct_pronoun = row[6]
                gold_list.append(correct_referent)
                gold_list.append(correct_pronoun)
                predicted_list = self.get_corefs(sentence)
                row_to_add = {
                    "prediction_coref": str(predicted_list),
                    "gold_corefs": str(gold_list),
                    "sentence": sentence,
                }
                df = pd.concat(
                    [df, pd.DataFrame(row_to_add, index=[0])], ignore_index=True
                )
                sentence_count += 1
                gender = row[4]
                if predicted_list == gold_list:  # Comparing predicted and gold
                    total_correct_count += 1
                    correct_predictions[gender] += 1
            df.to_csv("coreferee_predictions_and_gold.csv", index=False)
            df.to_csv("coreferee_predictions_and_gold.tsv", index=False, sep="\t")


if __name__ == "__main__":
    compare = Coreferee().compare_results("new_sentences_updated.tsv")
