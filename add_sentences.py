import csv
import pandas as pd


class MakeNewSentences:
    def load_new_sentences(self, input_file):
        """
        This method simply takes in the existing sentence templates from
        the Winogender dataset and creates sentences with six different
        pronouns. These include neopronouns that were not part of the
        original research.
        """
        NOM = "$NOM_PRONOUN"
        POSS = "$POSS_PRONOUN"
        ACC = "$ACC_PRONOUN"

        # Define dictionary of neopronouns
        pronoun_dict = {"masculine": {NOM: "he", POSS: "his", ACC: "him"}}
        pronoun_dict["feminine"] = {NOM: "she", POSS: "her", ACC: "her"}
        pronoun_dict["neutral_they"] = {NOM: "they", POSS: "their", ACC: "them"}
        pronoun_dict["neutral_ze"] = {NOM: "ze", POSS: "zir", ACC: "zir"}
        pronoun_dict["neutral_xe"] = {NOM: "xe", POSS: "xir", ACC: "xem"}
        pronoun_dict["neutral_ey"] = {NOM: "ey", POSS: "eir", ACC: "em"}

        with open(input_file, "r") as tsvfile:  # Opens templates file
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            df = pd.DataFrame(
                columns=["occupation(0)", "other-participant(1)", "answer", "sentence"]
            )
            next(tsvreader)
            for row in tsvreader:
                for key, value in pronoun_dict.items():
                    new_sentence = row[3]
                    if "$NOM_PRONOUN" in new_sentence:
                        # Gets the pronoun in the sentence
                        actual_pronoun_in_sentence = pronoun_dict[key][NOM]
                    elif "$POSS_PRONOUN" in new_sentence:
                        actual_pronoun_in_sentence = pronoun_dict[key][POSS]
                    else:
                        actual_pronoun_in_sentence = pronoun_dict[key][ACC]

                    new_sentence = new_sentence.replace("$OCCUPATION", row[0])
                    new_sentence = new_sentence.replace("$PARTICIPANT", row[1])
                    new_sentence = new_sentence.replace(
                        "$NOM_PRONOUN", pronoun_dict[key][NOM]
                    )
                    new_sentence = new_sentence.replace(
                        "$POSS_PRONOUN", pronoun_dict[key][POSS]
                    )
                    new_sentence = new_sentence.replace(
                        "$ACC_PRONOUN", pronoun_dict[key][ACC]
                    )
                    new_sentence = new_sentence.replace("They was", "They were")
                    new_sentence = new_sentence.replace("they was", "they were")
                    if row[2] == "1":
                        answer_person = row[1]  # Gets occupation of coreferent
                    else:
                        answer_person = row[0]

                    # To find which comes first in order to assign e1 versus e2
                    occupation_position = new_sentence.split().index(row[0])
                    participant_position = new_sentence.split().index(row[1])
                    if occupation_position < participant_position:
                        e1_indx = str(occupation_position)
                        e2_indx = str(participant_position)
                    else:
                        e1_indx = str(participant_position)
                        e2_indx = str(occupation_position)
                    # To find index of pronoun in the sentence
                    e3_indx = str(
                        new_sentence.split().index(actual_pronoun_in_sentence)
                    )

                    answer_tuple = str((row[2], key))

                    answer_person = answer_person.lower()
                    row_to_add = {
                        "sentence": new_sentence,
                        "answer": row[2],
                        "occupation(0)": row[0],
                        "other-participant(1)": row[1],
                        "gender": key,
                        "answer_person": answer_person,
                        "pronoun": actual_pronoun_in_sentence,
                        "e1_idx": e1_indx,
                        "e2_idx": e2_indx,
                        "e3_idx": e3_indx,
                        "answer_tuple": answer_tuple,
                    }
                    df = pd.concat(
                        [df, pd.DataFrame(row_to_add, index=[0])], ignore_index=True
                    )
            df.to_csv("new_sentences_updated.csv", index=False)
            df.to_csv("new_sentences_updated.tsv", index=False, sep="\t")


if __name__ == "__main__":
    input_file = "data/templates.tsv"  # Using pre-existing templates
    new_sentences = MakeNewSentences().load_new_sentences(input_file)
