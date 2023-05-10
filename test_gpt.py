import openai
import csv

#  Add your personal openai api key here:
openai.api_key = ""

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


class GenerateGPTPredictions:
    def load_data(self, file_name):
        """
        Reads in the dataset to iterate over each sentence. It returns the
        sentences.
        """
        with open(file_name, "r") as f:
            next(f)  # Skip column titles
            reader = csv.reader(f, delimiter="\t")
            sentences = [row for row in reader]
        return sentences

    def create_prompt(self, sentence):
        return f"""
        You are an AI assistant designed to identify people, 
        occupations, and pronouns in a sentence. Your job is to analyze the 
        following sentence, and identify the participants/people in the 
        sentence and the pronoun in the sentence. You must also identify 
        which of the two people the pronoun corefers to.

        Sentence: {sentence}
        
        Return three things:
        1. A list of the people/participants in the sentence.
        2. The pronoun in the sentence, if present, as a string.
        3. The person/participant that the pronoun corefers to, as a string.

        Example output: people: ["chef", "customer"], pronoun: "he", person: "chef"
        """

    def process_sentences(self, sentences, engine):
        """
        Iterates over each sentence, prompts GPT for a response, and prints
        the response (or throws an error).
        """
        # Process each sentence in the dataset
        for sentence in sentences:
            actual_sentence = sentence[TSV_COLUMNS.index("sentence")]

            # Create the prompt for the GPT model
            prompt = self.create_prompt(actual_sentence)

            try:
                # Call the GPT model with the prompt
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=50,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )

                # Print the people, pronoun, and coreferent (in theory)
                print(response["choices"][0]["text"])
            except Exception as e:
                print(
                    f"Error processing this sentence: "
                    f"{actual_sentence}\nError message: {e}"
                )


if __name__ == "__main__":
    gpt = GenerateGPTPredictions()
    sentences = gpt.load_data("new_sentences_updated.tsv")
    gpt.process_sentences(sentences, "text-davinci-002")
    # Possible engines include: "text-davinci-002", "curie", "ada", "babbage"
