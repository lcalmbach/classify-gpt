import openai
import streamlit as st
from helper import get_var

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMP = 0.3
DEAFULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 500
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
MODEL_OPTIONS = ["gpt-3.5-turbo", "text-davinci-003"]
SYSTEM_PROMPT_TEMPLATE = """You will be provided with a text together with a list of indexed categories, 
and your task is to assign this text to one to maximum three of the following categories: [{}]
Answer with a list of index of the matching categories for the given answer
if none of the categories apply return -99
example: 
input: <categories: [1: Bildung, 2: Bevölkerung, 3: Arbeit und Erwerb, 4: Energie],
       text: "Wieviele Personen in Basel sind 100 jährig oder älter?"
output: [2]
input: <categories: [1: Bildung, 2: Bevölkerung, 3: Arbeit und Erwerb, 4: Natur und Umwelt],
       text: "Wieviel Erdgas wurde in Basel im Jahr 2022 verbraucht?"
input: <categories: [1: Bildung, 2: Bevölkerung, 3: Arbeit und Erwerb, 4: Natur und Umwelt],
       text: "Wie spät ist es?"
output: [-99]
"""


class Classifier:
    """
    A class for categorizing text data using OpenAI's GPT-3 API.

    Attributes:
        df_texts (pandas.DataFrame): A DataFrame containing the text data to be categorized.
        dict_categories (dict): A dictionary mapping category names to their corresponding IDs.
        category_list_expression (str): A string containing a comma-separated list of category names and IDs.

    Methods:
        run(): Runs the GPT-3 API on each row of the input DataFrame and categorizes the text according to the user's instructions.
    """

    def __init__(self, df_texts, dict_categories, settings: dict = {}):
        df_texts["result"] = [[] for _ in range(len(df_texts.index))]
        self.df_texts = df_texts
        self.settings = settings
        self.dict_categories = dict_categories
        cat_list = []
        for k, v in dict_categories.items():
            cat_list.append(f'{k}: "{v}"')
        self.category_list_expression = ",".join(cat_list)
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            self.category_list_expression
        )
        self.settings = settings
        self.complete_settings()

    def complete_settings(self):
        if "model" not in self.settings:
            self.settings["model"] = DEFAULT_MODEL
        if "temperature" not in self.settings:
            self.settings["temperature"] = DEFAULT_TEMP
        if "max_tokens" not in self.settings:
            self.settings["max_tokens"] = DEFAULT_MAX_TOKENS
        if "top_p" not in self.settings:
            self.settings["top_p"] = DEAFULT_TOP_P
        if "frequency_penalty" not in self.settings:
            self.settings["frequency_penalty"] = DEFAULT_FREQUENCY_PENALTY
        if "presence_penalty" not in self.settings:
            self.settings["presence_penalty"] = DEFAULT_PRESENCE_PENALTY

    def get_completion(self, answer):
        """Generates a response using the OpenAI ChatCompletion API based on
        the given answer.

        Args:
            answer (str): The user's input.

        Returns:
            str: The generated response.

        Raises:
            None
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": answer},
            ],
            temperature=self.settings["temperature"],
            max_tokens=self.settings["max_tokens"],
            top_p=self.settings["top_p"],
            frequency_penalty=self.settings["frequency_penalty"],
            presence_penalty=self.settings["presence_penalty"],
        )
        if "choices" in response:
            response = response.choices[0].message["content"]
        return response

    def run(self):
        """
        Runs the GPT-3 API on each row of the input DataFrame and categorizes
        the text according to the user's instructions.

        Returns:
            pandas.DataFrame: A copy of the input DataFrame with an additional
            'result' column containing the API's categorized output.

        Raises:
            OpenAIError: If there is a problem with the OpenAI API request.
            ValueError: If the 'OPENAI_API_KEY' environment variable is not
            set.
        """
        openai.api_key = get_var("OPENAI_API_KEY")

        for index, row in self.df_texts.iterrows():
            answer = row["text"]
            self.df_texts.loc[index, "result"] = self.get_completion(answer)
            self.df_texts.to_csv("output.csv")
        return self.df_texts
