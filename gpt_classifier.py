import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from helper import get_var


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

    def __init__(self, df_texts, dict_categories):
        df_texts["result"] = [[] for _ in range(len(df_texts.index))]
        self.df_texts = df_texts
        self.dict_categories = dict_categories
        cat_list = []
        for k, v in dict_categories.items():
            cat_list.append(f'{k}: "{v}"')
        self.category_list_expression = ",".join(cat_list)

    def run(self):
        """
        Runs the GPT-3 API on each row of the input DataFrame and categorizes the text according to the user's instructions.

        Returns:
            pandas.DataFrame: A copy of the input DataFrame with an additional 'result' column containing the API's categorized output.

        Raises:
            OpenAIError: If there is a problem with the OpenAI API request.
            ValueError: If the 'OPENAI_API_KEY' environment variable is not set.
        """
        openai.api_key = get_var("OPENAI_API_KEY")
        llm = ChatOpenAI(temperature=0.5)
        prompt = ChatPromptTemplate.from_template(
            """Weise die Antwort ###{answer}### einer bis drei der folgenden Kategorien zu: [{category_list_expression}]

            Formattiere die Antwort wie folgt: Gib eine Liste von Ids der zugehörigen Kategorien zurück, Beipiel: [7, 1]
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        for index, row in self.df_texts.head(20).iterrows():
            answer = row["text"]
            response = chain.run(
                {
                    "answer": answer,
                    "category_list_expression": self.category_list_expression,
                }
            )
            # response = json.loads(response)
            self.df_texts.loc[index, "result"] = response
            self.df_texts.to_csv("output.csv")
        return self.df_texts
