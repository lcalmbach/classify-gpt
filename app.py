import streamlit as st
import pandas as pd
from gpt_classifier import Classifier

__version__ = "0.0.1"
__author__ = "Lukas Calmbach"
__author_email__ = "lcalmbach@gmail.com"
VERSION_DATE = "2023-06-18"
APP_NAME = "ClassifyGPT"
GIT_REPO = "https://github.com/lcalmbach/classify-gpt"

MAX_RECORDS = 10
DEMO_CATEGORY_FILE = "./categories.xlsx"
DEMO_TEXT_FILE = "./2013_F13.xlsx"
APP_INFO = f"""<div style="background-color:powderblue; padding: 10px;border-radius: 15px;">
    <small>App created by <a href="mailto:{__author_email__}">{__author__}</a><br>
    version: {__version__} ({VERSION_DATE})<br>
    <a href="{GIT_REPO}">git-repo</a><br>
    This app is built using <a href="https://docs.streamlit.io/">Streamlit</a>,
    the <a href="https://platform.openai.com/docs/introduction">openAI API</a>
    and <a href="https://python.langchain.com/docs/get_started/introduction.html">LangChain</a>
    """


def show_info():
    """
    This function displays information about the application and the required input format.
    The info is shown in an expander widget.
    :return: None
    """

    with st.expander("Info"):
        text = f"""The {APP_NAME} application provides a convenient solution 
        for assigning predefined categories to a given list of short texts. 
        To utilize this functionality, you need to provide two types of 
        inputs in MS Excel (xlsx) format: a list of "id, text" records 
        and a list of "id, category" records.

In the demo mode, you can explore the application without providing any 
inputs. A sample dataset is provided, allowing you to experience the 
classification process firsthand. However, please note that in the 
current version, the number of records that can be classified is currently 
limited to {MAX_RECORDS} randomly selected from the full dataset. This 
limitation is implemented to manage costs effectively.

With {APP_NAME}, you can streamline the categorization process and 
effortlessly assign categories to your short texts, simplifying your 
workflow and saving valuable time."""
        st.write(text)


def preview_input(df_texts, dic_categories):
    with st.expander("Texts:", expanded=True):
        st.write(df_texts)

    with st.expander("Categories:", expanded=True):
        st.write(dic_categories)


def get_uploaded_files():
    """
    This function allows the user to upload two Excel files - one with expressions and one with categories.
    If the user uploads an Excel file with expressions, the function loads the file into a pandas DataFrame and
    drops any rows with missing values. It then sets the index to the 'id' column and returns the resulting DataFrame.

    If the user uploads an Excel file with categories, the function loads the file into a pandas DataFrame,
    renames the columns to 'id' and 'category', and returns a dictionary of category IDs and category names.

    If either file is not uploaded, an empty dictionary and an empty DataFrame are returned.

    Returns:
        df_texts (pandas.DataFrame): DataFrame containing the loaded expressions Excel file (if any).
        dic_categories (dict): Dictionary containing the ID-category mapping from the loaded categories Excel file (if any).
    """

    uploaded_file = st.file_uploader(
        "Upload Excel file with expressions", type=["xlsx"]
    )
    uploaded_categories = st.file_uploader(
        "Upload xlsx file with categories", type=["xlsx"]
    )
    dic_categories, df_texts = {}, pd.DataFrame()
    if uploaded_file:
        # Load the Excel file into a DataFrame
        df_texts = pd.read_excel(uploaded_file)
        df_texts.dropna(inplace=True)
        df_texts.columns = ["id", "text"]
        df_texts["id"] = df_texts["id"].astype(int)
        df_texts.set_index("id", inplace=True)

    if uploaded_categories:
        # Load the Excel file into a DataFrame
        df_categories = pd.read_excel(uploaded_categories)
        df_categories.columns = ["id", "category"]
        dic_categories = dict(
            zip(list(df_categories["id"]), list(df_categories["category"]))
        )
    return df_texts, dic_categories


@st.cache_data
def get_demo_data():
    """
    This function loads demo data from two Excel files - one with expressions and one with categories.
    It reads the demo expressions Excel file into a pandas DataFrame and drops any rows with missing values.
    It then renames the columns to 'id' and 'text', converts the 'id' column to integer type and sets it as the index.
    Next, it reads the categories Excel file into another pandas DataFrame, renames the columns to 'id' and 'category',
    creates a dictionary of category IDs and category names and returns it along with the expressions DataFrame.

    Returns:
        df_texts (pandas.DataFrame): DataFrame containing the loaded demo expressions Excel file.
        dic_categories (dict): Dictionary containing the ID-category mapping from the loaded demo categories Excel file.
    """

    df_texts = pd.read_excel(DEMO_TEXT_FILE)
    df_texts.dropna(inplace=True)
    df_texts.columns = ["id", "text"]
    df_texts["id"] = df_texts["id"].astype(int)
    df_texts.set_index("id", inplace=True)

    df_categories = pd.read_excel(DEMO_CATEGORY_FILE)
    df_categories.columns = ["id", "category"]
    dic_categories = dict(
        zip(list(df_categories["id"]), list(df_categories["category"]))
    )
    return df_texts, dic_categories


@st.cache_data
def record_selection(df: pd.DataFrame, no_records: int) -> pd.DataFrame:
    """
    This function randomly select a fixed number of records from a pandas DataFrame that needs to be provided as input.

    Args:
        df (pandas.DataFrame): The DataFrame to sample from.
        no_records (int): The number of records required in the returned DataFrame.

    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    df = df.dropna()
    return df.sample(n=no_records)


def main() -> None:
    """
    This function runs an app that classifies text data. Depending on the user's input option, it retrieves data from a demo or an uploaded file.
    Then, randomly selects a fixed number of records from the dataframe provided using record_selection function. The selected dataframe and dictionary of categories are previewed on the screen. If the user presses
    classify, the function runs the Classifier class on the selected dataframe, returns a response dataframe, and offers the user the option to download the dataframe to a CSV file.

    Args:
        None

    Returns:
        None
    """
    st.header(APP_NAME)
    show_info()
    mode_options = ["Demo", "File Upload"]
    sel_mode = st.radio("Mode", options=mode_options)
    if mode_options.index(sel_mode) == 0:
        df_texts, dic_categories = get_demo_data()
    else:
        df_texts, dic_categories = get_uploaded_files()

    if not df_texts.empty and dic_categories:
        if MAX_RECORDS > 0:
            df_texts = record_selection(df_texts, MAX_RECORDS)

        preview_input(df_texts, dic_categories)

        if not df_texts.empty and dic_categories:
            if st.button("Classify"):
                classifier = Classifier(df_texts, dic_categories)
                with st.spinner("Classifying..."):
                    response_df = classifier.run()
                st.success("Done!")
                st.write(response_df)

                st.download_button(
                    label="Download CSV",
                    data=response_df.to_csv(sep="\t").encode("utf-8"),
                    file_name="classified.csv",
                    mime="text/csv",
                )
    st.markdown(APP_INFO, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
