import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests
from helper import get_used_languages, init_lang_dict_complete, get_lang
import gpt_classifier

__version__ = "0.0.3"
__author__ = "Lukas Calmbach"
__author_email__ = "lcalmbach@gmail.com"
VERSION_DATE = "2023-07-31"
APP_NAME = "ClassifyGPT"
GIT_REPO = "https://github.com/lcalmbach/classify-gpt"

MAX_RECORDS = 10
DEMO_CATEGORY_FILE = "./categories.xlsx"
DEMO_TEXT_FILE = "./2013_F13.xlsx"
lang = {}
LOTTIE_URL = "https://lottie.host/1690cd0e-184d-4481-a621-0ddc622fb335/9bUMwArBUr.json"


def get_app_info():
    """
    Returns a string containing information about the application.
    Returns:
    - info (str): A formatted string containing details about the application.
    """
    created_by = lang["created_by"]
    powered_by = lang["powered_by"]
    version = lang["version"]
    translation = lang["text_translation"]

    info = f"""<div style="background-color:powderblue; padding: 10px;border-radius: 15px;">
    <small>{created_by} <a href="mailto:{__author_email__}">{__author__}</a><br>
    {version}: {__version__} ({VERSION_DATE})<br>
    {powered_by} <a href="https://streamlit.io/">Streamlit</a> and 
    <a href="https://platform.openai.com/">OpenAI API</a><br> 
    <a href="{GIT_REPO}">git-repo</a>
    """

    created_by = lang["created_by"]
    powered_by = lang["powered_by"]
    version = lang["version"]

    info = f"""<div style="background-color:powderblue; padding: 10px;border-radius: 15px;">
    <small>{created_by} <a href="mailto:{__author_email__}">{__author__}</a><br>
    {version}: {__version__} ({VERSION_DATE})<br>
    {powered_by} <a href="https://streamlit.io/">Streamlit</a> and 
    <a href="https://platform.openai.com/">OpenAI API</a><br> 
    <a href="{GIT_REPO}">git-repo</a><br> 
    {translation} <a href="https://lcalmbach-gpt-translate-app-i49g8c.streamlit.app/">PolyglotGPT</a>
    """
    return info


def show_info():
    """
    This function displays information about the application and the required input format.
    The info is shown in an expander widget.
    :return: None
    """

    with st.expander(lang["info-label"]):
        text = lang["app-info"].format(APP_NAME, MAX_RECORDS, APP_NAME)
        st.write(text)


def preview_input(df_texts, dic_categories):
    with st.expander(lang["texts"], expanded=True):
        st.write(df_texts)

    with st.expander(lang["categories"], expanded=True):
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

    uploaded_file = st.file_uploader(lang["upload-excel-expressions"], type=["xlsx"])
    uploaded_categories = st.file_uploader(
        lang["upload-excel-categories"], type=["xlsx"]
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


def get_user_input():
    """
    Prompts the user to enter text and categories, and returns them as a
    DataFrame and a dictionary.

    Returns:
        df_texts (pd.DataFrame): A DataFrame containing the user-entered text.
        dic_categories (dict): A dictionary mapping category indices to
        category names.
    """
    text = st.text_area(lang["text_to_classify"])
    categories = st.text_area(lang["categories_csv"])

    df_texts = pd.DataFrame({"id": [1], "text": [text]})
    df_texts.set_index("id", inplace=True)

    categories = categories.split(",")
    dic_categories = {i + 1: item for i, item in enumerate(categories)}
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


def display_language_selection():
    """
    The display_info function displays information about the application. It
    uses the st.expander container to create an expandable section for the
    information. Inside the expander, displays the input and output format.
    """
    index = list(st.session_state["used_languages_dict"].keys()).index(
        st.session_state["lang"]
    )
    x = st.sidebar.selectbox(
        label=f'🌐{lang["language"]}',
        options=st.session_state["used_languages_dict"].keys(),
        format_func=lambda x: st.session_state["used_languages_dict"][x],
        index=index,
    )
    if x != st.session_state["lang"]:
        st.session_state["lang"] = x
        refresh_lang()


def refresh_lang():
    """
    The refresh_lang function is responsible for refreshing the language dictionary used
    in the application. It updates the lang_dict variable in the session state with
    the new language dictionary obtained from the get_lang function.

    The function then displays the updated language dictionary and finally
    triggers a rerun of the application to refresh all language on the UI.
    """
    st.session_state["lang_dict"] = get_lang(st.session_state["lang"])
    st.write(st.session_state["lang_dict"])
    st.experimental_rerun()


@st.cache_data()
def get_lottie():
    """Performs a GET request to fetch JSON data from a specified URL.

    Returns:
        tuple: A tuple containing the JSON response and a flag indicating the
        success of the request.

    Raises:
        requests.exceptions.RequestException: If an error occurs during the
        GET request. ValueError: If an error occurs while parsing the JSON
        response.
    """
    ok = True
    r = None
    try:
        response = requests.get(LOTTIE_URL)
        r = response.json()
    except requests.exceptions.RequestException as e:
        print(lang["get-request-error"]).format(e)
        ok = False
    except ValueError as e:
        print(lang["json-parsing-error"].format(e))
        ok = False
    return r, ok


def get_settings():
    """
    Prompts the user to enter model poarameters and returns them as a dictionary.
    Default values are dinfed in the gpt_classifier module.

    Returns:
        settings (dict): A dictionary containing the user-entered settings.
    """
    settings = {}
    with st.sidebar.expander(lang["llm_settings"]):
        settings["model"] = st.selectbox("model", gpt_classifier.MODEL_OPTIONS)
        settings["temperature"] = st.slider(
            label="Temperature",
            value=gpt_classifier.DEFAULT_TEMP,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help=lang["help_temperature"],
        )
        settings["top_p"] = st.slider(
            "top_p",
            value=gpt_classifier.DEAFULT_TOP_P,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help=lang["help_top_p"],
        )
        settings["max_tokens"] = st.slider(
            "Maximum tokens",
            value=gpt_classifier.DEFAULT_MAX_TOKENS,
            min_value=0,
            max_value=4096,
            step=1,
            help=lang["help_max_tokens"],
        )
        settings["frequency_penalty"] = st.slider(
            "Frequency penalty",
            value=gpt_classifier.DEFAULT_FREQUENCY_PENALTY,
            min_value=-2.0,
            max_value=2.0,
            step=0.1,
            help=lang["help_presence_penalty"],
        )
        settings["presence_penalty"] = st.slider(
            "Presence penalty",
            value=gpt_classifier.DEFAULT_PRESENCE_PENALTY,
            min_value=-2.0,
            max_value=2.0,
            step=0.1,
            help=lang["help_frequency_penalty"],
        )

    return settings


def main() -> None:
    """
    This function runs an app that classifies text data. Depending on the user's
    input option, it retrieves data from a demo or an uploaded file. Then,
    randomly selects a fixed number of records from the dataframe provided using
    record_selection function. The selected dataframe and dictionary of categories
    are previewed on the screen. If the user presses classify, the function runs the
    Classifier class on the selected dataframe, returns a response dataframe, and
    offers the user the option to download the dataframe to a CSV file.
    """
    st.header(APP_NAME)
    global lang

    init_lang_dict_complete("app.py")

    if not ("lang" in st.session_state):
        # first item is default language
        st.session_state["used_languages_dict"] = get_used_languages()
        st.session_state["lang"] = next(
            iter(st.session_state["used_languages_dict"].items())
        )[0]
        refresh_lang()

    lang = st.session_state["lang_dict"]
    lottie_search_names, ok = get_lottie()
    if ok:
        with st.sidebar:
            st_lottie(lottie_search_names, height=140, loop=20)
    else:
        pass
    display_language_selection()
    show_info()
    mode_options = lang["mode-options"]
    llm_settings = get_settings()

    sel_mode = st.radio(lang["mode"], options=mode_options)
    if mode_options.index(sel_mode) == 0:
        df_texts, dic_categories = get_demo_data()
    if mode_options.index(sel_mode) == 1:
        df_texts, dic_categories = get_uploaded_files()
    if mode_options.index(sel_mode) == 2:
        df_texts, dic_categories = get_user_input()

    if not df_texts.empty and dic_categories:
        if MAX_RECORDS > 0 and len(df_texts) > MAX_RECORDS:
            df_texts = record_selection(df_texts, MAX_RECORDS)

        preview_input(df_texts, dic_categories)

        if not df_texts.empty and dic_categories:
            if st.button(lang["classify"]):
                classifier = gpt_classifier.Classifier(
                    df_texts, dic_categories, llm_settings
                )
                with st.spinner(lang["classifying"]):
                    response_df = classifier.run()
                st.success(lang["classify-success"])
                st.write(response_df)

                st.download_button(
                    label=lang["download-csv"],
                    data=response_df.to_csv(sep="\t").encode("utf-8"),
                    file_name="classified.csv",
                    mime="text/csv",
                )
    st.sidebar.markdown(get_app_info(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
