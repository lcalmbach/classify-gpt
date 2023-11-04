import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests
import os
from helper import get_used_languages, init_lang_dict_complete, get_lang
import gpt_classifier

__version__ = "0.1.0"
__author__ = "Lukas Calmbach"
__author_email__ = "lcalmbach@gmail.com"
VERSION_DATE = "2023-11-04"
APP_NAME = "ClassifyGPT"
GIT_REPO = "https://github.com/lcalmbach/classify-gpt"
MAX_RECORDS = 10000
DEMO_CATEGORY_FILE = "./demo_categories.xlsx"
DEMO_TEXT_FILE = "./demo_texts.xlsx"
lang = {}
LOTTIE_URL = "https://lottie.host/1690cd0e-184d-4481-a621-0ddc622fb335/9bUMwArBUr.json"
selection_options = ["All", "Define an interval", "Random selection"]


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


def show_info(llm_settings):
    """
    This function displays information about the application and the required input format.
    The info is shown in an expander widget.
    :return: None
    """

    with st.expander(lang["info-label"]):
        # st.write(llm_settings)
        text = lang["app-info"].format(APP_NAME, 10, APP_NAME)
        st.write(text)


def preview_input(df_texts, dic_categories):
    """
    Displays the input data for preview in the Streamlit app.

    Args:
    - df_texts (pandas.DataFrame): A DataFrame containing the input texts.
    - dic_categories (dict): A dictionary containing the categories and their corresponding labels.

    Returns:
    - None
    """
    with st.expander(lang["texts"], expanded=False):
        st.write(df_texts)

    with st.expander(lang["categories"], expanded=False):
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
    df_texts["text"] = df_texts["text"].str.replace(r"[\r\n]", ",", regex=True)
    df_texts.set_index("id", inplace=True)

    df_categories = pd.read_excel(DEMO_CATEGORY_FILE)
    df_categories.columns = ["id", "category"]
    dic_categories = dict(
        zip(list(df_categories["id"]), list(df_categories["category"]))
    )

    return df_texts, dic_categories


@st.cache_data
def record_selection(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    This function randomly select a fixed number of records from a pandas DataFrame that needs to be provided as input.

    Args:
        df (pandas.DataFrame): The DataFrame to sample from.
        no_records (int): The number of records required in the returned DataFrame.

    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    df = df.dropna()
    if settings["selection_type"] == selection_options[0]:
        df = df[:MAX_RECORDS]
    elif settings["selection_type"] == selection_options[1]:
        df = df[settings["from_rec"] : settings["to_rec"]]
    elif settings["selection_type"] == selection_options[2]:
        df = df.sample(n=settings["max_records"])

    df["text"] = df["text"].str.replace(r"[\r\n]", " ", regex=True)

    return df


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

        settings["selection_type"] = st.selectbox(
            "Selection type",
            options=selection_options,
            help=lang['selection_type_help'],
        )
        if selection_options.index(settings["selection_type"]) == 0:
            settings["from_rec"] = 0
            settings["to_rec"] = MAX_RECORDS
        elif selection_options.index(settings["selection_type"]) == 1:
            settings["from_rec"] = st.number_input("From record", 0, MAX_RECORDS, 0, 10)
            settings["to_rec"] = st.number_input("To record", 0, MAX_RECORDS, 10, 10)
        elif selection_options.index(settings["selection_type"]) == 2:
            settings["max_records"] = st.number_input(
                "Max. number of records", 10, MAX_RECORDS, 10, 10
            )
        settings["model"] = st.selectbox(lang['model'], gpt_classifier.MODEL_OPTIONS)
        settings["temperature"] = st.slider(
            label=lang['temperature'],
            value=gpt_classifier.DEFAULT_TEMP,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help=lang["help_temperature"],
        )
        settings["max_tokens"] = st.slider(
            label=lang['max_number_of_tokens'],
            value=gpt_classifier.DEFAULT_MAX_TOKENS,
            min_value=0,
            max_value=4096,
            step=1,
            help=lang["help_max_tokens"],
        )
        settings["max_categories"] = 3

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
    mode_options = lang["mode-options"]
    llm_settings = get_settings()

    show_info(llm_settings)
    sel_mode = st.radio(lang["mode"], options=mode_options)
    if mode_options.index(sel_mode) == 0:
        df_texts, dic_categories = get_demo_data()
    if mode_options.index(sel_mode) == 1:
        df_texts, dic_categories = get_uploaded_files()
        with st.columns(3)[0]:
            llm_settings["max_categories"] = st.number_input(
                lang['max_categories'], 1, 100, 3, 1
            )
    if mode_options.index(sel_mode) == 2:
        df_texts, dic_categories = get_user_input()

    llm_settings["code_for_no_match"] = st.selectbox(
        lang["code_for_no_match"],
        list(dic_categories.keys()),
        format_func=lambda x: dic_categories[x],
    )
    if not df_texts.empty and dic_categories:
        df_texts = record_selection(df_texts, llm_settings)
        preview_input(df_texts, dic_categories)
        if not df_texts.empty and dic_categories:
            if not ("classifier" in st.session_state):
                st.session_state["classifier"] = gpt_classifier.Classifier()
            classifier = st.session_state["classifier"]
            classifier.texts_df = df_texts
            classifier.category_dic = dic_categories
            classifier.settings = llm_settings
            if st.button(lang["classify"]):
                placeholder = st.empty()

                with st.spinner(lang["classifying"]):
                    classifier.run(placeholder)
                if len(classifier.errors) > 0:
                    st.warning(
                        f"{lang['error_message_result']} {','.join(classifier.errors)}"
                    )
                else:
                    st.success(lang["classify-success"])

                if os.path.exists(classifier.output_file_zip):
                    # Read the zip file as bytes
                    with open(classifier.output_file_zip, "rb") as fp:
                        btn = st.download_button(
                            label=lang['download_results'],
                            data=fp,
                            file_name="download.zip",
                            mime="application/zip",
                            help=lang['download_help_text'],
                        )

    st.sidebar.markdown(get_app_info(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
