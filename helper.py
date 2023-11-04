import streamlit as st
import iso639
import json
from io import BytesIO
import os
import socket
import string
import csv
import zipfile
import random

lang_dict_complete = {}


def get_lang_name(lang_code):
    return get_all_language_dict()[lang_code]


def init_lang_dict_complete(module: str):
    """
    Retrieves the complete language dictionary from a JSON file.

    Returns:
    - lang (dict): A Python dictionary containing all the language strings.
    """
    global lang_dict_complete

    lang_file = f"./lang/{module.replace('.py','.json')}"
    try:
        with open(lang_file, "r") as file:
            lang_dict_complete = json.load(file)
    except FileNotFoundError:
        print("File not found.")
        return {}
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return {}
    except Exception as e:
        print("An error occurred:", str(e))
        return {}


def get_all_language_dict():
    """
    Retrieves a dictionary containing all the available languages and their
    ISO 639-1 codes.

    Returns:
        language_dict (dict): A Python dictionary where the keys are the ISO 639-1 codes and the values are the language names.
    """
    keys = [lang["iso639_1"] for lang in iso639.data if lang["iso639_1"] != ""]
    values = [lang["name"] for lang in iso639.data if lang["iso639_1"] != ""]
    language_dict = dict(zip(keys, values))
    return language_dict


def get_used_languages():
    language_dict = get_all_language_dict()
    used_languages = list(lang_dict_complete.keys())
    extracted_dict = {
        key: language_dict[key] for key in used_languages if key in language_dict
    }
    return extracted_dict


def get_lang(lang_code: str):
    return lang_dict_complete[lang_code]


def download_button(data, download_filename, button_text):
    """
    Function to create a download button for a given object.

    Parameters:
    - object_to_download: The object to be downloaded.
    - download_filename: The name of the file to be downloaded.
    - button_text: The text to be displayed on the download button.
    """
    # Create a BytesIO buffer
    json_bytes = json.dumps(data).encode("utf-8")
    buffer = BytesIO(json_bytes)

    # Set the appropriate headers for the browser to recognize the download
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.download_button(
        label=button_text,
        data=buffer,
        file_name=download_filename,
        mime="application/json",
    )


def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False


def get_var(varname: str) -> str:
    """
    Retrieves the value of a given environment variable or secret from the Streamlit configuration.

    If the current host is the local machine (according to the hostname), the environment variable is looked up in the system's environment variables.
    Otherwise, the secret value is fetched from Streamlit's secrets dictionary.

    Args:
        varname (str): The name of the environment variable or secret to retrieve.

    Returns:
        The value of the environment variable or secret, as a string.

    Raises:
        KeyError: If the environment variable or secret is not defined.
    """
    if socket.gethostname().lower() == LOCAL_HOST:
        return os.environ[varname]
    else:
        return st.secrets[varname]


def get_random_word(length=5) -> str:
    """
    Generate a random word of a given length.

    This function generates a random word by choosing `length` number of random letters from ASCII letters.

    Parameters:
    length (int): The length of the random word to generate. Default is 5.

    Returns:
    str: The generated random word.
    """
    # Choose `length` random letters from ascii_letters
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(length))


def create_file(file_name: str, columns: list) -> None:
    """
    Creates a new file and writes the columns list to the file.

    Parameters:
    file_name (str): The name of the file to be created.
    columns (list): The list of columns to be written to the file.

    Returns:
    None
    """
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(columns)


def append_row(file_name: str, row: list) -> None:
    """
    Appends a row to a CSV file.

    Args:
        file_name (str): The name of the CSV file to append to.
        row (list): The row to append to the CSV file.

    Returns:
        None
    """
    with open(file_name, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(row)


def zip_files(file_names: list, target_file: str):
    """
    Compresses a list of files into a zip file. The zip file will be
    downloaded to the user's computer if download button is clicked.

    :return: None
    """

    # Create a new zip file and add files to it
    with zipfile.ZipFile(target_file, "w") as zipf:
        for file in file_names:
            # Add file to the zip file
            # The arcname parameter avoids storing the full path in the zip file
            zipf.write(file, arcname=os.path.basename(file))


LOCAL_HOST = "liestal"
