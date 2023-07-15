# classify-gpt

## Introduction
ClassifyGPT is an app that categorizes text using the OpenAI GPT-3 API. The app is built with the Streamlit framework, and implements the API through the gpt_classifier.py module.

## Usage

To use the app, you need to upload two Excel (xlsx) files:

A list of expressions with the columns 'id' and 'text'
A list of categories with the columns 'id' and 'category'

Alternatively you can run in demo mode, using a preloaded dataset
You can then run the classification process by clicking the 'Classify' button. The resulting categorized expressions will be shown on the screen, and you can download them as a CSV file using the 'Download CSV' button.

## Run

You can clone and run the app locally with the following commands:

```
>git clone https://github.com/lcalmbach/classify-gpt.git
>cd classify-gpt
>pyython -m venv env
>env\scripts\activate
>pip install requirements.txt
>streamlit run app.py
```

The app will run on localhost:8501.

## License

This project is licensed under the terms of the MIT license.