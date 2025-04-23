# movie-recommendation

## install dependecies
`pip install -r requirements.txt`

## Install en_core_web_lg
After installing necessary packages from the requirements file, you need to install en_core_web_lg using `python -m spacy download en_core_web_lg`

## setup your Google Gemini API key in a .env file
[Google page](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)

## command to run the server
`uvicorn main:app --reload`
`uvicorn ml_project:app --reload --port 8001`

Visit [localhost](http://127.0.0.1:8000/docs)
