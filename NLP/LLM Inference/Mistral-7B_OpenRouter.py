import os
import json
import re

from api import api_key, BASE_PATH, FILE_PATH
from openai import OpenAI

READ_FILES = True

json_schema = [
  {
    "categoria": "Economia",
    "descrizione": "Fatturato annuo",
    "valore": 35000000,
    "unità": "EUR"
  },
  {
    "categoria": "Turismo",
    "descrizione": "Presenze turistiche",
    "valore": 105000000,
    "unità": "persone"
  },
  {
    "categoria": "Demografia",
    "descrizione": "Popolazione totale",
    "valore": 60000000,
    "unità": "persone"
  }
]


def read_file(file_path):
  try:
    with open(file_path, 'r', encoding='utf-8') as file:
      content = file.read()
      # print(content)
      return content
  except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
  except Exception as e:
    print(f"Could not read {file_path}: {e}")

def read_files(folder_path):
  contents = []
  try:
    files = os.listdir(folder_path)
  except FileNotFoundError:
    print(f"The folder at {folder_path} does not exist!!")
    return
  for filename in files:
    file_path = os.path.join(folder_path, filename)
    # check if is a file
    if os.path.isfile(file_path):
      try:
        with open(file_path, "r", encoding="utf-8") as file:
          content = file.read()
          # print(content)
          contents.append(content)
      except Exception as e:
        print(f"Couldn't read the file at {file_path}: ERROR {e}")
        return
  return contents

def main():
  if READ_FILES is True:
    content = read_files(BASE_PATH)
  else:
    content = read_file(FILE_PATH)

  for i, content in enumerate(content):
    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=api_key,
    )

    completion = client.chat.completions.create(
      model="mistralai/mistral-7b-instruct:free",
      messages=[
        {
          "role": "user",
          "content": f"""
              Analizza il testo fornito ed estrai TUTTI I PRINCIPALI valori numerici, inclusi quelli legati a quantità, misure, statistiche, percentuali, date e somme di denaro. 
              Organizza i dati in una tabella con due colonne: 'descrizione' e 'valore'. Assicurati che ogni descrizione sia chiara e rappresentativa del dato numerico estratto. 

              Restituisci SOLO il risultato in formato JSON valido, seguendo esattamente questa struttura:
              {json_schema}

              Rispondi esclusivamente in ITALIANO. NON AGGIUNGERE COMMENTI, TESTO DI ALTRO TIPO O SPIEGAZIONI, FORNISCI SOLO IL JSON. 

              Testo da analizzare: {content}
              """
        }
      ]
    )
    text = completion.choices[0].message.content.strip()
    print(text)

    try:
      data = json.loads(text)
      print("The LLM model output a valid JSON :)")

      filename = f"extractd_data{i}.json"
      # Save JSON to a file
      with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
      print(f"JSON file saved as {filename}")

    except json.JSONDecodeError as e:
      print(f"The LLM model output an invalid JSON :( Please try again: {e}")

main()