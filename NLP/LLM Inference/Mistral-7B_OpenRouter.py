import os
import json
import re

from api import api_key, BASE_PATH, FILE_PATH
from openai import OpenAI

READ_FILES = True

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
          "content": f"Trova nel testo fornito ed estrai tutti i valori numerici rilevanti. Organizzali in una tabella con due colonne: 'descrizione' e 'valore'. Restituisci SOLAMENTE il risultato in un formato JSON valido. Rispondi in ITALIANO. NON AGGIUNGERE COMMENTI O TESTO DI ALTRO TIPO NELLA TUA RISPOSTA, FORNISCIMI SOLO IL JSON. Testo da analizzare: {content}"
        }
      ]
    )
    text = completion.choices[0].message.content
    print(text)

    # Regular expression to extract JSON content
    match = re.search(r"```json\s*(\[.*?])\s*```", text, re.DOTALL)

    if match:
      json_text = match.group(1)  # Extract JSON content
      data = json.loads(json_text)  # Convert to Python dictionary
      print("Extracted JSON:", data)

      filename = f"extractd_data{i}.json"

      # Save JSON to a file
      with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
      print(f"JSON file saved as {filename}")
    else:
      print("No JSON found in the text.")

main()