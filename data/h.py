import json
from typing import DefaultDict
# from utilities import database
with open('data/topic_tfidfs.json' ,'r') as f:
  tfidfs = json.load(f)

subject_keywords = DefaultDict(list)

for wordi in tfidfs:
  word = list(wordi.keys())[0]
  code = list(wordi[word])[0]['code']
  subject_keywords[code].append(word)

jsonable = []
for subject, words in subject_keywords.items():
  jsonable.append({'subjectCode': subject, 'keywords': words})

with open('subject_keywords.json', 'w') as f:
  json.dump(jsonable, f)
