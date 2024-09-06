```
conda create -n medline-qa python=3.9
conda activate medline-qa


!python3 -m spacy download en_core_web_lg # For Sensitive data masking

fastapi dev src/main.py      


 PYTHONPATH=src python ./test/rag_test.py

```