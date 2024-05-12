FROM python:3

# ADD script.py /script.py
COPY . .
ADD requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt
RUN pip3 install huggingface_hub

RUN huggingface-cli download MohammadKarami/bert-human-detector
RUN huggingface-cli download MohammadKarami/roberta-human-detector
RUN huggingface-cli download MohammadKarami/electra-human-detector

ENTRYPOINT [ "python3", "/script.py", "--input", "$inputDataset", "--output", "$outputDir" ]

