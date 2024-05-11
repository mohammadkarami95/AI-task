FROM python:3

ADD script.py /script.py
ADD requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt
RUN huggingface-cli download MohammadKarami/bert-human-detector
RUN huggingface-cli download MohammadKarami/roberta-human-detector
RUN huggingface-cli download MohammadKarami/electra-human-detector

ENTRYPOINT [ "python3", "/script.py", "--input", "$inputDataset", "--output", "$outputDir" ]

