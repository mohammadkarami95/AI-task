# Has already torch and cuda installed
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip3 install huggingface_hub argparse transformers pandas
RUN huggingface-cli download MohammadKarami/bert-human-detector
RUN huggingface-cli download MohammadKarami/roberta-human-detector
RUN huggingface-cli download MohammadKarami/electra-human-detector


ADD script.py /script.py


ENTRYPOINT [ "python3", "/script.py", "--input", "$inputDataset", "--output", "$outputDir" ]

