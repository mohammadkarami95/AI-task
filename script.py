import os
import argparse
import json
import glob
import re
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

roberta_classifier= pipeline('text-classification', model='MohammadKarami/roberta-human-detector', tokenizer="MohammadKarami/roberta-human-detector", max_length=512, truncation=True)
electra_classifier= pipeline('text-classification', model='MohammadKarami/electra-human-detector', tokenizer="MohammadKarami/electra-human-detector", max_length=512, truncation=True)
bert_classifier= pipeline('text-classification', model='MohammadKarami/bert-human-detector', tokenizer="MohammadKarami/bert-human-detector", max_length=512, truncation=True)

import pandas as pd
import json
def most_frequent(List):
    return max(set(List), key = List.count)

def pars_args():
    parser= argparse.ArgumentParser(description= "PAN 2024 Style Change Detection Task.")
    parser.add_argument("--input", type= str, help= "Folder containing input files for task(.txt)", required= True)
    parser.add_argument("--output", type= str, help= "Folder containing output/solution files(.json)", required= True)
    args = parser.parse_args()
    return args


def main():
    args= pars_args()
    is_human= 0
    df= pd.read_json(args.input, lines= True)
    df['text1']= df['text1'].apply(lambda x: re.sub('[\n]+', ' ', x))
    df['text2']= df['text2'].apply(lambda x: re.sub('[\n]+', ' ', x))

    os.makedirs(args.output, exist_ok= True)
    for index, row in df.iterrows():
        print(index)
        ro_score1= roberta_classifier(row['text1'])
        el_score1= electra_classifier(row['text1'])
        be_score1= bert_classifier(row['text1'])
        most1= most_frequent([ro_score1[0]['label'], el_score1[0]['label'], be_score1[0]['label']])
        

        ro_score2= roberta_classifier(row['text2'])
        el_score2= electra_classifier(row['text2'])
        be_score2= bert_classifier(row['text2'])

        most2= most_frequent([ro_score2[0]['label'], el_score2[0]['label'], be_score2[0]['label']])
        if most1 == 'human' and most2 =='AI':
           is_human = 0.3
        elif most2 == 'human' and most1 == 'AI':
           is_human = 1.0
        else:
           is_human = 0.5
        
        with open(args.output+'/y_pred.jsonl', 'a') as out:
          json.dump({'id': row['id'], 'is_human': is_human}, out)
          out.write('\n')
          out.flush()

        
        
if __name__ == '__main__':
    main()
