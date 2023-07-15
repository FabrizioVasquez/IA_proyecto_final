from datasets import DatasetDict, load_dataset
import pandas as pd
import csv
import json
from datasets import load_dataset

def main():
    label2id = {"positive": 2, "neutral": 1, "negative": 0}

    for split in ["train", "test"]:
        input_file = csv.DictReader(open(f"{split}_csv.txt"))

        with open(f'{split}.jsonl', 'w') as fOut:
            for row in input_file:
                fOut.write(json.dumps({'textID': row['textID'], 'text': row['text'], 'label': label2id[row['sentiment']], 'label_text': row['sentiment']})+"\n")
            
    


    """
    train_dset = load_dataset("csv", data_files="raw_data/train_csv", split="train")
    train_dset = train_dset.remove_columns(["selected_text"])
    test_dset = load_dataset("csv", data_files="raw_data/train_csv", split="train")
    raw_dset = DatasetDict()
    raw_dset["train"] = train_dset
    raw_dset["test"] = test_dset

    for split, dset in raw_dset.items():
        dset = dset.rename_column("sentiment", "label_text")
        dset = dset.map(lambda x: {"label": label2id[x["label_text"]]}, num_proc=8)
        dset.to_json(f"{split}.jsonl")
    """
    
def read_jsonl_to_csv():   
    for split in ["train", "test"]: 
        data = []
        with open(f'{split}.jsonl') as f:
            for line in f:
                data.append(json.loads(line))

        # Convierte la lista de diccionarios en un DataFrame
        df = pd.DataFrame(data)

        # Escribe el DataFrame a un archivo .csv
        df.to_csv(f'news_finance_tweets_{split}.csv', index=False)
    
def get_database_real_tweets():
    dataset_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    dataset_test  = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    df_train = dataset_train.to_pandas()
    df_test = dataset_train.to_pandas()
    
    print('Todos los labels antes de Formatear')
    print(df_train['label'].value_counts(normalize=True))
    df_train = df_train[df_train['label'] != 2]
    df_test = df_test[df_test['label'] != 2]

    convert_positive_negative = lambda x: x if x == 1 else -1
    df_train['label'] = df_train['label'].apply(convert_positive_negative)
    df_test['label'] = df_test['label'].apply(convert_positive_negative)
    print('Todos los labels despues de Formatear')
    print(df_train['label'].value_counts(normalize=True))

    df_train.to_csv(f'zeroshot_news_finance_tweets_train.csv', index=False)
    df_test.to_csv(f'zeroshot_news_finance_tweets_test.csv', index=False)

if __name__ == "__main__":
    #main()
    #read_jsonl_to_csv()
    get_database_real_tweets()
