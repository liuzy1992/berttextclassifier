#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split

def trim_string(s):
    l = s.strip().split()
    ns = ' '.join(l[:200])

    return ns

def preprocessing(infile, outdir):
    # Read raw data
    # df_raw = pd.read_csv(infile, sep='\t',header=0, dtype={'id':str, 'title':str, 'content':str, 'label':int})
    df_raw = pd.read_csv(infile, sep=',', header=0, dtype={'id':str, 'title':str, 'content':str, 'label':str})

    # Prepare columns
    df_raw['label'] = (df_raw['label'] == 'FAKE').astype('int')
    df_raw = df_raw.where(df_raw.notnull(), '')
    df_raw['titlecontent'] = df_raw['title'] + ". " + df_raw['content']
    df_raw = df_raw.reindex(columns=['label', 'title', 'content', 'titlecontent'])

    # Drop rows with empty content
    df_raw.drop( df_raw[df_raw.content.str.len() < 5].index, inplace=True)

    # Trim content and titlecontent
    df_raw['content'] = df_raw['content'].apply(trim_string)
    df_raw['titlecontent'] = df_raw['titlecontent'].apply(trim_string) 

    # Split according to label
    df_neg = df_raw[df_raw['label'] == 0]
    df_pos = df_raw[df_raw['label'] == 1]

    # Train-test split
    df_pos_full_train, df_pos_test = train_test_split(df_pos, train_size = 0.1, random_state = 1)
    df_neg_full_train, df_neg_test = train_test_split(df_neg, train_size = 0.1, random_state = 1)

    # Train-valid split
    df_pos_train, df_pos_valid = train_test_split(df_pos_full_train, train_size = 0.8, random_state = 1)
    df_neg_train, df_neg_valid = train_test_split(df_neg_full_train, train_size = 0.8, random_state = 1)

    # Concatenate splits of different labels
    df_train = pd.concat([df_pos_train, df_neg_train], ignore_index=True, sort=False)
    df_valid = pd.concat([df_pos_valid, df_neg_valid], ignore_index=True, sort=False)
    df_test = pd.concat([df_pos_test, df_neg_test], ignore_index=True, sort=False)

    # Write preprocessed data
    df_train.to_csv(outdir + '/train.tsv', index=False, sep='\t')
    df_valid.to_csv(outdir + '/valid.tsv', index=False, sep='\t')
    df_test.to_csv(outdir + '/test.tsv', index=False, sep='\t')

