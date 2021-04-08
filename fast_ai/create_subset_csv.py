import pandas as pd
import os

art_df = pd.read_csv('./data/all_data_info.csv')
ROOT = 'data'
FOLDER = 'train_2'


def get_file_names():
    names = []
    for name in os.listdir(f'{ROOT}/{FOLDER}'):
        names.append(name)
    return names


def main():
    file_names = get_file_names()
    filter = ['Impressionism', 'Surrealism']

    art_subset_df = art_df[art_df.new_filename.isin(file_names)]

    art_subset_df = art_subset_df[['style', 'new_filename']]
    art_subset_df = art_subset_df[art_subset_df['style'].isin(filter)]
    art_subset_df.reset_index(drop=True, inplace=True)

    print(art_subset_df[art_subset_df['style'] == 'Impressionism'].value_counts())
    print(art_subset_df[art_subset_df['style'] == 'Surrealism'].value_counts())

    pd.to_pickle(art_subset_df, 'art_subset_df_2.pkl')


if __name__ == '__main__':
    main()
