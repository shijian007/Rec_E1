import os
import random
import pandas as pd

random.seed(42)

domain_list = ['All_Beauty', 'Baby_Products', 'Video_Games']

ori_file_dir = 'data/amazon_review/processed'
output_dir = 'data/amazon_review/processed_filtered'

for domain in domain_list:
    input_dir = os.path.join(ori_file_dir, domain)
    transductive_dir = os.path.join(output_dir, 'transductive', domain)
    inductive_dir = os.path.join(output_dir, 'inductive', domain)
    os.makedirs(transductive_dir, exist_ok=True)
    os.makedirs(inductive_dir, exist_ok=True)

    def load_data(filename):
        return pd.read_csv(os.path.join(input_dir, filename), sep='\t')

    # Load the training and evaluation datasets
    train_df = load_data(f'{domain}.train.inter').head(100000)
    valid_df = load_data(f'{domain}.valid.inter')
    test_df = load_data(f'{domain}.test.inter')

    # Get all unique items that appear in training set (including in sequences)
    train_items = set(train_df['item_id:token'])
    seq_items = train_df['item_id_list:token_seq'].dropna().str.split().explode()
    train_items.update(seq_items)

    # Check if a row is fully inductive: all items NOT in training set
    def is_strictly_inductive(row):
        all_items = [row['item_id:token']]
        if pd.notna(row['item_id_list:token_seq']) and row['item_id_list:token_seq'].strip() != '':
            all_items += row['item_id_list:token_seq'].split()
        return all(item not in train_items for item in all_items)

    # Check if a row is fully transductive: all items ARE in training set
    def is_transductive(row):
        all_items = [row['item_id:token']]
        if pd.notna(row['item_id_list:token_seq']) and row['item_id_list:token_seq'].strip() != '':
            all_items += row['item_id_list:token_seq'].split()
        return all(item in train_items for item in all_items)

    # Split the dataset into transductive and inductive subsets
    def split_sets(df, total=1000):
        trans_mask = df.apply(is_transductive, axis=1)
        ind_mask = df.apply(is_strictly_inductive, axis=1)

        trans = df[trans_mask].sample(n=min(total, trans_mask.sum()), random_state=42)
        ind = df[ind_mask].sample(n=min(total, ind_mask.sum()), random_state=42)
        return trans, ind

    # Apply split to validation and test sets
    valid_trans, valid_ind = split_sets(valid_df, 1000)
    test_trans, test_ind = split_sets(test_df, 1000)

    # Save results
    train_df.to_csv(os.path.join(transductive_dir, f'{domain}.train.inter'), sep='\t', index=False)
    valid_trans.to_csv(os.path.join(transductive_dir, f'{domain}.valid.inter'), sep='\t', index=False)
    test_trans.to_csv(os.path.join(transductive_dir, f'{domain}.test.inter'), sep='\t', index=False)

    train_df.to_csv(os.path.join(inductive_dir, f'{domain}.train.inter'), sep='\t', index=False)
    valid_ind.to_csv(os.path.join(inductive_dir, f'{domain}.valid.inter'), sep='\t', index=False)
    test_ind.to_csv(os.path.join(inductive_dir, f'{domain}.test.inter'), sep='\t', index=False)

    # Print status
    print(f"[{domain}] Done ✓")
    print(f"  Transductive - valid: {len(valid_trans)} | test: {len(test_trans)}")
    print(f"  Inductive   - valid: {len(valid_ind)} | test: {len(test_ind)}")
