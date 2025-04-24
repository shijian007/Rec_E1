import pandas as pd
import numpy as np
import os
import json
import pdb

PROMPT = """You are an expert in query generation. Given a query, your task is to create query terms to retrieve retrieve the most relevant products, ensuring they best meet customer needs.
Below is the query:
```{user_query}```"""

def make_prefix(dp):
    input_str = PROMPT.format(user_query=dp['query'])
    input_str = """<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. The generated query should use Boolean operators (AND, OR) to structure your query logically. For example,
<answer>
{
    "query": xxx
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n"""

    input_str += f'{dp["output"]}<|im_end|>'

    return input_str


def construct_data(split):
    data_path = os.path.join(f'data/esci/inst/sparse/subset/{split}.parquet')
    data = pd.read_parquet(data_path)
    
    # read results/esci/{split}/gpt-4o-esci_esci.json
    with open(f'results/esci/{split}/gpt-4o-esci_esci.json', 'r') as f:
        gpt_4o_res_dict = json.load(f)
    
    assert len(data) == len(gpt_4o_res_dict)
    # data[i]['output] = gpt_4o_res[i]['output']
    for i in range(len(data)):
        assert len(eval(gpt_4o_res_dict[str(i)]['target'])) == len(data.loc[i, 'item_id'])
        data.loc[i, 'output'] = gpt_4o_res_dict[str(i)]['generated_text']

    # add one column text, = make_prefix + output
    data['text'] = data.apply(lambda x: make_prefix(x), axis=1)
    # remove all other columns
    data = data[['text']]

    # save to data/esci/inst/sparse/sft/train.parquet
    data.to_parquet(f'data/esci/inst/sparse/sft/{split}.parquet')




if __name__ == '__main__':
    construct_data('train')
    construct_data('val')