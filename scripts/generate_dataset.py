import numpy as np
import json, random
from random import sample
from tqdm import tqdm
import os
import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from options import option

args = option.get_args_parser()
dataname = 'HumanML3D' if args.dataname == 't2m' else 'KIT-ML'

def prepare(split):
    with open(f'./dataset/{dataname}/{split}.txt', 'r') as f:
        lines = f.readlines()

    dataset = []
    for line in tqdm(lines):
        line = line.strip()
        if not os.path.exists(f'./dataset/{dataname}/VQVAE/{line}.npy'): continue
        with open(f'./dataset/{dataname}/texts/{line}.txt', 'r') as f:
            texts = f.readlines()
            text = sample(texts, 1)[0].split('#')[0]

        data = np.load(f'./dataset/{dataname}/VQVAE/{line}.npy')
        list_data = data.reshape(-1).tolist()
        m_length = len(list_data)
        suffix = str(list_data[-1])
        prefix = str(list_data[0])
        sample_num = random.randint(3, 5)
        index = sample(list(range(len(list_data))), sample_num)
        index.sort()
        tokens = ','.join(str(num) for num in list(map(list_data.__getitem__, index)))
        str_data = ','.join(str(num) for num in list_data)
        dataset.append({'instruction': 'Generate a sequence of motion tokens matching the following human motion description.', 'input': f'{text}', 'output': str_data, 'motion': line, 'length': m_length})
        dataset.append({'instruction': 'Generate a sequence of motion tokens matching the following human motion description given the initial token.', 'input': f'{text}<Motion Token>{prefix}</Motion Token>', 'output': str_data, 'motion': line, 'length': m_length})
        dataset.append({'instruction': 'Generate a sequence of motion tokens matching the following human motion description given the last token.', 'input': f'{text}<Motion Token>{suffix}</Motion Token>', 'output': str_data, 'motion': line, 'length': m_length})
        dataset.append({'instruction': 'Generate a sequence of motion tokens matching the following human motion description given several key tokens.', 'input': f'{text}<Motion Token>{tokens}</Motion Token>', 'output': str_data, 'motion': line, 'length': m_length})

    dataset = json.dumps(dataset)
    with open(f'./data/{split}.json', 'w') as f:
        f.write(dataset)


def main():
    prepare('train')
    prepare('val')


if __name__ == '__main__':
    main()
