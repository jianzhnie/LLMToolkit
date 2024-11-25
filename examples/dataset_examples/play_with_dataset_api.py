from pprint import pprint

from datasets import load_dataset

if __name__ == '__main__':
    # 加载数据集
    root_dir = 'data/aclImdb/'
    imdb_dataset = load_dataset('examples/dataset_examples/custom_imdb.py',
                                data_dir=root_dir)
    print(imdb_dataset)
    print('Length of training set: ', len(imdb_dataset))
    print('First example from the dataset: \n')
    pprint(imdb_dataset['train'][:2])
    pprint(imdb_dataset['test'][:2])
    pprint(imdb_dataset['unsupervised'][:2])
