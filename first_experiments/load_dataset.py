import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split

class LoadDataset():

    def __init__(self, filename):
        self.filename = filename
        self.X = False
        self.y = False
        self.identifier = {}
        self.label = {}

    def load(self):
        

        # I need to select only to classes Iris-setosa and Iris-versicolor for my tests today
        df_set = pd.read_csv(self.filename)

        unique_labels = (list(df_set['Species'].unique()))

        label = dict(zip(unique_labels, range(len(unique_labels))))
        self.label = label
        set_size = len(df_set)
        ncolumns = len(df_set.columns) - 2
        column_names = df_set.columns       

        X = torch.zeros((set_size, ncolumns), dtype=torch.float)
        y = torch.zeros((set_size), dtype=torch.uint8)
        
        columns2sv = []
        for i, column in enumerate(column_names[1:-1]):
            columns2sv.append(column)
            X[:, i] = torch.tensor(df_set[column].values, dtype=torch.float64)
            X[:,i] = (X[:,i] - X[:,i].mean())/ X[:,i].std()   
            
        column = column_names[-1]        
        y[:] = torch.tensor([label[name] for name in df_set[column].values], dtype=torch.long)
        self.X = X
        self.y = y
        self.identifier = {}
        for i in range(len(X)):
            features = X[i].tolist()
            feature_tuple = tuple([round(n,1) for n in features])
            self.identifier[feature_tuple] = df_set['Id'].values[i]


    def split_dataset(self, per=0.3):
        dataset = TensorDataset(self.X, self.y)
        test_size = int(per*len(dataset))
        train_size = len(dataset) - test_size

        lengths = [train_size, test_size]
        train_data, test_data = random_split(dataset, lengths)
        return train_data, test_data
