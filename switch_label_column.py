import pandas as pd
from opt_meta_function import opt_meta_function

def switch_label_column(filepath):
	data = pd.read_csv(filepath)
	swichted = pd.concat([data.iloc[:,-1], data.iloc[:,:-1]], axis=1)
	swichted.to_csv(filepath, header=False, index=False)

if __name__ == '__main__':
	opt_meta_function(switch_label_column)


