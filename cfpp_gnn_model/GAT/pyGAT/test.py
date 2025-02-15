import numpy
from sklearn.model_selection import train_test_split

test_list = range(100)


train_data, test_data = train_test_split(test_list, test_size=0.2, random_state=42)

print(train_data)
print(test_data)