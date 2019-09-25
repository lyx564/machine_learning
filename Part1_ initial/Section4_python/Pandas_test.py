import numpy as np
import pandas as pd

# myarray = np.array([1, 2, 3])
# index = ['a', 'b', 'c']
# myseries = pd.Series(myarray, index=index)
# print(myseries)
# print('Series中的第一个元素：')
# print(myseries[0])
# print('Series中的c index元素：')
# print(myseries['c'])

myarray = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
rowindex = ['row1', 'row2', 'row3']
colname = ['col1', 'col2', 'col3']
mydataframe = pd.DataFrame(data=myarray, index=rowindex, columns=colname)
print(mydataframe)
print('访问col3的数据：')
print(mydataframe['col3'])