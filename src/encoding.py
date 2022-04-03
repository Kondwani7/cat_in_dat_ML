import numpy as np
from scipy import sparse
from sklearn import preprocessing
example = np.array([
    [2,0,4],
    [0,6,7],
    [8,0,1]
])

#print("Size of dense array: ", example.nbytes)

example2 = np.array([
    [0,0,1],
    [0,1,0],
    [1,0,1]
])

sparse_example = sparse.csc_matrix(example)
#print("array in sparse format", sparse_example.data.nbytes)
#large dense array
n_rows = 10000
n_colums = 10000
lg_example = np.random.binomial(1, p=.05, size=(n_rows, n_colums))
#print(f"Size of dense array: {lg_example.nbytes}")
lg_sparse_example = sparse.csr_matrix(lg_example.nbytes)
full_size = (
    lg_sparse_example.data.nbytes + 
    lg_sparse_example.indptr.nbytes + 
    lg_sparse_example.indices.nbytes
)
#print(f"Size of sparse array: {full_size}")
#print(f"Amount size saved using a sparse array:{lg_example.nbytes - full_size} bytes")
'''One hot encoding'''
#1d array with 1001 cateorgires
example3 = np.random.randint(1000, size=1000000)
#initalize the one hot encoder
#for a dense array
ohe1 = preprocessing.OneHotEncoder(sparse=False)
ohe_example = ohe1.fit_transform(example3.reshape(-1,1))
print(f"Size one hot encoded dense array:{ohe_example.nbytes}")
#for sparse array
ohe2 = preprocessing.OneHotEncoder(sparse=True)
ohe_example2 = ohe2.fit_transform(example3.reshape(-1,1))
print(f"Size of one hot encoded sparse array:{ohe_example2.data.nbytes}")
print(f"bytes saved from using a one hot encoded sparse array:\
      {ohe_example.nbytes - ohe_example2.data.nbytes}bytes")