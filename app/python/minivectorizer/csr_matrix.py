import numpy as np

# Simplified sparse matrix
# See https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
class MiniSparseMatrix():
    def __init__(self, data, indices, indptr, shape, dtype=np.int32):
        self.data = data.astype(dtype=dtype)
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def __str__(self):
        string = ""

        # Nested mapping to resolve combined feature counting in documents
        # Using indptr index as base
        combinedFeatures = sum(
                map(lambda a:
                    list(map(lambda b:
                        ((a, self.indices[b]), self.data[b]),
                        range(self.indptr[a], self.indptr[a+1])))
                , range(len(self.indptr) - 1))
            , [])

        for f in combinedFeatures:
            fline = "%s\t\t%s\n" % (str(f[0]), f[1])
            string = string + fline

        string += str(self.data) + "\n"
        string += str(self.indices) + "\n"
        string += str(self.indptr) + "\n"

        return string
        

    def sort_indices(self):
        pass