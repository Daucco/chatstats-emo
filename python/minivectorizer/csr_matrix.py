import numpy as np

# Simplified sparse document matrix
# See https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
class DocSparseMatrix():
    def __init__(self, data, indices, indptr, dtype=np.int32):
        self.data = data.astype(dtype=dtype)
        self.indices = indices
        self.indptr = indptr
        self.shape = (len(indptr) - 1, len(indices)) # data.len == indices.len in doc matrix
        self.has_sorted_indices = False

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        self._shape = val

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
        
    def _document_frequency(self):
        """
            Counts the number of document ocurrences for each feature (encoded as an index)
            Ignores how many times appears each term in each doc
        """
        return np.bincount(self.indices, minlength=len(self.indices))


    def sort_indices(self):
        """
            Sorts matrix indices in place and per ptr

            0 1 2 3 4 - range(len indices)
            3 4 1 0 2 - indices

            (sort indices once)

            3 2 4 0 1 - unsorted range from sorted indices. Use as template to map both, indices and data
            0 1 2 3 4 - sorted indices

            ---

            3 4 1 0 2 - indices
            a b c d e - data

            3 2 4 0 1 - mapping

            (map indices and data)
            
            0 1 2 3 4 - sorted indices
            d c e a b - data from sorted indices
        """
        if not self.has_sorted_indices:
            mapping = []
            for i in range(len(self.indptr) - 1):
                row_start = self.indptr[i]
                row_end = self.indptr[i + 1]
                
                # Resolves partial sort for each row, then aggregates it to a global mapping list
                partial_mapping = np.argsort(self.indices[row_start:row_end]) + row_start
                mapping.extend(partial_mapping)

            # Applies map
            mapping = np.asarray(mapping)
            self.indices = self.indices[mapping]
            self.data = self.data[mapping]
            self.has_sorted_indices = True