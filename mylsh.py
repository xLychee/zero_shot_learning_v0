import numpy as np
import scipy

class LSH:
    def __init__(self, input_dim, num_planes):
        self.planes = []
        for _ in range(num_planes):
            x = np.random.randn(input_dim)
            x = x/np.linalg.norm(x)
            self.planes.append(x)

    def indexing(self, input_vec):
        result = 0
        #if type(input_vec) == scipy.sparse.csr.csr_matrix:
        #    input_vec = input_vec.toarray().flatten()
        for x in self.planes:
            if input_vec.dot(x)>=0:
                result = result * 2 + 1
            else:
                result = result * 2 + 0
        assert result >= 0 and result < 2**len(self.planes)
        return result
'''
class LSH_family:
    def __init__(self, input_dim, output_dim, num_table):
        self.tables = []
        for _ in range(num_table):
            self.tables.append(LSH(input_dim,output_dim))

    def indexing(self, input_vec):
        result = []
        for table in self.tables:
            result.append(table.indxing(input_vec))
        return result
'''

