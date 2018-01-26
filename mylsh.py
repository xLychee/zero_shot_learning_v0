import numpy as np

class LSH:
    def __init__(self, input_dim, output_dim):
        self.planes = []
        for _ in range(output_dim):
            x = np.random.randn(input_dim)
            x = x/np.linalg.norm(x)
            self.planes.append(x)

    def indexing(self, input_vec):
        result = 0
        for x in self.planes:
            if x.dot(input_vec)>=0:
                result = result * 2 + 1
            else:
                result = result * 2 + 0
        assert result >= 0 and result < 2**len(self.planes)
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

