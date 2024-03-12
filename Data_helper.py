import numpy as np
import math
version = 2
hmdd20_folder_path = r'Data/HMDD v2.0/'
hmdd32_folder_path = r'Data/HMDD v3.2/'
class Data_help():
    def __init__(self, version):
        self.data_folder_path = hmdd20_folder_path if version == 2 else hmdd32_folder_path
        self.data = dict()

        self.data['miRNA'] = np.loadtxt(self.data_folder_path + 'miRNA number.txt', delimiter='\t', dtype=str)
        self.data['disease'] = np.loadtxt(self.data_folder_path + 'disease number.txt', delimiter='\t', dtype=str)

        # known miRNA-disease association
        self.data['known_md'] = np.loadtxt(self.data_folder_path + 'known miRNA-disease association.txt', dtype=int) - 1

        # miRNA functional similarity
        self.data['mfs'] = np.loadtxt(self.data_folder_path + 'miRNA functional similarity matrix.txt', dtype=float)

        # disease semantic similarity
        if version == 2:
            ds1 = np.loadtxt(self.data_folder_path + 'disease semantic similarity matrix 1.txt', dtype=float)
            ds2 = np.loadtxt(self.data_folder_path + 'disease semantic similarity matrix 2.txt', dtype=float)
            self.data['dss'] = (ds1 + ds2) / 2
        else:
            self.data['dss'] = np.loadtxt(self.data_folder_path + 'disease semantic similarity matrix.txt', dtype=float)

        # 统计数量
        self.data['num_of_mirna'] = self.data['miRNA'].shape[0]
        self.data['num_of_disease'] = self.data['disease'].shape[0]
        self.data['num_of_known_association'] = self.data['known_md'].shape[0]
        self.data['num_of_unknown_association'] = (self.data['num_of_mirna'] * self.data['num_of_disease']) - self.data['known_md'].shape[0]

        # 构建关系邻接矩阵
        self.data['Adj'] = np.zeros((self.data['num_of_mirna'], self.data['num_of_disease']), dtype=int)
        for i in range(self.data['num_of_known_association']):
            self.data['Adj'][self.data['known_md'][i, 0], self.data['known_md'][i, 1]] = 1

        # miRNA gaussian similarity
        self.data['mgs'] = self.get_gaussian(self.data['Adj'])

        # disease gaussian similarity
        self.data['dgs'] = self.get_gaussian(self.data['Adj'].transpose())

        # miRNA integrated similarity
        self.data['mis'] = 1 * (self.data['mfs'] > 0) * self.data['mfs'] + 1 * (self.data['mfs'] == 0) * self.data['mgs']

        # disease integrated similarity
        self.data['dis'] = 1 * (self.data['dss'] > 0) * self.data['dss'] + 1 * (self.data['dss'] == 0) * self.data['dgs']

        # edge index
        self.data['labeled_index'] = np.argwhere(self.data['Adj'] == 1)
        self.data['unlabeled_index'] = np.argwhere(self.data['Adj'] == 0)

    def recalculate_integrated_similarity_by_weight(self, alpha=None, beta=None):
        '''
        不同论文中的integrated similarity计算方式不一样
        例如在AEMDA中integrated similarity是通过对miRNA function similarity和miRNA gaussian similarity赋予不同的权重
        :param alpha:两种miRNA特征的融合比例,alpha*x1 + (1-alpha)*x2,若为None则不重新计算该特征
        :param beta:两种disease特征的融合比例,beta*x1 + (1-beta)*x2,若为None则不重新计算该特征
        '''
        if alpha is not None:
            self.data['mis'] = alpha * self.data['mfs'] + (1 - alpha) * self.data['mgs']

        if beta is not None:
            self.data['dis'] = beta * self.data['dss'] + (1 - beta) * self.data['dgs']

    def get_gaussian(self, adj):
        Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float32)
        gamaa = 1
        sumnorm = 0
        for i in range(adj.shape[0]):
            norm = np.linalg.norm(adj[i]) ** 2
            sumnorm = sumnorm + norm
        gama = gamaa / (sumnorm / adj.shape[0])
        for i in range(adj.shape[0]):
            for j in range(adj.shape[0]):
                Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))
        return Gaussian