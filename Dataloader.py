from torch.utils.data import Dataset, DataLoader
class miRNA_disease_Dataset(Dataset):
    def __init__(self, miRNA_disease_index, miRNA_disease_label):
        self.md_ij = torch.tensor(miRNA_disease_index).long()
        self.md_ij_label = torch.tensor(miRNA_disease_label).float()
        self.len = miRNA_disease_label.shape[0]

    def __getitem__(self, index):
        return self.md_ij[index], self.md_ij_label[index]

    def __len__(self):
        return self.len