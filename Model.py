import torch
import torch.nn as nn
import math

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class FeatureAttentionLayer(nn.Module):
    def __init__(self, in_dim, num_head):
        super().__init__()

        self.num_head = num_head
        self.head_dim = in_dim // num_head
        self.W_Q = nn.Linear(in_dim, self.num_head * self.head_dim)
        self.W_K = nn.Linear(in_dim, self.num_head * self.head_dim)
        self.W_V = nn.Linear(in_dim, self.num_head * self.head_dim)
        self.W_O = nn.Linear(self.num_head * self.head_dim, in_dim)

        self.layerNorm1 = nn.LayerNorm(in_dim)
        self.lin = nn.Linear(in_dim, in_dim)
        self.layerNorm2 = nn.LayerNorm(in_dim)

    def forward(self, x):
        # stage1 计算多头注意力
        ## 计算Q,K,V获得注意分数矩阵
        ## [b,num_feature,in_dim] -> [b,num_feature,num_head*head_dim] -> [b,num_feature,num_head,head_dim] -> [b,num_head,num_feature,head_dim]
        batch_size, num_feature = x.shape[:2]
        Q = self.W_Q(x).view(batch_size, num_feature, self.num_head, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, num_feature, self.num_head, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, num_feature, self.num_head, self.head_dim).transpose(1, 2)
        ## [b,num_head,num_feature,num_feature]
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        ## 注意力分数矩阵
        attention_score = attention_score.softmax(dim=-1)
        ## [b,num_head,num_feature,head_dim]
        attention_out = torch.matmul(attention_score, V)
        attention_out = attention_out.transpose(1, 2).contiguous().view(batch_size, num_feature, self.num_head * self.head_dim)
        attention_out = self.W_O(attention_out)
        ## layerNorm + 残差
        x = self.layerNorm1(x + attention_out)

        # stage2
        x_lin = self.lin(x)
        x = self.layerNorm2(x + x_lin)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, num_head):
        super().__init__()

        self.num_head = num_head

        self.W_Q = nn.Linear(1, num_head)
        self.W_K = nn.Linear(1, num_head)
        self.W_V = nn.Linear(1, num_head)
        self.W_O = nn.Linear(num_head, 1)

        self.layerNorm1 = nn.LayerNorm(in_dim)
        self.lin = nn.Linear(in_dim, in_dim)
        self.layerNorm2 = nn.LayerNorm(in_dim)

    def forward(self, x):
        batch_size, num_feature = x.shape  # (b,num_feature)
        x = x.unsqueeze(dim=2)  # (b,num_feature,1)
        # (b,num_feature,num_head*1)->(b,num_feature,num_head,1)->(b,num_head,num_feature,1)
        Q = self.W_Q(x).view(batch_size, num_feature, self.num_head, 1).transpose(1, 2)
        K = self.W_K(x).view(batch_size, num_feature, self.num_head, 1).transpose(1, 2)
        V = self.W_V(x).view(batch_size, num_feature, self.num_head, 1).transpose(1, 2)
        d_k = Q.shape[-2]  # d_k在原论文中即是与head_dim同维度,本应用中head_dim只有1,所以采用输入长度为缩放因子
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attention_score = attention_score.softmax(dim=-1)  # (b,num_head,num_feature,num_feature)
        attention_out = torch.matmul(attention_score, V)
        attention_out = attention_out.transpose(1, 2).contiguous().view(batch_size, num_feature, self.num_head * 1)
        attention_out = self.W_O(attention_out).squeeze(dim=2)  # (b,num_feature,1)->(b,num_feature)

        # stage1
        x = self.layerNorm1(x.squeeze(dim=2) + attention_out)

        # stage2
        x_lin = self.lin(x)
        x = self.layerNorm2(x + x_lin)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_head):
        super().__init__()

        # attention
        self.featureAttention = FeatureAttentionLayer(in_dim, num_head)

        # encoder
        self.l1 = nn.Linear(in_dim, (in_dim + hidden_dim) // 2)
        self.bn1 = nn.BatchNorm1d(3)
        self.l2 = nn.Linear((in_dim + hidden_dim) // 2, hidden_dim)

        # decoder
        self.l3 = nn.Linear(hidden_dim, (in_dim + hidden_dim) // 2)
        self.bn3 = nn.BatchNorm1d(3)
        self.l4 = nn.Linear((in_dim + hidden_dim) // 2, in_dim)

        #
        self.dropout = nn.Dropout(p=dropout_rate)
        self.action = gelu

    def forward(self, x):
        AE_0 = self.featureAttention(x)
        #
        AE_1 = self.dropout(self.bn1(self.action(self.l1(AE_0))))
        AE_2 = self.l2(AE_1)

        #
        AE_3 = self.dropout(self.bn3(self.action(self.l3(AE_2))))
        AE_4 = self.l4(AE_3)
        return AE_2, AE_4  # , AE_0.detach()


class MDA(nn.Module):
    def __init__(self, data):
        super().__init__()

        #   stage.0
        self.mfs = nn.Embedding.from_pretrained(torch.FloatTensor(data.data['mfs']))
        self.mgs = nn.Embedding.from_pretrained(torch.FloatTensor(data.data['mgs']))
        self.mis = nn.Embedding.from_pretrained(torch.FloatTensor(data.data['mis']))
        self.dss = nn.Embedding.from_pretrained(torch.FloatTensor(data.data['dss']))
        self.dgs = nn.Embedding.from_pretrained(torch.FloatTensor(data.data['dgs']))
        self.dis = nn.Embedding.from_pretrained(torch.FloatTensor(data.data['dis']))

        #   stage.1 通过自编码器削减特征维度
        self.miRNA_AE = AutoEncoder(in_dim=data.data['num_of_mirna'], hidden_dim=ae_hidden_dim, num_head=n_head)
        self.disease_AE = AutoEncoder(in_dim=data.data['num_of_disease'], hidden_dim=ae_hidden_dim, num_head=n_head)

        #   stage.2

        #   stage.3
        self.l1 = nn.Linear(in_features=ae_hidden_dim * 3 * 4, out_features=ae_hidden_dim * 3 * 2)
        self.bn1 = nn.BatchNorm1d(ae_hidden_dim * 3 * 2)
        self.att1 = AttentionLayer(in_dim=ae_hidden_dim * 3 * 2, num_head=n_head)
        self.l2 = nn.Linear(in_features=ae_hidden_dim * 3 * 2, out_features=ae_hidden_dim * 3)
        self.bn2 = nn.BatchNorm1d(ae_hidden_dim * 3)
        self.att2 = AttentionLayer(in_dim=ae_hidden_dim * 3, num_head=n_head)
        self.l3 = nn.Linear(in_features=ae_hidden_dim * 3, out_features=1)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.action = gelu

    def forward(self, input):
        #   stage.0 组合当前特征
        miRNA_index, disease_index = input[:, 0], input[:, 1]
        mfs, mgs, mis = self.mfs(miRNA_index)[:, None, :], self.mgs(miRNA_index)[:, None, :], self.mis(miRNA_index)[:, None, :]
        dss, dgs, dis = self.dss(disease_index)[:, None, :], self.dgs(disease_index)[:, None, :], self.dis(disease_index)[:, None, :]
        miRNA_feature = torch.cat((mfs, mgs, mis), dim=1)
        disease_feature = torch.cat((dss, dgs, dis), dim=1)

        #   stage.1
        miRNA_hidden, miRNA_ae_out = self.miRNA_AE(miRNA_feature)
        disease_hidden, disease_ae_out = self.disease_AE(disease_feature)

        #   stage.2
        miRNA = miRNA_hidden.flatten(start_dim=1)
        disease = disease_hidden.flatten(start_dim=1)
        b1 = miRNA + disease
        b2 = torch.cat((miRNA, disease), dim=1)
        b3 = miRNA * disease
        b = torch.cat((b1, b2, b3), dim=1)

        #   stage.3
        out = self.dropout(self.bn1(self.action(self.l1(b))))
        out = self.att1(out)
        out = self.dropout(self.bn2(self.action(self.l2(out))))
        out = self.att2(out)
        out = self.l3(out)
        return out.sigmoid().flatten(), miRNA_feature, disease_feature, miRNA_ae_out, disease_ae_out
    

num_kfold = 10
batch_size = 256
epoch_num = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ae_hidden_dim = 64
dropout_rate = 0.3
n_head = 8