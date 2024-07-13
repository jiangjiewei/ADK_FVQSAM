import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim=2048, num_embeddings=10000, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost      #约束损失权重为：0.25

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)          #初始化嵌入层，形状为：（1000，2048）
        torch.nn.init.uniform_(self.embeddings.weight, 0, 3)                            #使用均匀分布从0-3初始化

    def forward(self, x, target=None):#接受输入张量和目标张量
        encoding_indices = self.get_code_indices(x, target)#使用get_code_indices方法获取输入向量的离散编码
        quantized = self.quantize(encoding_indices)#使用quantize 方法对输出向量量化
        # weight, encoding_indices = self.get_code_indices(x)
        # quantized = self.quantize(weight, encoding_indices)

        if not self.training:
            return quantized, encoding_indices

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # print("??????????????????",loss)

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices

    def get_code_indices(self, flat_x, target=None):
        # flag = self.training
        flat_x = F.normalize(flat_x, p=2, dim=1)
        weight = self.embeddings.weight
        weight = F.normalize(weight, p=2, dim=1)
        flag = False
        if flag:
            # print(target.dtype)
            # raise ValueError("target type error! ")
            encoding_indices = target
        else:
            # compute L2 distance
            distances = (
                    torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                    torch.sum(weight ** 2, dim=1) -
                    2. * torch.matmul(flat_x, weight.t())
            )  # [N, M]
            # print("hehe")
            # dis, encoding_indices = distances.topk(k=10)
            # index = F.gumbel_softmax(distances, tau=1, hard=False)
            # encoding_indices = torch.argmin(index, dim=1)  # [N,]
            encoding_indices = torch.argmin(distances, dim=1)  # [N,]
            # weight = F.softmax(dis / 2, dim=1)
        return encoding_indices
        # return weight, encoding_indices

    # def quantize(self, weight, encoding_indices):
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        # b, k = weight.size()
        # self.embeddings(encoding_indices)
        # quantized = torch.stack(
        #     [torch.index_select(input=self.embeddings.weight, dim=0, index=encoding_indices[i, :]) for i in range(b)])
        # weight = weight.view(b, 1, k).contiguous()
        # quantized = torch.bmm(weight, quantized).view(b, -1).contiguous()
        # return quantized
        return self.embeddings(encoding_indices)

#新增pqrq量化
# class PQRQuantizer(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, subvectors=4, commitment_cost=0.25):
#         super(PQRQuantizer, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.subvectors = subvectors
#         self.commitment_cost = commitment_cost
#
#         assert self.embedding_dim % self.subvectors == 0, "embedding_dim must be divisible by subvectors"
#
#         self.subvector_dim = self.embedding_dim // self.subvectors
#
#         # Initialize codebook and residuals
#         self.codebook = nn.Parameter(torch.randn(self.num_embeddings, self.subvectors * self.subvector_dim))
#         self.residuals = nn.Parameter(torch.randn(self.subvectors * self.subvector_dim))
#
#     def forward(self, inputs, target=None):
#         # Split input tensor into subvectors
#         inputs = inputs.view(-1, self.subvectors * self.subvector_dim)
#
#         # Compute distances to codebook
#         distances = torch.cdist(inputs.unsqueeze(1), self.codebook.unsqueeze(0), p=2)
#
#         # Find nearest embeddings (indices)
#         indices = torch.argmin(distances, dim=-1)  # Shape: [batch_size, 1]
#
#         # Quantize input to nearest embeddings
#         quantized = self.codebook[indices.squeeze()]
#
#         if not self.training:
#             return quantized, indices
#
#         # Compute residuals
#         residuals = inputs.unsqueeze(1) - quantized
#
#         # Add residuals to codebook vectors
#         quantized += self.residuals
#
#         # Reshape quantized tensor
#         quantized = quantized.view(-1, self.embedding_dim)
#
#         # Compute loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss
#
#         # Return quantized tensor and loss
#         return quantized, loss, indices

class PQRQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, subvectors=4, commitment_cost=0.25):
        super(PQRQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.subvectors = subvectors
        self.commitment_cost = commitment_cost

        assert self.embedding_dim % self.subvectors == 0, "embedding_dim must be divisible by subvectors"

        self.subvector_dim = self.embedding_dim // self.subvectors

        # Initialize codebook and residuals
        self.codebook = nn.Parameter(torch.randn(self.num_embeddings, self.subvectors * self.subvector_dim))
        self.residuals = nn.Parameter(torch.randn(self.subvectors * self.subvector_dim))

    def forward(self, inputs, target=None):
        # Split input tensor into subvectors
        inputs = inputs.view(-1, self.subvectors, self.subvector_dim)

        # Compute distances to codebook
        # distances = torch.cdist(inputs, self.codebook.view(self.num_embeddings, self.subvectors, self.subvector_dim))
        # 在 forward 方法中，修改计算距离的部分
        distances = torch.cdist(inputs.view(-1, self.subvectors * self.subvector_dim), self.codebook)

        # Find nearest embeddings (indices)
        indices = torch.argmin(distances, dim=1)  # Shape: [batch_size, subvectors]

        # Quantize input to nearest embeddings
        quantized = self.codebook[indices.view(-1)].view(inputs.size())


        # Compute residuals
        residuals = inputs - quantized

        # Add residuals to codebook vectors
        quantized += residuals

        # Reshape quantized tensor
        quantized = quantized.view(-1, self.embedding_dim)

        if not self.training:
            return quantized, indices

        # Compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs.view(-1, self.embedding_dim))
        q_latent_loss = F.mse_loss(quantized, inputs.view(-1, self.embedding_dim).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Return quantized tensor and loss
        return quantized, loss, indices


# resnet版本
# class new_clsnet(nn.Module):
#     def __init__(self, model):
#         super(new_clsnet, self).__init__()
#         # 将模型的所有层除了最后两层（全局平均池化和全连接层）提取出来
#         self.features = nn.Sequential(*list(model.children())[:-2])
#         # 添加一个自适应全局平均池化层
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         # 添加一个全连接层
#         self.fc = nn.Linear(model.fc.in_features, 3)  # 输出类别数为 3
#
#         # 初始化 PQRQuantizer
#         self.codebook = PQRQuantizer(num_embeddings=1000, embedding_dim=model.fc.in_features, subvectors=4,
#                                      commitment_cost=0.25)
#
#         # 初始化注意力机制参数
#         self.K = nn.Parameter(torch.FloatTensor(model.fc.in_features, model.fc.in_features), requires_grad=True)
#         self.Q = nn.Parameter(torch.FloatTensor(model.fc.in_features, model.fc.in_features), requires_grad=True)
#         self.V = nn.Parameter(torch.FloatTensor(model.fc.in_features, model.fc.in_features), requires_grad=True)
#         nn.init.kaiming_normal_(self.K)
#         nn.init.kaiming_normal_(self.Q)
#         nn.init.kaiming_normal_(self.V)
#
#         # 初始化融合层参数
#         self.fc_fuse = nn.Sequential(nn.Linear(self.fc.in_features * 2, self.fc.in_features),
#                                      nn.ReLU(True))
#
#     def forward(self, x, target=None):
#         bs = x.shape[0]
#         x = self.features(x)
#         x = self.avgpool(x)
#         feat = torch.flatten(x, 1)
#
#         # 非训练模式下的处理
#         if not self.training:
#             # 使用 PQRQuantizer 进行向量量化
#             quantized, index = self.codebook(feat)
#             # 对量化结果进行处理
#             fuse = torch.stack([quantized, feat], dim=2)
#             K = torch.bmm(self.K.repeat(bs, 1, 1), fuse)
#             Q = torch.bmm(self.Q.repeat(bs, 1, 1), fuse)
#             V = torch.bmm(self.V.repeat(bs, 1, 1), fuse)
#             A = F.softmax(torch.bmm(K.permute(0, 2, 1), Q), dim=1)
#             fuse = torch.bmm(V, A).permute(0, 2, 1).reshape(bs, -1).contiguous()
#             fuse = self.fc_fuse(fuse)
#             pred = self.fc(fuse)
#             # print('hhh')
#             return pred
#
#         # 训练模式下的处理
#         quantized, e_q_loss, index = self.codebook(feat, target)
#         fuse = torch.stack([quantized, feat], dim=2)
#         K = torch.bmm(self.K.repeat(bs, 1, 1), fuse)
#         Q = torch.bmm(self.Q.repeat(bs, 1, 1), fuse)
#         V = torch.bmm(self.V.repeat(bs, 1, 1), fuse)
#         A = F.softmax(torch.bmm(K.permute(0, 2, 1), Q) / torch.sqrt(torch.tensor(2.0).to(fuse.device)), dim=1)
#         fuse = torch.bmm(V, A).permute(0, 2, 1).reshape(bs, -1).contiguous()
#         fuse = self.fc_fuse(fuse)
#         pred = self.fc(fuse)
#         # 计算交叉熵损失
#         device = torch.device("cuda")
#         criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.5, 1.0, 3.0])).to(device)
#         ce_loss = criterion(pred, target)
#         return e_q_loss, ce_loss, pred

# densenet121版本
class new_clsnet(nn.Module):
    def __init__(self, model):
        super(new_clsnet, self).__init__()
        self.densenet_layer = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(model.classifier.in_features, 3)
        # 初始化 PQRQuantizer
        self.codebook = PQRQuantizer(num_embeddings=1000, embedding_dim=model.classifier.in_features, subvectors=4, commitment_cost=0.25)

        self.K = nn.Parameter(torch.FloatTensor(model.classifier.in_features, model.classifier.in_features), requires_grad=True)
        self.Q = nn.Parameter(torch.FloatTensor(model.classifier.in_features, model.classifier.in_features), requires_grad=True)
        self.V = nn.Parameter(torch.FloatTensor(model.classifier.in_features, model.classifier.in_features), requires_grad=True)
        nn.init.kaiming_normal_(self.K)
        nn.init.kaiming_normal_(self.Q)
        nn.init.kaiming_normal_(self.V)

        self.fc_fuse = nn.Sequential(nn.Linear(self.fc.in_features * 2, self.fc.in_features),
                                     nn.ReLU(True))

    def forward(self, x, target=None):
        bs = x.shape[0]
        x = self.densenet_layer(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if not self.training:
            # 使用 PQRQuantizer 进行向量量化
            quantized, index = self.codebook(feat)
            # 在 forward 方法中，修改堆叠张量的部分
            # quantized = quantized.view(-1, self.subvectors * self.subvector_dim)  # 调整 quantized 的形状
            # feat = feat.view(-1, self.subvectors * self.subvector_dim)  # 调整 feat 的形状
            fuse = torch.stack([quantized, feat], dim=2)  # b,d,2
            # fuse = torch.stack([quantized, feat], dim=2)  # b,d,2
            K = torch.bmm(self.K.repeat(bs, 1, 1), fuse)  # b,d,2
            Q = torch.bmm(self.Q.repeat(bs, 1, 1), fuse)  # b.d.2
            V = torch.bmm(self.V.repeat(bs, 1, 1), fuse)  # b,d,2
            A = F.softmax(torch.bmm(K.permute(0, 2, 1), Q), dim=1)  # b,2,2
            fuse = torch.bmm(V, A).permute(0, 2, 1).reshape(bs, -1).contiguous()
            fuse = self.fc_fuse(fuse)
            pred = self.fc(fuse)
            return pred

        quantized, e_q_loss, index = self.codebook(feat, target)

        fuse = torch.stack([quantized, feat], dim=2)  # b,d,2
        K = torch.bmm(self.K.repeat(bs, 1, 1), fuse)  # b,d,2
        Q = torch.bmm(self.Q.repeat(bs, 1, 1), fuse)  # b.d.2
        V = torch.bmm(self.V.repeat(bs, 1, 1), fuse)  # b,d,2
        A = F.softmax(torch.bmm(K.permute(0, 2, 1), Q) / torch.sqrt(torch.tensor(2.0).to(fuse.device)), dim=1)  # b,2,2
        fuse = torch.bmm(V, A).permute(0, 2, 1).reshape(bs, -1).contiguous()
        fuse = self.fc_fuse(fuse)
        pred = self.fc(fuse)
        device = torch.device("cuda")  # Use all available GPUs
        # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.5, 1.1, 3.0])).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        ce_loss = criterion(pred, target)
        return e_q_loss, ce_loss, pred


if __name__ == '__main__':
    model = models.densenet121(pretrained=True)
    model = new_clsnet(model)
    print(model)
