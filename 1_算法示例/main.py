import torch
import torch.nn as nn
import torch.optim as optim
import os


class RESCAL(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RESCAL, self).__init__()

        # 实体嵌入：每个实体有一个向量
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # 关系嵌入：每个关系是一个矩阵
        self.relation_embeddings = nn.Parameter(torch.randn(num_relations, embedding_dim, embedding_dim))

        # 偏置项
        self.relation_bias = nn.Parameter(torch.zeros(num_relations))

        # Sigmoid 激活函数，用于将得分限制在 [0, 1] 范围内
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, r, t):
        """
        h, r, t: 对应头实体、关系和尾实体的索引
        """
        # 获取头实体、尾实体的嵌入表示
        h_emb = self.entity_embeddings(h)
        t_emb = self.entity_embeddings(t)

        # 获取关系的矩阵嵌入
        r_emb = self.relation_embeddings[r]

        # 计算得分：h^T * R_r * t
        score = torch.sum(h_emb * torch.matmul(r_emb, t_emb.T), dim=-1) + self.relation_bias[r]

        # 使用sigmoid函数将得分限制在 [0, 1] 范围内
        return self.sigmoid(score)


# 损失函数，包括正则化项
def loss_function(model, triplets, lambda_reg=1e-5):
    # 总损失初始化
    total_loss = 0.0

    # 损失计算
    for h, r, t, label in triplets:
        # 得到三元组的得分
        score = model(h, r, t)

        # 计算损失：二元交叉熵
        loss = (score - label) ** 2

        # 累加损失
        total_loss += loss

    # 正则化项（L2正则化）
    entity_embeddings = model.entity_embeddings.weight
    relation_embeddings = model.relation_embeddings
    entity_norm = torch.norm(entity_embeddings, p=2)
    relation_norm = sum(torch.norm(r, p=2) for r in relation_embeddings)

    # 总损失
    total_loss += lambda_reg * (entity_norm + relation_norm)

    return total_loss


# 西游记实体集
entity_to_id = {
    "孙悟空": 0, "唐僧": 1, "猪八戒": 2, "沙僧": 3, "如来佛": 4, "西天": 5,
    "白龙马": 6, "观音菩萨": 7, "东海龙王": 8, "牛魔王": 9, "花果山": 10, "天宫": 11,
    "东土大唐": 12, "白虎山": 13, "火焰山": 14
}

# 西游记关系集
relation_to_id = {
    "是朋友": 0, "师傅": 1, "是佛": 2, "去往": 3, "居住在": 4, "结拜兄弟": 5,
    "战斗": 6, "保护": 7, "封印": 8
}

# 西游记三元组数据（格式：头实体, 关系, 尾实体, 标签）
triplets = [
    # 是朋友
    (0, 0, 1, 1),  # (孙悟空, 是朋友, 唐僧)
    (2, 0, 1, 1),  # (猪八戒, 是朋友, 唐僧)
    (3, 0, 1, 1),  # (沙僧, 是朋友, 唐僧)
    (6, 0, 1, 1),  # (白龙马, 是朋友, 唐僧)

    # 师傅
    (1, 1, 0, 1),  # (唐僧, 师傅, 孙悟空)
    (1, 1, 2, 1),  # (唐僧, 师傅, 猪八戒)
    (1, 1, 3, 1),  # (唐僧, 师傅, 沙僧)

    # 是佛
    (4, 2, 1, 1),  # (如来佛, 是佛, 唐僧)
    (4, 2, 0, 1),  # (如来佛, 是佛, 孙悟空)

    # 去往
    (0, 3, 5, 1),  # (孙悟空, 去往, 西天)
    (1, 3, 5, 1),  # (唐僧, 去往, 西天)
    (2, 3, 5, 1),  # (猪八戒, 去往, 西天)

    # 居住在
    (10, 4, 0, 1),  # (花果山, 居住在, 孙悟空)
    (12, 4, 1, 1),  # (东土大唐, 居住在, 唐僧)

    # 结拜兄弟
    (0, 5, 2, 1),  # (孙悟空, 结拜兄弟, 猪八戒)
    (0, 5, 3, 1),  # (孙悟空, 结拜兄弟, 沙僧)

    # 战斗
    (0, 6, 9, 1),  # (孙悟空, 战斗, 牛魔王)
    (2, 6, 9, 1),  # (猪八戒, 战斗, 牛魔王)

    # 保护
    (1, 7, 0, 1),  # (唐僧, 保护, 孙悟空)

    # 封印
    (4, 8, 9, 1),  # (如来佛, 封印, 牛魔王)
    (4, 8, 2, 1)  # (如来佛, 封印, 猪八戒)
]

# 转换为Tensor
triplets_tensor = torch.tensor(triplets)

# 设置超参数
embedding_dim = 128  # 嵌入维度
num_entities = 15  # 15个实体
num_relations = 9  # 9种关系
learning_rate = 0.01
num_epochs = 1000  # 训练1000次
lambda_reg = 1e-5  # 正则化参数

# 模型保存路径
model_path = "rescal_model.pth"

# 创建RESCAL模型
def create_model():
    return RESCAL(num_entities, num_relations, embedding_dim)

# 训练模型并保存
def train_and_save_model():
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()

        loss = loss_function(model, triplets_tensor, lambda_reg)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至{model_path}")
    return model

# 加载现有模型
def load_model():
    model = create_model()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("加载模型中")
    else:
        print("未发现模型，开始训练模型")
        model = train_and_save_model()
    return model

# 测试三元组得分
def test_triplet_score(model, h, r, t):
    # 计算得分
    score = model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t]))

    # 获取实体和关系的真实名称
    h_name = list(entity_to_id.keys())[list(entity_to_id.values()).index(h)]
    t_name = list(entity_to_id.keys())[list(entity_to_id.values()).index(t)]
    r_name = list(relation_to_id.keys())[list(relation_to_id.values()).index(r)]

    # 输出得分，注意score[0, 0]获取的是得分的数值
    print(f"三元组(<{h_name}, {r_name}, {t_name}>)的得分为: {score[0, 0].item()}")

# def test_triplet_score(model, h, r, t):
#     score = model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t]))
#     print(f"三元组(<{h}, {r}, {t}>)的得分为: {score[0, 0].item()}")


# 主函数
def main():

    model = load_model()
    # 测试模型,计算三元组“<孙悟空，师傅，唐僧>”的得分
    test_triplet_score(model, 0, 1, 1)  # (孙悟空, 师傅, 唐僧)

if __name__ == "__main__":
    main()
