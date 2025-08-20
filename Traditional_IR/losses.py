class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # 计算欧氏距离
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet Loss 公式: max(0, d(A,P) - d(A,N) + margin)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() # 返回批次内所有损失的平均值