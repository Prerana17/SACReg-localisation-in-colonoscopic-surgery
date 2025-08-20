from sklearn.metrics.pairwise import cosine_similarity
# from model import IRModel, GeM
# from dataset import ColonoscopyTripletDataset # 用于加载测试数据，需要修改为不生成三元组的模式

# 评估时使用的 Dataset，只加载单张图片
class InferenceDataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels # 用于评估的地面实况标签 (例如: video_id, case_id)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.img_labels[idx]

def get_descriptors(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    
    descriptors = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting Descriptors"):
            imgs = imgs.to(device)
            desc = model(imgs)
            descriptors.append(desc.cpu().numpy())
            labels.extend(lbls.cpu().numpy() if isinstance(lbls, torch.Tensor) else lbls)
            
    return np.vstack(descriptors), np.array(labels)

def evaluate_retrieval(query_descriptors, gallery_descriptors, query_labels, gallery_labels):
    # 计算查询描述符与画廊描述符之间的余弦相似度
    # （因为模型输出的特征向量是归一化的，欧氏距离与余弦相似度等价，但余弦相似度更直观）
    similarities = cosine_similarity(query_descriptors, gallery_descriptors)
    
    rank1_count = 0
    aps = [] # 用于计算mAP

    for i in range(len(query_labels)):
        query_label = query_labels[i]
        
        # 获取与当前查询相关的所有画廊项的索引
        # 这里定义“相关”为与查询具有相同标签（例如，同一视频ID，同一病例ID）的画廊项
        relevant_gallery_indices = [j for j, label in enumerate(gallery_labels) if label == query_label]
        
        # 排除查询本身在画廊中的情况 (如果查询本身也在画廊中)
        if query_descriptors[i] in gallery_descriptors: # 这是一个概念性的检查，实际用索引更准确
            # 找到查询自身在 gallery 中的索引并排除
            # 更准确的做法是，如果 query 和 gallery 是同一个数据集，那么查询自己的索引应该被排除
            # For exact matches on identity, you need to pass original image IDs.
            # For simplicity, assume query is distinct from gallery items for rank1.
            pass # 具体排除逻辑取决于你的 query/gallery 准备方式

        # 按相似度降序排列画廊项的索引
        sorted_indices = np.argsort(similarities[i])[::-1]
        
        # === 计算 RANK-1 ===
        # 遍历排序后的结果，找到第一个相关的项
        found_rank1 = False
        for k, gallery_idx in enumerate(sorted_indices):
            if gallery_labels[gallery_idx] == query_label: # 假设标签匹配即相关
                # 如果这个相关项不是查询自身 (如果查询可能出现在画廊中)
                # 例如，如果 query_labels 和 gallery_labels 是文件名，你可以比较文件名。
                # 由于这里只有标签，我们假设只要标签匹配就相关
                if k == 0: # 如果第一个就是相关的
                    rank1_count += 1
                found_rank1 = True
                break # 找到第一个相关项即可退出循环

        # === 计算 Average Precision (AP) ===
        num_relevant_found = 0
        sum_precisions = 0
        
        for k, gallery_idx in enumerate(sorted_indices):
            if gallery_labels[gallery_idx] == query_label: # 如果当前画廊项是相关的
                num_relevant_found += 1
                precision_at_k = num_relevant_found / (k + 1) # 精确率@k
                sum_precisions += precision_at_k
        
        if num_relevant_found > 0:
            ap = sum_precisions / num_relevant_found
            aps.append(ap)
        else:
            aps.append(0.0) # 如果没有相关项，AP 为 0

    rank1 = rank1_count / len(query_labels)
    mAP = np.mean(aps) if len(aps) > 0 else 0.0

    return rank1, mAP