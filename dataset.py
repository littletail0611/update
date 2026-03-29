# 数据加载模块：解析 txt 文件，构建 PyTorch Geometric (PyG) 图或邻接矩阵

import os
import torch

class UKGDataset:
    def __init__(self, data_dir="data/CN15K"):
        self.data_dir = data_dir
        
        self.ent2id = {}
        self.rel2id = {}
        self.num_ent = 0
        self.num_rel = 0
        self.belief_state = {}
        self.new_entities = set()
        
        print(">>> 正在加载 Base 图谱数据...")
        # 去掉了硬编码的 .txt 后缀，交由 _load_file 自动判断
        self.base_train = self._load_file(os.path.join(data_dir, "base", "train"), is_inc=False)
        self.base_valid = self._load_file(os.path.join(data_dir, "base", "valid"), is_inc=False)
        self.base_test  = self._load_file(os.path.join(data_dir, "base", "test"),  is_inc=False)
        
        self.base_num_ent = self.num_ent 
        print(f"Base 阶段加载完成: 实体数={self.base_num_ent}, 关系数={self.num_rel}")
        
        print(">>> 正在加载 Inc 增量数据...")
        self.inc_train = self._load_file(os.path.join(data_dir, "inc", "train"), is_inc=True)
        self.inc_valid = self._load_file(os.path.join(data_dir, "inc", "valid"), is_inc=True)
        self.inc_test  = self._load_file(os.path.join(data_dir, "inc", "test"),  is_inc=True)
        # 布尔掩码：标记 inc_train 中哪些事实有已知置信度（第4元素不为 None）
        self.inc_labeled_mask = [f[3] is not None for f in self.inc_train]
        labeled_count = sum(self.inc_labeled_mask)
        print(f"Inc 阶段加载完成: 发现新实体数={len(self.new_entities)}, "
              f"有标注事实={labeled_count}/{len(self.inc_train)}")

    def _get_ent_id(self, ent, is_inc):
        if ent not in self.ent2id:
            self.ent2id[ent] = self.num_ent
            if is_inc:
                self.new_entities.add(self.num_ent)
            self.num_ent += 1
        return self.ent2id[ent]

    # def _get_rel_id(self, rel):
    #     if rel not in self.rel2id:
    #         self.rel2id[rel] = self.num_rel
    #         self.num_rel += 1
    #     return self.rel2id[rel]
    def _get_rel_id(self, rel):
        """获取关系ID，同时自动为其注册一个反向关系ID"""
        if rel not in self.rel2id:
            # 正常关系 ID
            self.rel2id[rel] = self.num_rel
            self.num_rel += 1
            
            # 立即为其注册反向关系 ID (命名约定：原关系名 + "_inv")
            inv_rel = rel + "_inv"
            self.rel2id[inv_rel] = self.num_rel
            self.num_rel += 1
            
        return self.rel2id[rel]

    def _load_file(self, filepath_prefix, is_inc):
        """自动适配 .txt 或 .tsv 后缀。
        
        对于增量文件 (is_inc=True)，支持混合格式：
          - 4列行 h\tr\tt\tconf：有标注事实，置信度已知
          - 3列行 h\tr\tt：无标注事实，置信度未知 (第4元素存为 None)
        非增量文件 (base) 仍要求 4 列。
        belief_state 仅在增量训练文件中由有标注事实填充。
        """
        triplets = []
        filepath = None
        
        if os.path.exists(filepath_prefix + ".txt"):
            filepath = filepath_prefix + ".txt"
        elif os.path.exists(filepath_prefix + ".tsv"):
            filepath = filepath_prefix + ".tsv"
        else:
            print(f"警告: 找不到文件 {filepath_prefix}.txt 或 .tsv")
            return triplets
            
        # 判断当前读的是不是训练集
        is_train = "train" in filepath_prefix
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                parts = line.split('\t')
                if len(parts) >= 4:
                    h_id = self._get_ent_id(parts[0], is_inc)
                    r_id = self._get_rel_id(parts[1])
                    t_id = self._get_ent_id(parts[2], is_inc)
                    c_val = float(parts[3])
                    
                    # 1. 加入正向边: h -> r -> t
                    triplets.append((h_id, r_id, t_id, c_val))
                    
                    # 2. 加入反向边: t -> r_inv -> h
                    # 反向关系 ID 必然是正向关系 ID + 1 (基于我们上面的分配逻辑)
                    r_inv_id = r_id + 1 
                    triplets.append((t_id, r_inv_id, h_id, c_val))
                    
                    if is_train:
                        # 记忆库也应该双向保存，保证状态一致性
                        self.belief_state[(h_id, r_id, t_id)] = c_val
                        self.belief_state[(t_id, r_inv_id, h_id)] = c_val

                elif len(parts) == 3 and is_inc:
                    # 无标注的增量事实：置信度未知，用 None 标记
                    h_id = self._get_ent_id(parts[0], is_inc)
                    r_id = self._get_rel_id(parts[1])
                    t_id = self._get_ent_id(parts[2], is_inc)

                    # 1. 加入正向边 (conf=None 表示无标注)
                    triplets.append((h_id, r_id, t_id, None))

                    # 2. 加入反向边
                    r_inv_id = r_id + 1
                    triplets.append((t_id, r_inv_id, h_id, None))

                    # 不写入 belief_state，等 updater 预测后再填充
                        
        return triplets

    def get_base_graph_data(self):
        # 防御性编程：如果数据没加载上来，直接返回空张量防止报错
        if not self.base_train:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.float)
            
        heads = [t[0] for t in self.base_train]
        tails = [t[2] for t in self.base_train]
        rels  = [t[1] for t in self.base_train]
        confs = [t[3] for t in self.base_train]
        
        edge_index = torch.tensor([heads, tails], dtype=torch.long)
        edge_type = torch.tensor(rels, dtype=torch.long)
        edge_confidence = torch.tensor(confs, dtype=torch.float)
        
        return edge_index, edge_type, edge_confidence

    def update_belief(self, fact_tuple, new_confidence):
        self.belief_state[fact_tuple] = new_confidence

    def get_incremental_batches(self, batch_size=1024):
        """产出增量训练事实的批次。
        
        每个元素为 (h, r, t, conf_or_None)，其中：
          - conf_or_None 为 float 时表示该事实有已知置信度（有标注）
          - conf_or_None 为 None 时表示该事实无标注，置信度需由模型预测
        """
        for i in range(0, len(self.inc_train), batch_size):
            yield self.inc_train[i:i + batch_size]