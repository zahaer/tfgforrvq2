import os
import json
import torch
import random
from collections import defaultdict

dataset_ = 'FB15K-237N'
# dataset_ = 'CoDeX-S'
p_train_file = f'data/{dataset_}-train.json'
p_valid_file = f'data/{dataset_}-valid.json'
p_test_file = f'data/{dataset_}-test.json'

# RQ-VAE参数
seq_len = 16
code_num = 1024
num_quantizers = 4  # RQ-VAE量化器数量
ratio = 16  # 正负样本比例
original_r = 3  # kopa原始比例 fb3 codex4

out_dir = f'data/data4llm/{dataset_}_{seq_len}_{code_num}_{num_quantizers}_rq_r{ratio}_nocode/'
# RQ-VAE代码文件路径
code_file = f'/raid/hpc/qika/24project/Encoder/KGENC/MyKGVQ/Class/codes_new/{dataset_}_{seq_len}_{code_num}_{num_quantizers}_rq.pt'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def load_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

p_train_data = load_json(p_train_file)  # kopa数据
p_valid_data = load_json(p_valid_file)
p_test_data = load_json(p_test_file)
print('KoPA数据：')
print(len(p_train_data))
print(len(p_valid_data))
print(len(p_test_data))

# 加载RQ-VAE代码
codes = torch.load(code_file).numpy()  # 形状: [num_entities, seq_len, num_quantizers]
print(f"RQ codes shape: {codes.shape}")

hr2t_dic = defaultdict(list)
train_data_file = f'data/{dataset_}/train2id.txt'  # 原始数据
valid_data_file = f'data/{dataset_}/valid2id.txt'
test_data_file = f'data/{dataset_}/test2id.txt'

def load_triple(file_name):
    flag = 0
    data_ = []
    with open(file_name, 'r') as f:
        for line in f:
            if flag == 0:
                flag = 1
            else:
                h, t, r = [int(item) for item in line.strip().split(' ')]
                hr2t_dic[(h, r)].append(t)
                data_.append([h, r, t])
    return data_

train_data = load_triple(train_data_file)
valid_data = load_triple(valid_data_file)
test_data = load_triple(test_data_file)
print('原始数据：')
print(len(train_data))
print(len(valid_data))
print(len(test_data))

def get_relname():
    rel_list = []
    with open(f'data/{dataset_}/relation2id.txt') as f:
        flag = 0
        for line in f:
            if flag == 0:
                flag += 1
                continue
            line_list = line.strip().split('\t')
            a, b = line_list
            rel_list.append(a)
    return rel_list

def get_codex_relname():
    rel_list = []
    with open(f'data/{dataset_}/relation2id.txt') as f:
        flag = 0
        for line in f:
            if flag == 0:
                flag += 1
                continue
            line_list = line.strip().split('\t')
            a, b = line_list
            b = int(b)
            rel_list.append(a)
    rel_list_name = []
    with open(f'data/{dataset_}/relations.json') as f:
        rel_json = json.load(f)
    for rel in rel_list:
        rel_list_name.append(rel_json[rel]['label'])
    return rel_list_name

def get_entname():
    entityname_list = []
    ent_list = []
    with open(f'data/{dataset_}/entity2id.txt') as f:
        flag = 0
        for line in f:
            if flag == 0:
                flag += 1
                continue
            line_list = line.strip().split('\t')
            a, b = line_list
            b = int(b)
            ent_list.append(a)

    if 'FB' in dataset_:
        entid2name_dic = {}
        with open(f'data/{dataset_}/entity2text.txt') as f:
            for line in f:
                line_list = line.strip().split('\t')
                a, b = line_list
                entid2name_dic[a] = b
        for id_ in ent_list:
            entityname_list.append(entid2name_dic[id_])
    else:
        with open(f'data/{dataset_}/entities.json') as f:
            ent_json = json.load(f)
        for id_ in ent_list:
            entityname_list.append(ent_json[id_]['label'])
    return entityname_list

if 'FB' in dataset_:
    relname_list = get_relname()
else:
    relname_list = get_codex_relname()

entname_list = get_entname()

def get_neg_data(data_, mode='train'):
    ent_list = list(range(len(entname_list)))
    if mode == 'train':
        rrr = ratio - original_r  # 原始为4
        if ratio <= original_r:
            return []
    else:
        rrr = 1
    new_data = []
    for item in data_:
        h, r, t = item
        true_t_list = hr2t_dic[(h, r)]
        a = random.sample(list(set(ent_list)-set(true_t_list)), rrr)
        for ent in a:
            new_data.append([h, r, ent])
    return new_data

neg_train = get_neg_data(train_data)   # 增加负样本

def gen_rq_code_str(code_list):
    """
    生成RQ-VAE代码字符串
    Args:
        code_list: 形状为 [seq_len, num_quantizers] 的代码列表
    Returns:
        格式化的代码字符串
    """
    code_strs = []
    for i in range(code_list.shape[0]):  # seq_len
        level_codes = []
        for j in range(code_list.shape[1]):  # num_quantizers
            level_codes.append(f'[Q{j}_{code_list[i, j]}]')
        code_strs.append(' '.join(level_codes))
    return ' | '.join(code_strs)

def get_prompt(h, r, t):
    h_name = entname_list[h]
    r_name = relname_list[r]
    t_name = entname_list[t]

    prompt = 'Given a triple in the knowledge graph, you need to predict its validity based on the triple itself.\n'
    prompt += 'The triple is: ({}, {}, {})\n'.format(h_name, r_name, t_name)

    # 添加RQ-VAE量化表示（可选）
    # h_code_str = gen_rq_code_str(codes[h])
    # t_code_str = gen_rq_code_str(codes[t])
    # prompt += f'The residual quantized representation of entity "{h_name}" is: {h_code_str}\n'
    # prompt += f'The residual quantized representation of entity "{t_name}" is: {t_code_str}\n'

    prompt += 'Please determine the validity of the triple and response True or False.'
    return prompt

def construct_data(data=p_test_data, mode='train'):
    train_ture_num = 0
    train_false_num = 0
    data_new = []
    for item in data:
        temp_json = {}
        output = item['output']
        h, r, t = item['embedding_ids']
        temp_json['prompt'] = get_prompt(h, r, t)
        temp_json['completion'] = output
        if mode == 'train' and output == 'False' and ratio < original_r:  # 采样负样本
            if random.random() <= ratio / original_r:  # 只保留一部分
                data_new.append(temp_json)
        else:
            data_new.append(temp_json)
    if mode == 'train' and len(neg_train) > 0:  # 新增负样本
        for item in neg_train:
            h, r, t = item
            temp_json = {}
            temp_json['prompt'] = get_prompt(h, r, t)
            temp_json['completion'] = 'False'
            data_new.append(temp_json)
    for item in data_new:
        output = item['completion']
        if output == 'True':
            train_ture_num += 1
        elif output == 'False':
            train_false_num += 1
    return data_new, train_ture_num, train_false_num

train_json, train_ture_num, train_false_num = construct_data(p_train_data, mode='train')
valid_json, _, _ = construct_data(p_valid_data, mode='test')
test_json, _, _ = construct_data(p_test_data, mode='test')

print('最终样本数量：')
print(train_false_num, train_ture_num, train_false_num/train_ture_num)
print(len(train_json))
print(len(valid_json))
print(len(test_json))
random.shuffle(train_json)

with open(out_dir+'train.json', 'w') as f:
    json.dump(train_json, f, indent=4)
with open(out_dir+'valid.json', 'w') as f:
    json.dump(valid_json, f, indent=4)
with open(out_dir+'test.json', 'w') as f:
    json.dump(test_json, f, indent=4)

print(f"RQ-VAE数据生成完成，保存到: {out_dir}")
