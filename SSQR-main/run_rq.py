import argparse
import time
from pprint import pprint
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import dgl
import wandb
from utils.process_data import load_data
from torch.nn.utils import clip_grad_norm_

from model import CompGCN_DistMult, CompGCN_ConvE, VQGCN, VQGCN_MLP
from model.RQGCN import RQGCN  # 导入新的RQ-VAE模型
from utils import process, TrainDataset, TestDataset


class Runner(object):
    def __init__(self, params):
        self.p = params

        # wandb.init(project="VQKG", name='VQTest_notransformer',config=params)
        wandb.init(project="RQKG_1015", name='RQ4KG',config=params)  # 更新项目名称

        self.prj_path = Path(__file__).parent.resolve()  # 目录
        self.data, self.entity_dict, self.relation_dict, self.entity_list, self.relation_list = \
            load_data(self.p.dataset)
        self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, \
            self.data.train, self.data.valid, self.data.test, self.data.num_rels
        self.triplets = process({'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data},
                                self.num_rels)  # 训练集合
        self.device = torch.device(f'cuda:{self.p.gpu}')
        self.data_iter = self.get_data_iter()
        self.g = self.build_graph()
        self.edge_type, self.edge_norm = self.get_edge_dir_and_norm()

        self.g = self.g.to(self.device)

        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        # RQ-VAE需要为每个量化器创建独立的优化器
        self.optimizer_rq = torch.optim.Adam(self.model.rq_codebook.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.best_val_mrr, self.best_epoch, self.best_val_results = 0., 0., {}

        # ent_emd 3072 40943
        self.ent_text_emd = np.load('../MyKGVQ/data/FB15k-237/ent_emd.npy')
        self.ent_text_emd = torch.tensor(self.ent_text_emd, requires_grad=False).float().to(self.device)
        print(self.ent_text_emd.size(), self.ent_text_emd.dtype)
        pprint(vars(self.p))

    def fit(self):
        save_root = self.prj_path / 'checkpoints'
        if not save_root.exists():
            save_root.mkdir()
        save_path = save_root / (self.p.name + '.pt')

        if self.p.restore:
            self.load_model(save_path)
            print('Successfully Loaded previous model')

        # 训练循环
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss, pre_loss, x_loss, q_loss = self.train()
            val_results = self.evaluate('valid')
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val_results = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
            print(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
            wandb.log({"train_loss": train_loss, "pre_loss": pre_loss, "x_loss": x_loss, "q_loss": q_loss, "MRR": val_results['mrr'], "MR": val_results['mr'],
                       "hits@1": val_results['hits@1'], "hits@3": val_results['hits@3'], "hits@10": val_results['hits@10']})

        # 加载最佳模型并测试
        self.load_model(save_path)
        print(f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        test_results = self.evaluate('test')
        print(
            f"MRR: Tail {test_results['left_mrr']:.5}, Head {test_results['right_mrr']:.5}, Avg {test_results['mrr']:.5}")
        print(f"MR: Tail {test_results['left_mr']:.5}, Head {test_results['right_mr']:.5}, Avg {test_results['mr']:.5}")
        print(f"hits@1 = {test_results['hits@1']:.5}")
        print(f"hits@3 = {test_results['hits@3']:.5}")
        print(f"hits@10 = {test_results['hits@10']:.5}")

        # 生成entity code并保存
        self.gen_ent_codes()
        self.gen_ent_codes_emds()

    def gen_ent_codes(self):
        """生成并保存所有实体的RQ代码"""
        self.model.eval()
        with torch.no_grad():
            codes = self.model.cal_allent_codes(self.g)
        codes = codes.cpu()
        print(f"Generated RQ codes shape: {codes.shape}")
        dataset = self.p.dataset.split('/')[-1]
        save_name = f'./codes_new/{dataset}_{self.p.seq_len}_{self.p.num_code}_{self.p.num_quantizers}_rq.pt'
        torch.save(codes, save_name)

    def gen_ent_codes_emds(self):
        """生成并保存所有实体的RQ嵌入表示"""
        self.model.eval()
        with torch.no_grad():
            codes_emds = self.model.cal_allent_codes_emds(self.g)
        codes_emds = codes_emds.cpu()
        dataset = self.p.dataset.split('/')[-1]
        save_name = f'./codes_new/{dataset}_{self.p.seq_len}_{self.p.num_code}_{self.p.num_quantizers}_rq_emd.pt'
        torch.save(codes_emds, save_name)

    def train(self):  # 所有整体训练
        self.model.train()
        losses = []
        pre_losses = []
        x_losses = []
        q_losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            triplets, labels = triplets.to(self.device), labels.to(self.device)
            subj, rel = triplets[:, 0], triplets[:, 1]
            pred, q_loss, x_loss = self.model(self.g, subj, rel, self.ent_text_emd, stage=3)
            pre_loss = self.model.calc_loss(pred, labels)
            loss = pre_loss + q_loss

            self.optimizer.zero_grad()
            self.optimizer_rq.zero_grad()
            loss.backward()
            
            clip_grad_norm_(self.model.parameters(), max_norm=1)
            clip_grad_norm_(self.model.rq_codebook.parameters(), max_norm=1)
            self.optimizer.step()
            self.optimizer_rq.step()
            
            losses.append(loss.item())
            pre_losses.append(pre_loss.item())
            x_losses.append(x_loss.item())
            q_losses.append(q_loss.item())

        loss = np.mean(losses)
        pre_loss = np.mean(pre_losses)
        x_loss = np.mean(x_losses)
        q_loss = np.mean(q_losses)
        return loss, pre_loss, x_loss, q_loss

    def evaluate(self, split):
        """评估模型性能"""
        def get_combined_results(left, right):
            results = dict()
            assert left['count'] == right['count']
            count = float(left['count'])
            results['left_mr'] = round(left['mr'] / count, 5)
            results['left_mrr'] = round(left['mrr'] / count, 5)
            results['right_mr'] = round(right['mr'] / count, 5)
            results['right_mrr'] = round(right['mrr'] / count, 5)
            results['mr'] = round((left['mr'] + right['mr']) / (2 * count), 5)
            results['mrr'] = round((left['mrr'] + right['mrr']) / (2 * count), 5)
            for k in [1, 3, 10]:
                results[f'left_hits@{k}'] = round(left[f'hits@{k}'] / count, 5)
                results[f'right_hits@{k}'] = round(right[f'hits@{k}'] / count, 5)
                results[f'hits@{k}'] = round((results[f'left_hits@{k}'] + results[f'right_hits@{k}']) / 2, 5)
            return results

        self.model.eval()
        left_result = self.predict(split, 'tail')
        right_result = self.predict(split, 'head')
        res = get_combined_results(left_result, right_result)
        return res

    def predict(self, split='valid', mode='tail'):
        """预测函数"""
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_{mode}']
            for step, (triplets, labels) in enumerate(test_iter):
                triplets, labels = triplets.to(self.device), labels.to(self.device)
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                pred, _, _ = self.model(self.g, subj, rel)
                b_range = torch.arange(pred.shape[0], device=self.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0)

                for k in [1, 3, 10]:
                    results[f'hits@{k}'] = torch.numel(ranks[ranks <= k]) + results.get(f'hits@{k}', 0)
        return results

    def save_model(self, path):
        """保存模型"""
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'optimizer_rq': self.optimizer_rq.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, path)

    def load_model(self, path):
        """加载模型"""
        state = torch.load(path)
        self.best_val_results = state['best_val']
        self.best_val_mrr = self.best_val_results['mrr']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if 'optimizer_rq' in state:
            self.optimizer_rq.load_state_dict(state['optimizer_rq'])

    def build_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_ent)
        g.add_edges(self.train_data[:, 0], self.train_data[:, 2])
        g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
        return g

    def get_data_iter(self):
        """获取数据加载器"""
        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers
            )

        return {
            'train': get_data_loader(TrainDataset, 'train'),
            'valid_head': get_data_loader(TestDataset, 'valid_head'),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail'),
            'test_head': get_data_loader(TestDataset, 'test_head'),
            'test_tail': get_data_loader(TestDataset, 'test_tail')
        }

    def get_edge_dir_and_norm(self):
        """获取边类型和归一化"""
        in_deg = self.g.in_degrees(range(self.g.number_of_nodes())).float().numpy()
        norm = in_deg ** -0.5
        norm[np.isinf(norm)] = 0
        self.g.ndata['xxx'] = torch.tensor(norm)
        self.g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        norm = self.g.edata.pop('xxx').squeeze().to(self.device)
        edge_type = torch.tensor(np.concatenate([self.train_data[:, 1], self.train_data[:, 1] + self.num_rels])).to(
            self.device)
        return edge_type, norm

    def get_model(self):
        """获取RQ-VAE模型"""
        model = RQGCN(num_ent=self.num_ent, num_rel=self.num_rels, gcn_layers=self.p.gcn_layers,
                      tf_layers=self.p.tf_layers, init_dim=self.p.init_dim,
                      gcn_dim=self.p.gcn_dim, edge_type=self.edge_type, edge_norm=self.edge_norm,
                      gcn_drop=self.p.gcn_drop, act=self.p.act, opn='mult', 
                      seq_len=self.p.seq_len, num_code=self.p.num_code, 
                      num_quantizers=self.p.num_quantizers, att_head=self.p.att_head)
        model.to(self.device)
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', default='test_run', help='Set run name for saving/restoring models')
    parser.add_argument('--data', dest='dataset', default='data/FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('--opn', dest='opn', default='mult', help='Composition Operation to be used in CompGCN')
    parser.add_argument('--act', dest='act', default='tanh', help='activation function')

    parser.add_argument('--batch', dest='batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default=5, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs', type=int, default=800, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=1e-8, help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.0005, help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=12345, type=int, help='Seed for randomization')

    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Restore from the previously saved model')

    parser.add_argument('--init_dim', dest='init_dim', default=200, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('--seq_len', dest='seq_len', default=32, type=int, help='Number of VQ sequence')
    parser.add_argument('--num_code', dest='num_code', default=2048, type=int, help='Number of VQ codebook')
    parser.add_argument('--num_quantizers', dest='num_quantizers', default=4, type=int, help='Number of RQ quantizers')
    parser.add_argument('--gcn_layers', dest='gcn_layers', default=2, type=int, help='Number of GCN Layers to use')
    parser.add_argument('--tf_layers', dest='tf_layers', default=1, type=int, help='Number of Transformer Layers to use')
    parser.add_argument('--att_head', dest='att_head', default=1, type=int, help='Number of attention head in Transformer')
    parser.add_argument('--gcn_drop', dest='gcn_drop', default=0.2, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--vq_weight', dest='vq_weight', default=1.0, type=float, help='VQ codebook weight in the loss')

    args = parser.parse_args()
    if not args.restore:
        dataset = args.dataset.split('/')[-1]
        save_name = f'{dataset}_{args.seq_len}_{args.num_code}_{args.num_quantizers}_rq' + time.strftime('%Y_%m_%d') + '_' + time.strftime(
            '%H:%M:%S')
        args.name = save_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    runner = Runner(args)
    runner.fit()
