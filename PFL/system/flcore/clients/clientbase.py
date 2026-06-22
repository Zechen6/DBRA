import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class CustomCifarDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)

        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.save_folder_name = args.save_folder_name

        # 只认 main.py 传下来的原始模型名
        if hasattr(args, "model_tag") and args.model_tag is not None:
            self.model_name = str(args.model_tag)
        else:
            self.model_name = self.model.__class__.__name__

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        self.best_test_acc = -1.0
        self.best_test_auc = -1.0
        self.best_round = -1
        self.best_model_path = None

        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        if 'ifar' in self.dataset.lower():
            # CIFAR-10/100 需要数据增强
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
            train_dataset = CustomCifarDataset(train_data, transform=transform)
            train_data = train_dataset
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        #self.model.eval()
        model_to_test = self.model_per if hasattr(self, "model_per") else self.model
        model_to_test.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model_to_test(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        """
        不再保存 model.pt，模型统一由 save_best_client_model() 保存为 best_model.pt
        """
        if item_name == "model":
            return

        save_dir = self._get_client_save_dir(item_path)
        save_path = os.path.join(save_dir, f"{item_name}.pt")
        torch.save(item, save_path)

    def load_item(self, item_name, item_path=None, map_location=None):
        """
        如果读取 model，就去读 best_model.pt
        """
        save_dir = self._get_client_save_dir(item_path)

        if item_name == "model":
            load_path = os.path.join(save_dir, "best_model.pt")
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Best model not found for client {self.id}: {load_path}")

            if map_location is None:
                map_location = self.device

            checkpoint = torch.load(load_path, map_location=map_location)
            return checkpoint

        load_path = os.path.join(save_dir, f"{item_name}.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Item not found for client {self.id}: {load_path}")

        if map_location is None:
            map_location = self.device

        return torch.load(load_path, map_location=map_location)


    def _get_model_tag(self):
        """
        返回用于保存路径/文件名的模型标识，确保 CNN / ResNet18 等能区分开。
        """
        model_tag = str(getattr(self, "model_name", self.model.__class__.__name__))

        # 去掉路径和危险字符，避免保存路径异常
        model_tag = os.path.basename(model_tag)
        model_tag = model_tag.replace("\\", "_").replace("/", "_")
        model_tag = model_tag.replace(" ", "_").replace(":", "_")

        return model_tag

    def _get_client_save_dir(self, save_root=None):
        """
        为每个 算法 / 数据集 / 模型 / 客户端 单独建目录
        """
        if save_root is None:
            save_root = self.save_folder_name

        model_tag = self._get_model_tag()

        client_dir = os.path.join(
            save_root,
            str(self.algorithm),
            str(self.dataset),
            model_tag,
            f"client_{self.id}"
        )
        os.makedirs(client_dir, exist_ok=True)
        return client_dir

    def get_test_acc_auc(self):
        """
        返回当前客户端模型在测试集上的 acc / auc
        """
        test_correct, test_total, auc = self.test_metrics()
        acc = float(test_correct) / max(float(test_total), 1.0)
        return acc, auc, test_correct, test_total
    '''
    def save_best_client_model(self, round_idx=None, save_root=None, verbose=True):
        """
        只保存该客户端历史上最好的模型。
        优先比较 acc；acc 相同时比较 auc。
        """
        acc, auc, test_correct, test_total = self.get_test_acc_auc()

        is_better = False
        if acc > self.best_test_acc:
            is_better = True
        elif abs(acc - self.best_test_acc) < 1e-12 and auc > self.best_test_auc:
            is_better = True

        save_dir = self._get_client_save_dir(save_root)
        save_path = os.path.join(save_dir, "best_model.pt")

        if is_better:
            checkpoint = {
                "client_id": self.id,
                "algorithm": self.algorithm,
                "dataset": self.dataset,
                "round": round_idx,
                "best_test_acc": acc,
                "best_test_auc": auc,
                "test_correct": test_correct,
                "test_total": test_total,
                "model_state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            }

            torch.save(checkpoint, save_path)

            self.best_test_acc = acc
            self.best_test_auc = auc
            self.best_round = -1 if round_idx is None else round_idx
            self.best_model_path = save_path

            if verbose:
                print(
                    f"[Client {self.id}] best model saved | "
                    f"acc={acc:.4f}, auc={auc:.4f}, round={round_idx}, path={save_path}"
                )

            return True

        if verbose:
            print(
                f"[Client {self.id}] no update | "
                f"current_acc={acc:.4f}, best_acc={self.best_test_acc:.4f}, "
                f"current_auc={auc:.4f}, best_auc={self.best_test_auc:.4f}"
            )

        return False
    '''
    def save_best_client_model(self, round_idx=None, save_root=None, verbose=True):
        """
        只保存该客户端历史上最好的模型：best_model.pt
        优先比较 acc；acc 相同时比较 auc。
        Ditto 会优先保存 model_per；其他算法保存 model。
        """
        acc, auc, test_correct, test_total = self.get_test_acc_auc()

        is_better = False
        if acc > self.best_test_acc:
            is_better = True
        elif abs(acc - self.best_test_acc) < 1e-12 and auc > self.best_test_auc:
            is_better = True

        save_dir = self._get_client_save_dir(save_root)
        best_save_path = os.path.join(save_dir, "best_model.pt")

        if is_better:
            model_to_save = self.model_per if hasattr(self, "model_per") else self.model

            best_checkpoint = {
                "client_id": self.id,
                "algorithm": self.algorithm,
                "dataset": self.dataset,
                "round": round_idx,
                "best_test_acc": acc,
                "best_test_auc": auc,
                "test_correct": test_correct,
                "test_total": test_total,
                "saved_model_type": "model_per" if hasattr(self, "model_per") else "model",
                "model_state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in model_to_save.state_dict().items()
                }
            }

            torch.save(best_checkpoint, best_save_path)

            self.best_test_acc = acc
            self.best_test_auc = auc
            self.best_round = -1 if round_idx is None else round_idx
            self.best_model_path = best_save_path

            if verbose:
                print(
                    f"[Client {self.id}] best model saved | "
                    f"acc={acc:.4f}, auc={auc:.4f}, round={round_idx}, "
                    f"type={best_checkpoint['saved_model_type']}, "
                    f"path={best_save_path}"
                )

        return is_better
        

    def load_latest_client_model(self, save_root=None, map_location=None, verbose=True):
        """
        读取该客户端 latest_model.pt，并恢复到 self.model
        适用于 FedBN / FedALA / FedAvg 等不依赖额外控制变量的“近似续训”
        """
        load_path = os.path.join(self._get_client_save_dir(save_root), "latest_model.pt")

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Latest model not found for client {self.id}: {load_path}")

        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(load_path, map_location=map_location)

        if "model_state_dict" not in checkpoint:
            raise KeyError(f"'model_state_dict' not found in latest checkpoint: {load_path}")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        # 尽量把历史指标也恢复一下
        self.best_test_acc = checkpoint.get("best_test_acc", self.best_test_acc)
        self.best_test_auc = checkpoint.get("best_test_auc", self.best_test_auc)
        self.best_round = checkpoint.get("round", self.best_round)

        if verbose:
            print(
                f"[Client {self.id}] loaded latest model | "
                f"round={checkpoint.get('round', None)}, path={load_path}"
            )

        return checkpoint


    def load_best_client_model(self, save_root=None, map_location=None, verbose=True):
        load_path = os.path.join(self._get_client_save_dir(save_root), "best_model.pt")

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"未找到客户端 {self.id} 的最优模型: {load_path}")

        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(load_path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.best_test_acc = checkpoint.get("best_test_acc", -1.0)
        self.best_test_auc = checkpoint.get("best_test_auc", -1.0)
        self.best_round = checkpoint.get("round", -1)
        self.best_model_path = load_path

        if verbose:
            print(
                f"[Client {self.id}] 已加载最优模型 | "
                f"acc={self.best_test_acc:.4f}, auc={self.best_test_auc:.4f}, "
                f"round={self.best_round}, path={load_path}"
            )

        return checkpoint
    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
