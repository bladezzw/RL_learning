import time
import numpy as np
import torch as t

from torch.utils.data import Dataset

from collections import deque
import random
class ReplayBuffer(object):  # good
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)  # state[np.newaxis,:]
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))  #zip(*iterable) means 解压
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

def ModelParametersCopier(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

class BasicModule(t.nn.Module):
    """用pytorch的网络都继承它"""

    def save(self, modelpath=None):
        """用单个gpu算的时候可以使用此函数,多gpu用'torch.save(*.state_dict(),path)'"""
        if modelpath is None:
            modelpath = 'checkpoint' + time.strftime('%Y%m%d%H%M')
        t.save(self.state_dict(), modelpath)

    def load(self, modelpath=''):
        # 载入之前训练好的模型参数

        pretrained_dict = t.load(modelpath)
        net_state_dict = self.state_dict()
        net_state_dict.update(pretrained_dict)
        self.load_state_dict(net_state_dict)

    def zero_param(self):
        """清零模型参数"""
        for m in self.modules():
            if isinstance(m, t.nn.Conv2d):
                t.nn.init.constant_(m.weight.data, 0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, t.nn.Linear):
                t.nn.init.constant_(m.weight.data, 0)
                m.bias.data.zero_()

    # 按某种初始化方法来初始化权值(例如,Xavier,kaiming,normal_,uniform_等)
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, t.nn.Conv2d):
                t.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, t.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, t.nn.Linear):
                t.torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class MyDataset_ML(Dataset):
    """针对机器学习提供矩阵形式的整体输入,可用机器学习训练也可用深度学习训练时间间隔为1"""

    def __init__(self, x, y):
        """
        :param x: np.ndarray
        :param y: np.ndarray
        """
        if type(y[0]) != 'numpy.int16':
            y = y.astype('int16')
        self.xs = x.transpose()
        if -1 in y:
            y[np.where(y == -1)] = 0  # 只分两类
        self.ys = y
        self.wights = np.zeros(len(self.ys))
        index_1 = np.where(y == 1)[0]
        index_0 = np.where(y == 0)[0]
        self.wights[index_0] = 1 / len(index_0)
        self.wights[index_1] = 1 / len(index_1)

    def __getitem__(self, index):
        x, label = self.xs[index, :], self.ys[index]
        if self.transform is not None:
            x = self.transform(x)  # 在这里做transform，转为tensor等等
            # print("self.transform is not None:", self.transform is not None)
        return x, label

    def __len__(self):
        return self.xs.shape[1]

    def release_data(self):
        return self.xs, self.ys


# class MyDataset_1line(Dataset):
#     def __init__(self, x, y, window=5, transform=None, target_transform = None):
#
#         xs = []
#         for i in range(len(x)-window):
#             # 最后一个的label是不知道的,故长度为'range(len(x)-window)'
#             x_ = x[i:i+window]
#             label_ = y[i + window - 1]
#             xs.append((x_, label_))
#
#         self.xs = xs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         x, label = self.xs[index]
#         if self.transform is not None:
#             x = self.transform(x)  # 在这里做transform，转为tensor等等
#
#             # print("self.transform is not None:", self.transform is not None)
#         return x, label
#
#     def __len__(self):
#         return len(self.xs)-1

# # 测试用
# import numpy as np
# x = np.eye(10)
# y = np.array(range(1, 11))

class MyDataset_featuremap_conv2d(Dataset):
    """
    1. 直接2维特征图.
    2. 提供正太标准化
    """

    def __init__(self, x, y, window=5, transform=None, target_transform=None):

        xs = []
        lx = x.shape[1]
        self.real_x = []
        for i in range(window - 1, lx):
            x_ = x[:, i - window + 1:i + 1]
            self.real_x.append(x_[3][-1])  # 3代表close所在的行
            x_ = x_[np.newaxis, :]
            label_ = y[i]
            xs.append((x_, label_))

        self.real_x = np.array(self.real_x)
        self.xs = xs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.sum_y = sum(y)  # 为了计算y_bar
        self.ssa = (np.array(y) - self.y_bar()).__pow__(2).sum()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, label = self.xs[index]
        x = t.FloatTensor(x)
        if self.transform is not None:
            # print("self.transform is not None:", self.transform is not None)  # 在这里做transform，转为tensor等等
            mean, std = self.transform
            x = normalize(x, mean, std)
        return x, label

    def y_bar(self):
        """计算均值"""
        return self.sum_y / self.__len__()

    def __len__(self):
        return len(self.xs)


class MyDataset_featuremap_conv2d_mtmlabel(Dataset):
    """
    1. 直接2维特征图.
    2. mtm剔除非信号点，只留下真假信号
    3,只分两类, 只判断买入时机的真假
    """

    def __init__(self, x, y, mtm, window=5, transform=None, target_transform=None):
        if type(y[0]) != 'numpy.int16':
            y = y.astype('int16')
        if -1 in y:
            y[np.where(y == -1)] = 0  # 只分两类

        index_mtm = np.where(mtm != 0)[0]  # 剔除掉mtm中的非信号点，只留下真假信号
        xs = []
        lx = x.shape[1]
        num_0 = 0
        num_1 = 0
        for i in range(len(index_mtm)):  #
            if index_mtm[i] >= window - 1:
                x_ = x[:, index_mtm[i] - window + 1:index_mtm[i] + 1]
                x_ = x_[np.newaxis, :]
                label_ = y[index_mtm[i]]
                xs.append((x_, label_))
                if label_ == 0: num_0 += 1
                if label_ == 1: num_1 += 1

        self.xs = xs

        self.wight0 = num_1 / self.__len__()
        self.wight1 = num_0 / self.__len__()

        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, label = self.xs[index]
        x = t.FloatTensor(x)
        if self.transform is not None:
            # print("self.transform is not None:", self.transform is not None)  # 在这里做transform，转为tensor等等
            mean, std = self.transform
            x = normalize(x, mean, std)
        return x, label

    def __len__(self):
        return len(self.xs)


class MyDataset_featuremap_conv2d_clustering(Dataset):
    """
    1. 直接2维特征图.
    2. mtm剔除非信号点，只留下真假信号
    3,只分两类, 只判断买入时机的真假
    """

    def __init__(self, x, mtm, close, window=5, transform=None, target_transform=None):
        #  close应该和mtm同维数

        xs = []
        next_sell_index = None
        ret = np.zeros_like(mtm)

        # # 方法一：适用于有序数据，（buy，sell严格相邻出现）
        #
        # buy_index = np.where(mtm == 1)[0]
        # sell_index= np.where(mtm == -1)[0]
        # if (sell_index - buy_index).min() > 0:
        #     for i in range(len(buy_index)):
        #         if buy_index[i] >= window:
        #             if sell_index[i] > buy_index[i]:
        #                 next_sell_index = sell_index[i]
        #                 ret[buy_index[i]] = close[next_sell_index] - close[buy_index[i]]
        #                 x_ = x[:, buy_index[i] - window + 1:buy_index[i] + 1][np.newaxis, :]
        #                 xs.append((x_, mtm[buy_index[i]], ret[buy_index[i]]))
        #             elif sell_index[i-1] < buy_index[i]:  # 不是后面的就是前面的
        #                 next_sell_index = sell_index[i-1]
        #                 ret[buy_index[i]] = close[next_sell_index] - close[buy_index[i]]
        #                 x_ = x[:, buy_index[i] - window + 1:buy_index[i] + 1][np.newaxis, :]
        #                 xs.append((x_, mtm[buy_index[i]], ret[buy_index[i]]))
        # else:
        # 方法二
        pre_buy_index = None
        next_sell = None
        for i in range(window - 1, len(mtm)):
            if mtm[i] == 1:
                pre_buy_index = i
                for j in range(i, len(mtm)):
                    if mtm[j] == -1:
                        next_sell = j
                        break
                x_ = x[:, pre_buy_index - window + 1:pre_buy_index + 1][np.newaxis, :]
                temp = close[next_sell] - close[pre_buy_index]
                if temp < 0:
                    y = 0
                else:
                    y = 1
                xs.append((x_, y, temp))

        self.xs = xs
        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y, ret = self.xs[index]
        x = t.FloatTensor(x)
        if self.transform is not None:
            # print("self.transform is not None:", self.transform is not None)  # 在这里做transform，转为tensor等等
            mean, std = self.transform
            x = normalize(x, mean, std)
        return x, y, ret

    def __len__(self):
        return len(self.xs)


class MyDataset_featuremap_conv2d_eq_cls(Dataset):
    """
    1.直接2维特征图
    2.考虑类别平衡"""

    def __init__(self, x, y, window=5, transform=None, target_transform=None):

        # 包含多空0的可用数据集
        y_final = np.sort(np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0][::13], np.where(y == 2)[0]], axis=0))

        xs = []
        lx = x.shape[1]
        for index in y_final:
            # 最后一个的label是不知道的,故长度为'range(len(x)-window)'
            if index >= window - 1:
                x_ = x[:, index - window + 1:index + 1]
                x_ = x_[np.newaxis, :]  # x_[:,np.newaxis, :]
                label_ = y[index]
                xs.append((x_, label_))

        self.xs = xs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, label = self.xs[index]
        if self.transform is not None:
            print("self.transform is not None:", self.transform is not None)  # 在这里做transform，转为tensor等等

        return x, label

    def __len__(self):
        return len(self.xs)


class MyDataset_featuremap_conv2d_n_channels(Dataset):
    """
    1,不考虑类别平衡(其实对n+1来说有很多类)的类似2d卷积核方式去卷积多通道时间序列(对n+1天的预测时用),
    2,提供对输入数据正太标准化的变换"""

    def __init__(self, x, y, window=5, transform=None, target_transform=None):

        xs = []
        lx = x.shape[1]
        for i in range(window - 1, lx):
            x_ = x[:, i - window + 1:i + 1]
            x_ = x_[:, np.newaxis, :]
            label_ = y[i]
            xs.append((x_, label_))

        self.xs = xs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.sum_y = sum(y)  # 为了计算y_bar
        self.ssa = (np.array(y) - self.y_bar()).__pow__(2).sum()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, label = self.xs[index]
        x = t.FloatTensor(x)
        if self.transform is not None:
            # print("self.transform is not None:", self.transform is not None)  # 在这里做transform，转为tensor等等
            mean, std = self.transform
            x = normalize(x, mean, std)
        return x, label

    def y_bar(self):
        """计算均值"""
        return self.sum_y / self.__len__()

    def __len__(self):
        return len(self.xs)


class MyDataset_featuremap_conv2d_eq_cls_conv1d(Dataset):
    """
    1,考虑类别平衡的类2d卷积核方式卷积多通道时间序列,
    2,提供对输入数据正太标准化的变换"""

    def __init__(self, x, y, window=5, transform=None, target_transform=None):
        # index_y_0 = np.where(y == 0)[0]  # 空信号
        # index_y_1_final = np.where(y == 1)[0][::13]  # 0
        # index_y_2 = np.where(y == 2)[0]  # 多信号
        # 包含多空0的可用数据集
        y_final = np.sort(np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0][::13], np.where(y == 2)[0]], axis=0))

        xs = []
        lx = x.shape[1]
        for index in y_final:
            # 最后一个的label是不知道的,故长度为'range(len(x)-window)'
            if index >= window - 1:
                x_ = x[:, index - window + 1:index + 1]
                x_ = x_[:, np.newaxis, :]
                label_ = y[index]
                xs.append((x_, label_))

        self.xs = xs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, label = self.xs[index]
        x = t.FloatTensor(x)
        if self.transform is not None:
            # print("self.transform is not None:", self.transform is not None)  # 在这里做transform，转为tensor等等
            mean, std = self.transform
            x = normalize(x, mean, std)
        return x, label

    def __len__(self):
        return len(self.xs)


class MyDataset_featuremap_conv1d_eq_cls(Dataset):
    """考虑类别平衡的1维情况"""

    def __init__(self, x, y, window=5, transform=None, target_transform=None):

        # index_y_0 = np.where(y == 0)[0]  # 空信号
        # index_y_1_final = np.where(y == 1)[0][::13]  # 0
        # index_y_2 = np.where(y == 2)[0]  # 多信号

        # 包含多空0的可用数据集
        y_final = np.sort(np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0][::13], np.where(y == 2)[0]], axis=0))

        xs = []
        lx = x.shape[1]
        for index in y_final:
            # 最后一个的label是不知道的,故长度为'range(len(x)-window)'
            if index >= window - 1:
                x_ = x[:, index - window + 1:index + 1]
                label_ = y[index]
                xs.append((x_, label_))

        self.xs = xs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, label = self.xs[index]
        if self.transform is not None:
            # print("self.transform is not None:", self.transform is not None)  # 在这里做transform，转为tensor等等
            x = self.transform(x)
        return x, label

    def __len__(self):
        return len(self.xs)


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """

    if not inplace:
        tensor = tensor.clone()

    mean = t.tensor(mean, dtype=t.float32)
    std = t.tensor(std, dtype=t.float32)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor
