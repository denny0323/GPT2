import torch
import numpy as np

class Metric:
    def __init__(self, name=None, decimal_point=4):
        if name is None:
            self.name = ''

        self.name = name
        self.decimal_point = decimal_point
        self.n = 0
        self.metric_store = np.array([])

    def update(self, metric_list:list):
        # 자료형 변환
        if all(isinstance(metric, torch.Tensor) for metric in metric_list):
            metric_list = [metric.detach().item() for metric in metric_list]

        self.metric_store = np.append(self.metric_store, metric_list, axis=0)
        self.n = self.metric_store.shape[0]


    def reset(self):
        self.n = 0
        self.metric_store = np.array([])


    def compute_average(self):
        return round(np.mean(self.metric_store), self.decimal_point)