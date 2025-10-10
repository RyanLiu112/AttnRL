import numpy as np
import torch
from datasets import Dataset
from verl.utils.dataset.rl_dataset import RLHFDataset


class DataFilter:
    def __init__(self, dataset: RLHFDataset, use_mc=False):
        self.use_mc = use_mc
        self.reset(dataset)

    def format_data_key(self, data_source: str, index: int) -> str:
        return f"{data_source}--{index}"

    def reset(self, dataset: RLHFDataset):
        self.accs = np.full((dataset.all_data_num, 1), -1.0, dtype=np.float32)
        self.cnts = np.zeros((dataset.all_data_num, 1), dtype=np.int32)
        self.accs_all = np.full((dataset.all_data_num, 1), -1.0, dtype=np.float32)
        self.cnts_all = np.zeros((dataset.all_data_num, 1), dtype=np.int32)
        if self.use_mc:
            self.accs_mc = np.full((dataset.all_data_num, 1), -1.0, dtype=np.float32)
            self.cnts_mc = np.zeros((dataset.all_data_num, 1), dtype=np.int32)

    def preprocess_epoch(self, epoch: int, dataset):
        if epoch == 0 or self.accs.shape[1] > epoch:
            return
        self.accs = np.concatenate([self.accs, np.full((dataset.all_data_num, 1), -1.0, dtype=np.float32)], axis=1)
        self.cnts = np.concatenate([self.cnts, np.zeros((dataset.all_data_num, 1), dtype=np.int32)], axis=1)
        self.accs_all = np.concatenate([self.accs_all, np.full((dataset.all_data_num, 1), -1.0, dtype=np.float32)], axis=1)
        self.cnts_all = np.concatenate([self.cnts_all, np.zeros((dataset.all_data_num, 1), dtype=np.int32)], axis=1)
        if self.use_mc:
            self.accs_mc = np.concatenate([self.accs_mc, np.full((dataset.all_data_num, 1), -1.0, dtype=np.float32)], axis=1)
            self.cnts_mc = np.concatenate([self.cnts_mc, np.zeros((dataset.all_data_num, 1), dtype=np.int32)], axis=1)

    def add_reward(self, epoch: int, key: str, avg_reward: float, cnt: int):
        self.accs[key, epoch] = avg_reward
        self.cnts[key, epoch] = cnt
        self.accs_all[key, epoch] = avg_reward
        self.cnts_all[key, epoch] = cnt

    def add_reward_mc(self, epoch: int, key: str, avg_reward: float, cnt: int):
        self.accs_mc[key, epoch] = avg_reward
        self.cnts_mc[key, epoch] = cnt
        all_cnt = self.cnts[key, epoch] + cnt
        all_avg_reward = (self.accs[key, epoch] * self.cnts[key, epoch] + avg_reward * cnt) / all_cnt
        self.accs_mc[key, epoch] = all_avg_reward
        self.cnts_mc[key, epoch] = all_cnt

    def add_reward_batch(self, epoch, unique_data_indices, avg_accs, unique_data_cnts):
        self.accs[unique_data_indices, epoch] = avg_accs
        self.cnts[unique_data_indices, epoch] = unique_data_cnts
        self.accs_all[unique_data_indices, epoch] = avg_accs
        self.cnts_all[unique_data_indices, epoch] = unique_data_cnts

    def add_reward_batch_mc(self, epoch, unique_data_indices, avg_accs, unique_data_cnts):
        self.accs_mc[unique_data_indices, epoch] = avg_accs
        self.cnts_mc[unique_data_indices, epoch] = unique_data_cnts
        all_cnts = self.cnts[unique_data_indices, epoch] + unique_data_cnts
        all_avg_accs = (self.accs[unique_data_indices, epoch] * self.cnts[unique_data_indices, epoch] + avg_accs * unique_data_cnts) / all_cnts
        self.accs_all[unique_data_indices, epoch] = all_avg_accs
        self.cnts_all[unique_data_indices, epoch] = all_cnts

    def get_reward_trace(self, key: str):
        return self.data.get(key, [])

    def get_reward_trace_mc(self, key: str):
        return self.data_mc.get(key, [])

    def get_reward_trace_all(self, key: str):
        return self.data_all.get(key, [])

    def get_all_data_keys(self):
        return list(self.data.keys())

    def save(self, path: str):
        save_dict = {"accs": self.accs, "cnts": self.cnts, "accs_all": self.accs_all, "cnts_all": self.cnts_all}
        if self.use_mc:
            save_dict.update({"accs_mc": self.accs_mc, "cnts_mc": self.cnts_mc})
        torch.save(save_dict, path)

    def load(self, path: str):
        obj = torch.load(path, weights_only=False)
        self.accs, self.cnts = obj["accs"], obj["cnts"]
        self.accs_all, self.cnts_all = obj["accs_all"], obj["cnts_all"]
        if self.use_mc:
            self.accs_mc, self.cnts_mc = obj["accs_mc"], obj["cnts_mc"]

    def filter_all_dataset(self, epoch, dataset, mode="grpo"):
        if epoch == 0:
            print("Skipping dataset filtering in epoch 0")
            return dataset, {}

        data_num = len(dataset.dataframe_ori)
        assert hasattr(dataset, "dataframe_ori"), "Dataset must have a dataframe_ori attribute"
        dataset.dataframe = dataset.dataframe_ori.filter(
            lambda item: self.filter_easy_item(epoch, item, mode), num_proc=dataset.num_workers,
            desc=f"Filtering easy examples (mode: {mode})",
        )
        data_num_after_filter1 = len(dataset)

        dataset.dataframe = dataset.dataframe.filter(
            lambda item: self.filter_hard_item(epoch, item, mode), num_proc=dataset.num_workers,
            desc=f"Filtering hard examples (mode: {mode})",
        )
        data_num_after_filter2 = len(dataset)

        easy_filtered = data_num - data_num_after_filter1
        hard_filtered = data_num_after_filter1 - data_num_after_filter2
        print(f"Epoch {epoch}: {data_num} -> {data_num_after_filter1} -> {data_num_after_filter2}, "
              f"filtering {easy_filtered} easy examples, {hard_filtered} hard examples")

        info = {"batch/original": data_num, "batch/easy_filtered": easy_filtered, "batch/hard_filtered": hard_filtered}
        return dataset, info

    def filter_easy_item(self, epoch, data_item, mode="grpo", epsilon=0.1):
        key = data_item["extra_info"]["index"]
        assert self.accs.shape[1] == epoch
        if mode == "grpo":
            idxs = np.where(self.cnts[key, :epoch] > 0)[0]
            if idxs.size and self.accs[key, idxs].mean() == 1.0:
                return False
        elif mode == "all" or mode == "all_correct":
            idxs = np.where(self.cnts_all[key, :epoch] > 0)[0]
            if idxs.size and self.accs_all[key, idxs].mean() == 1.0:
                return False
        return True

    def filter_hard_item(self, epoch, data_item, mode="grpo", epsilon=0.1):
        key = data_item["extra_info"]["index"]
        assert self.accs.shape[1] == epoch
        if mode == "grpo":
            idxs = np.where(self.cnts[key, :epoch] > 0)[0]
            if idxs.size and self.accs[key, idxs].mean() == 0.0:
                return False
        elif mode == "all":
            idxs = np.where(self.cnts_all[key, :epoch] > 0)[0]
            if idxs.size and self.accs_all[key, idxs].mean() == 0.0:
                return False
        return True

    def __len__(self):
        return len(self.accs)
