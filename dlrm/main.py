import functools

import numpy as np
import pandas as pd
import ray.util.data as ml_data
import raydp
import torch
from torch.utils.data.dataset import IterableDataset

from .dlrm_s_pytorch import ModelArguments
from .preproc.spark_preproc import pre_process, PreProcArguments


def collate_fn(
        df: pd.DataFrame,
        num_numerical_features: int,
        max_ind_range: int,
        flag_input_torch_tensor=False):
    array = df.values
    x_int_batch = array[:, 1:1 + num_numerical_features].view(dtype=np.float32)
    x_cat_batch = array[:, 1 + num_numerical_features:]
    y_batch = array[:, 0]

    if max_ind_range > 0:
        x_cat_batch = x_cat_batch % max_ind_range

    if flag_input_torch_tensor:
        x_int_batch = torch.log(x_int_batch.clone().detach().type(torch.float) + 1)
        x_cat_batch = x_cat_batch.clone().detach().type(torch.long)
        y_batch = y_batch.clone().detach().type(torch.float32).view(-1, 1)
    else:
        x_int_batch = torch.log(torch.tensor(x_int_batch, dtype=torch.float) + 1)
        x_cat_batch = torch.tensor(x_cat_batch, dtype=torch.long)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

    batch_size = x_cat_batch.shape[0]
    feature_count = x_cat_batch.shape[1]
    lS_o = torch.arange(batch_size).reshape(1, -1).repeat(feature_count, 1)

    return x_int_batch, lS_o, x_cat_batch.t(), y_batch.view(-1, 1)


def create_torch_ds(ds: ml_data.MLDataset,
                    num_shards: int,
                    shard_index: int,
                    batch_size: int,
                    shuffle: bool):
    assert shard_index < num_shards
    shard_ids = []
    i = shard_index
    step = num_shards
    while i < ds.num_shards():
        shard_ids.append(i)
        i += step
    ds = ds.select_shards(shard_ids)
    return TorchDataset(ds, batch_size, shuffle)


class TorchDataset(IterableDataset):
    def __init__(self,
                 ds: ml_data.MLDataset,
                 batch_size: int,
                 shuffle: bool):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        it = self.ds.gather_async(batch_ms=0, num_async=self.ds.num_shards())
        it = iter(it)
        return_df = None
        while True:
            try:
                cur_df = next(it)
                cur_index = 0
                cur_size = cur_df.shape[0]
                while cur_df is not None or (
                        cur_index + self.batch_size) < cur_size:
                    if cur_df is None or cur_index == cur_size:
                        cur_df = next(it)
                        cur_index = 0
                        cur_size = cur_df.shape[0]
                    if return_df is not None:
                        ri = cur_index + self.batch_size - return_df.shape[0]
                        ri = min(ri, cur_size)
                        tmp = cur_df.iloc[cur_index:ri]
                        return_df = pd.concat([return_df, tmp])
                        cur_index = ri
                    else:
                        ri = cur_index + self.batch_size
                        ri = min(ri, cur_size)
                        return_df = cur_df.iloc[cur_index:ri]
                        cur_index = ri
                    if return_df.shape[0] == self.batch_size:
                        if self.shuffle:
                            return_df = return_df.sample(frac=1)
                        yield return_df
                        return_df = None
            except StopIteration:
                break

        if return_df is not None:
            if self.shuffle:
                return_df = return_df.sample(frac=1)
            yield return_df


def data_preprocess(
        app_name: str,
        num_executors: int = None,
        executor_cores: int = None,
        executor_memory: str = None,
        configs=None,
        args: PreProcArguments = None):
    try:
        spark = raydp.init_spark(
            app_name=app_name,
            num_executors=num_executors,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            configs=configs)

        train_folder, test_folder, model_size = pre_process(spark, args)
        return train_folder, test_folder, model_size
    except:
        raise
    finally:
        raydp.stop_spark()


def create_train_task(
        num_workers: int = None,
        train_folder: str = None,
        test_folder: str = None,
        model_size=None,
        args: ModelArguments = None,
        nbatches: int = -1,
        nbatches_test: int = -1):
    # setup dataset
    train_ds = ml_data.read_parquet(train_folder, num_workers)
    if test_folder:
        test_ds = ml_data.read_parquet(test_folder, num_workers)
    else:
        test_ds = None

    fn = functools.partial(collate_fn,
                           num_numerical_features=args.arch_dense_feature_size,
                           max_ind_range=args.max_ind_range,
                           flag_input_torch_tensor=False)

    def task_fn(context):

        train_dataset = create_torch_ds(train_ds, num_workers, context.world_rank,
            args.mini_batch_size, args.mlperf_bin_shuffle)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=None,
            batch_sampler=None,
            shuffle=False,
            num_workers=0,
            collate_fn=fn,
            pin_memory=False,
            drop_last=False,
            sampler=None
        )

        test_loader = None
        if test_ds is not None:
            test_dataset = create_torch_ds(test_ds, num_workers, context.world_rank,
                args.mini_batch_size, False)
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=None,
                batch_sampler=None,
                shuffle=False,
                num_workers=0,
                collate_fn=fn,
                pin_memory=False,
                drop_last=False,
                sampler=None
            )
        from dlrm.dlrm_s_pytorch import model_training
        model_training(args, train_loader, test_loader, model_size,
            nbatches, nbatches_test)
        return context.world_rank
    return task_fn
