from typing import List
from typing import Optional

from pybloom_live import BloomFilter
from numpy import ndarray, math
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from params.init_nets_params import BoostType


class TrainValSet:
    def __init__(
        self,
        x_train: ndarray,
        y_train: ndarray,
        x_val: Optional[ndarray],
        y_val: Optional[ndarray],
        train_val_bloom: "TrainValBloom" = None,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.strat_indices = []
        self.chosen_feats = []
        self.nr_classes = np.unique(y_train).size
        self.train_val_bloom = train_val_bloom

    # part_nr = say perc_of_data is 0.25, part_nr 3 will result in data between 50 % and 75 %
    def get_part(self, perc_of_data: float, part_nr: int):
        nr_samples = self.x_train.shape[0]
        part_size = math.ceil(nr_samples * perc_of_data)
        total_nr_parts = int(math.ceil(nr_samples / part_size))

        if perc_of_data == 1.0 or nr_samples == part_size:
            return TrainValSet(self.x_train, self.y_train, self.x_val, self.y_val)
        elif len(self.strat_indices) == 0:
            skf = StratifiedKFold(
                n_splits=total_nr_parts, shuffle=True, random_state=42
            )
            split = list(skf.split(self.x_train, self.y_train))
            for _, fold in split:
                self.strat_indices.append(fold)

        return TrainValSet(
            self.x_train[self.strat_indices[part_nr]],
            self.y_train[self.strat_indices[part_nr]],
            self.x_val,
            self.y_val,
        )

    def to_train_set(self):
        return TrainSet(self.x_train, self.y_train)

    def size(self):
        return self.x_train.shape[0]

    def subset_feats(self, chosen_feats: List[int]):
        self.chosen_feats = chosen_feats
        self.x_train = self.x_train[:, chosen_feats]
        self.x_val = self.x_val[:, chosen_feats]

    def rand_subset_data(self, subset_size: int):
        selected_indices = []

        indices_to_sample_from = np.asarray(list(range(0, self.y_train.shape[0])))

        selected_indices.extend(
            np.random.choice(indices_to_sample_from, size=subset_size, replace=False)
        )

        ret_train_val_set = TrainValSet(
            x_train=self.x_train[selected_indices, :],
            y_train=self.y_train[selected_indices],
            x_val=self.x_val,
            y_val=self.y_val,
        )

        return ret_train_val_set

    def merge(self, other_train_val_set: "TrainValSet"):
        return TrainValSet(
            x_train=np.vstack((self.x_train, other_train_val_set.x_train)),
            y_train=np.append(self.y_train, other_train_val_set.y_train),
            x_val=self.x_val,
            y_val=self.y_val,
        )

    # toss out old data and replace with wrong preds
    def merge_toss(
        self,
        wrong_pred_inds: List[int],
        orig_train_set: "TrainSet",
        next_boost_net_feats: List[int],
        boost_type: BoostType,
    ) -> "TrainValSet":
        # remove rows from validation set that will be in new train set
        val_inds_to_keep = []

        for i in range(0, orig_train_set.size()):
            if i in self.train_val_bloom.val_bloom and i not in wrong_pred_inds:
                val_inds_to_keep.append(i)

        new_x_val = orig_train_set.x_train[
            np.ix_(val_inds_to_keep, next_boost_net_feats)
        ]
        new_y_val = orig_train_set.y_train[val_inds_to_keep]

        old_train_inds, old_val_inds = self.train_val_bloom.into_indices()
        no_dup_train_inds = list(set(old_train_inds) - set(wrong_pred_inds))

        if boost_type == BoostType.fifty_fifty:
            subset_train_inds = np.random.choice(
                no_dup_train_inds, size=len(wrong_pred_inds), replace=False
            )
            new_train_inds = list(set(wrong_pred_inds + subset_train_inds))
        elif boost_type == BoostType.wrong_only:
            new_train_inds = list(set(wrong_pred_inds))
        else:
            new_train_inds = list(set(wrong_pred_inds + no_dup_train_inds))[
                : self.size()
            ]

        return TrainValSet(
            x_train=orig_train_set.x_train[
                np.ix_(new_train_inds, next_boost_net_feats)
            ],
            y_train=orig_train_set.y_train[new_train_inds],
            x_val=new_x_val,
            y_val=new_y_val,
            train_val_bloom=TrainValBloom(
                new_train_inds,
                next_boost_net_feats,
                orig_train_set.size(),
                val_inds_to_keep,
            ),
        )


class TrainValBloom:
    def __init__(
        self,
        train_row_nrs: List[int],
        chosen_feats: List[int],
        full_dataset_size: int,
        val_row_nrs: List[int] = None,
    ):
        self.train_bloom = BloomFilter(capacity=len(train_row_nrs), error_rate=0.01)
        self.full_dataset_size = full_dataset_size
        self.subset_size = len(train_row_nrs)
        self.chosen_feats = chosen_feats
        all_row_nrs = list(range(0, full_dataset_size))

        if val_row_nrs is None:
            val_row_nrs = [x for x in all_row_nrs if x not in train_row_nrs]

        self.val_bloom = BloomFilter(capacity=len(val_row_nrs), error_rate=0.01)

        for row_nr in train_row_nrs:
            self.train_bloom.add(row_nr)

        for row_nr in val_row_nrs:
            self.val_bloom.add(row_nr)

    def into_dataset(self, train_set: "TrainSet", save_bloom: bool) -> "TrainValSet":
        train_inds, val_inds = self.into_indices()

        if save_bloom:
            train_val_bloom = self
        else:
            train_val_bloom = None

        return TrainValSet(
            train_set.x_train[np.ix_(train_inds, self.chosen_feats)],
            train_set.y_train[train_inds],
            train_set.x_train[np.ix_(val_inds, self.chosen_feats)],
            train_set.y_train[val_inds],
            train_val_bloom,
        )

    def into_indices(self) -> (List[int], List[int]):
        train_inds = np.zeros(self.subset_size)
        val_inds = np.zeros(self.full_dataset_size - self.subset_size)

        train_cntr = 0
        val_cntr = 0
        for i in range(0, self.full_dataset_size):
            if i in self.train_bloom and train_cntr < self.subset_size:
                train_inds[train_cntr] = i
                train_cntr += 1
            else:
                val_inds[val_cntr] = i
                val_cntr += 1

        val_inds = np.trim_zeros(val_inds, "b")  # 'b' = trim only from back
        train_inds_lst: List[int] = train_inds.astype(int).tolist()
        val_inds_lst: List[int] = val_inds.astype(int).tolist()

        return train_inds_lst, val_inds_lst


class TrainSet:
    def __init__(self, x_train: ndarray, y_train: ndarray):
        self.x_train = x_train
        self.y_train = y_train
        self.nr_classes = np.unique(y_train).size

    def size(self):
        return self.x_train.shape[0]

    def split_into_train_val(self, val_size: float):
        x_train, x_val, y_train, y_val = train_test_split(
            self.x_train, self.y_train, stratify=self.y_train, random_state=42, test_size=val_size
        )

        return TrainValSet(x_train, y_train, x_val, y_val)


class ValSet:
    def __init__(self, x_val: ndarray, y_val: ndarray):
        self.x_val = x_val
        self.y_val = y_val
