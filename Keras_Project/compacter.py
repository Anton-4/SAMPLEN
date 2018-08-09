
import numpy as np
from numpy import ndarray
from sklearn import metrics
import itertools


def most_occuring(arr: ndarray):
    (values, counts) = np.unique(arr, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]


def perform_vote(votes_arr: ndarray):
    if len(votes_arr.shape) > 1 and votes_arr.shape[1] > 1:
        final_votes = np.apply_along_axis(most_occuring, axis=1, arr=votes_arr)
        return final_votes
    elif len(votes_arr.shape) > 1 and votes_arr.shape[1] == 1:
        return votes_arr
    else:
        votes_list = [np.asarray(votes) for votes in votes_arr]
        votes_arg = np.array(votes_list)
        print(type(votes_arg))
        print(type(votes_arg[0]))
        print(votes_arg)
        print(votes_arg.shape)
        final_votes = np.apply_along_axis(most_occuring, axis=1, arr=votes_arg)
        return final_votes


def trim_bottom_up(
    votes_arr: ndarray, y_val: ndarray, best_net_ind: int, best_net_acc: float
):
    best_net_votes = votes_arr[:, best_net_ind]
    nr_nets = votes_arr.shape[1]
    all_inds = list(range(0, nr_nets))
    all_inds.remove(best_net_ind)

    sets_of_two = list(itertools.combinations(all_inds, 2))
    best_acc = best_net_acc
    best_set = []

    for ind_set in sets_of_two:
        subset_votes = np.vstack(
            (votes_arr[:, ind_set[0]], votes_arr[:, ind_set[1]], best_net_votes)
        )
        subset_combined_votes = perform_vote(subset_votes.transpose())

        set_acc = metrics.accuracy_score(
            subset_combined_votes, y_val
        )

        if set_acc > best_acc:
            best_acc = set_acc
            best_set = list(ind_set)

    best_set.append(best_net_ind)

    for ind in best_set:
        if ind in all_inds:
            all_inds.remove(ind)

    for rem_ind in all_inds:
        test_subset = best_set + [rem_ind]
        subset_votes = np.vstack(([votes_arr[:, ind] for ind in test_subset]))
        set_acc = metrics.accuracy_score(
            perform_vote(subset_votes.transpose()), y_val
        )

        if set_acc > best_acc:
            best_acc = set_acc
            best_set = test_subset

    # print("best individual acc: " + str(best_net_acc))
    # print("best bottom up: " + str(best_acc))

    all_inds_full = list(range(0, nr_nets))
    drop_nets_inds = [ind for ind in all_inds_full if ind not in best_set]

    return drop_nets_inds


def trim_ensemble(votes_arr: ndarray, y_val: ndarray, acc_based: bool = True):
    final_votes = perform_vote(votes_arr)
    base_acc = metrics.accuracy_score(final_votes, y_val)
    if acc_based:
        print("base acc: " + str(base_acc))
    base_corr_ratio = np.sum(calc_correct_ratios(votes_arr, y_val))
    dropped_cols = []
    ind_counter = 0
    nr_nets = votes_arr.shape[1]
    for i in range(0, nr_nets):
        if votes_arr.shape[1] > 1:
            drop_col = votes_arr[:, ind_counter]

            votes_arr = np.delete(votes_arr, ind_counter, axis=1)
            final_votes_w_drop = perform_vote(votes_arr)

            if acc_based:
                new_acc = metrics.accuracy_score(final_votes_w_drop, y_val)
                if new_acc >= base_acc:
                    base_acc = new_acc
                    dropped_cols.append(i)
                    # print(new_acc)
                else:
                    votes_arr = np.column_stack((drop_col, votes_arr))
                    ind_counter = ind_counter + 1
            else:
                new_corr_ratio = np.sum(calc_correct_ratios(final_votes_w_drop, y_val))

                if new_corr_ratio >= base_corr_ratio:
                    base_corr_ratio = new_corr_ratio
                    # print(new_corr_ratio)
                    dropped_cols.append(i)
                else:
                    votes_arr = np.column_stack((drop_col, votes_arr))
                    ind_counter = ind_counter + 1

    # remaining = nr_nets - len(dropped_cols)
    # print("dropped " + str((len(dropped_cols)/nr_nets)*100) + " %, " + str(remaining) + " remaining")
    overlap = calc_overlap(votes_arr, y_val)
    # print("overlap: " + str(overlap))
    return dropped_cols, overlap


def calc_overlap(votes_arr: ndarray, y_val: ndarray):
    correct_ratios = calc_correct_ratios(votes_arr, y_val)

    return gini(np.asarray(correct_ratios))


# larger is more unequal
def gini(x: ndarray):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad / np.mean(x)
    g = 0.5 * rmad
    return g


def calc_correct_ratios(votes_arr: ndarray, y_val: ndarray):
    if len(votes_arr.shape) > 1:
        nr_nets = votes_arr.shape[1]
    else:
        nr_nets = 1
    correct_ratios = []

    for i in range(0, votes_arr.shape[0]):
        if nr_nets == 1:
            labels, counts = np.unique(votes_arr[i], return_counts=True)
        else:
            labels, counts = np.unique(votes_arr[i, :], return_counts=True)

        if y_val[i] in labels:
            ind_correct_label = np.where(labels == y_val[i])[0][0]
            nr_correct = counts[ind_correct_label]
        else:
            nr_correct = 0

        correct_ratio = nr_correct / float(nr_nets)
        correct_ratios.append(correct_ratio)

    return correct_ratios
