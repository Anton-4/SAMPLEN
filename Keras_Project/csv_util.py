import csv
import inspect
from typing import List, Tuple

# from keras.callbacks import History

from params.net_params import NetParams
from params.rambo_net_params import RamboNetParams


def save_param_list(
    net_param_list: List[NetParams],
    file_name: str,
    mode: str = "w",
    with_header: bool = True,
    rambo_params: bool = True
):
    with open(file_name, mode) as csvfile:
        if with_header:
            if rambo_params:
                header = inspect.signature(RamboNetParams.__init__)
            else:
                header = inspect.signature(NetParams.__init__)
            str_header = [str(param) for param in header.parameters.values()]
            csv_header = [col_title.split(":")[0] for col_title in str_header[1:]]
            filter_params = ["net_weights"]
            csv_header = [
                col_title for col_title in csv_header if col_title not in filter_params
            ]
            csvfile.write(";".join(csv_header) + "\n")

        val_accs = [params.val_acc for params in net_param_list]
        val_inds = list(range(len(val_accs)))
        zipped: List[Tuple[int, int]] = list(zip(val_accs, val_inds))
        zipped.sort(key=lambda t: t[0], reverse=True)
        for _, ind in zipped:
            row = str(net_param_list[ind])
            csvfile.write(row + "\n")

        csvfile.close()


# def save_history(net_id: str, history: History):
    # hist_file = "hyperopt_results/history.csv"
    # with open(hist_file, "a") as csvfile:
    #     csvfile.write(
    #         net_id
    #         + ";"
    #         + str(history.history["val_acc"])
    #         + ";"
    #         + str(history.history["acc"])
    #         + "\n"
    #     )


def load_ids_from_submission(sub_file_name: str):
    with open(sub_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # skip header
        reader.__next__()
        ids = []
        for row in reader:
            ids.append(row[0])

    return ids
