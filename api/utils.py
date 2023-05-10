from dataclasses import dataclass
from typing import List, Union

import plotly.graph_objects as go


@dataclass
class ANNModelConfig:
    dataset: Union[str, None]
    model_name: str
    n_input: int
    n_output: int
    hidden_layer_nodes: str
    output_mode: Union[str, None]
    train_batch_size: int
    test_batch_size: int
    epochs: int
    criterion: Union[str, None]
    optimizer: Union[str, None]
    parsed_hidden_layer_nodes: Union[List[int], None] = None

    def __setattr__(self, prop, val):
        if validator := getattr(self, f"validate_{prop}", None):
            object.__setattr__(self, prop, validator(val) or val)
        else:
            super().__setattr__(prop, val)

    def validate_dataset(self, val):
        if val not in ["MNIST"]:
            raise ValueError("Available dataset is MNIST")

    def validate_model_name(self, val):
        pass

    def validate_criterion(self, val):
        if val not in ["Cross Entropy Loss"]:
            raise ValueError("Available criterion: Cross Entropy Loss")

    def validate_optimizer(self, val):
        if val not in ["Adam"]:
            raise ValueError("Available optimizer: Adam")

    def validate_n_input(self, val):
        if val <= 0:
            raise ValueError("Number of model input needed to be >= 1")

    def validate_train_batch_size(self, val):
        if val <= 0:
            raise ValueError("Number of train batch size needed to be >= 1")

    def validate_test_batch_size(self, val):
        if val <= 0:
            raise ValueError("Number of test batch size needed to be >= 1")

    def validate_epochs(self, val):
        if val <= 0:
            raise ValueError("Number of epochs needed to be >= 1")

    def validate_n_output(self, val):
        if val <= 0:
            raise ValueError("Number of model output needed to be >= 1")

    def validate_hidden_layer_nodes(self, val):
        try:
            parsed_layer = parse_list_string(val)
        except:
            raise ValueError(
                "Wrong format on nodes per layer it needed to be num,num,num"
            )

    def validate_parsed_hidden_layer_nodes(self, val):
        if val:
            for n_layer in val:
                if not isinstance(n_layer, int):
                    raise ValueError(
                        "Prased hidden layer nodes needed to be a list of integer"
                    )

    def validate_output_mode(self, val):
        if val not in ["Log Softmax", "Sigmoid"]:
            raise ValueError("Valid output mode is either Log Softmax or Sigmoid")


def parse_list_string(list_string):
    """
    Parses a string representation of a list of integers, separated by commas,
    into a Python list of integers.

    Args:
        list_string (str): A string representation of a list of integers, e.g.
            "512,512,1".

    Returns:
        list: A list of integers parsed from the input string, e.g. [512, 512, 1].
    """
    out_list = list_string.split(",")
    out_list = [int(num) for num in out_list]
    return out_list


def save_lists_to_file(file_path, list1, list2, list3, list4):
    with open(file_path, "w") as f:
        for item1, item2, item3, item4 in zip(list1, list2, list3, list4):
            f.write(f"{item1}|{item2}|{item3}|{item4}\n")


def load_lists_from_file(file_path):
    list1 = []
    list2 = []
    list3 = []
    list4 = []

    with open(file_path, "r") as f:
        for line in f:
            item1, item2, item3, item4 = line.strip().split("|")
            list1.append(item1)
            list2.append(item2)
            list3.append(item3)
            list4.append(item4)

    return list1, list2, list3, list4


def add_one_line(fig, df, x, y, name):
    fig.add_trace(
        go.Scattergl(
            x=df[x],
            y=df[y],
            name=name,
            mode="lines",
        )
    )


def add_one_scatter(fig, df, x, y, name):
    fig.add_trace(
        go.Scattergl(
            x=df[x],
            y=df[y],
            name=name,
            mode="markers",
        )
    )


def add_line(fig, df, x, y, names):
    if len(y) != len(names):
        raise Exception("Length y != Length name")

    for col, name in zip(y, names):
        add_one_line(fig, df, x, col, name)


def add_scatter(fig, df, x, y, names):
    if len(y) != len(names):
        raise Exception("Length y != Length name")

    for col, name in zip(y, names):
        add_one_scatter(fig, df, x, col, name)
