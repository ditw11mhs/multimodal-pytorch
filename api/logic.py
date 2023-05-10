from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from stqdm import stqdm
from torch.utils.data import DataLoader  # load data
from torchvision import datasets, transforms

from api.utils import ANNModelConfig, parse_list_string, save_lists_to_file


class ANNModel(nn.Module):
    def __init__(self, config: ANNModelConfig):
        super().__init__()
        self.config = config
        self.config.parsed_hidden_layer_nodes = self.parse_config()
        self.layers = []

        for index, n_layer in enumerate(self.config.parsed_hidden_layer_nodes[:-2]):
            self.layers.append(
                nn.Linear(n_layer, self.config.parsed_hidden_layer_nodes[index + 1])
            )
            self.layers.append(nn.ReLU())

        self.layers.append(
            nn.Linear(
                self.config.parsed_hidden_layer_nodes[-2],
                self.config.parsed_hidden_layer_nodes[-1],
            )
        )

        if self.config.output_mode == "Log Softmax":
            self.layers.append(nn.LogSoftmax(dim=1))
        elif self.config.output_mode == "Sigmoid":
            self.layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.layers)

    def parse_config(self):
        return (
            [self.config.n_input]
            + parse_list_string(self.config.hidden_layer_nodes)
            + [self.config.n_output]
        )

    def forward(self, x):
        x = self.model(x)
        return x


def load_dataset(root_path: Path, config: ANNModelConfig):
    dataset_folder = root_path / "data"
    transform_func = transforms.ToTensor()

    train_data = datasets.MNIST(
        root=dataset_folder.as_posix(),
        train=True,
        download=True,
        transform=transform_func,
    )
    test_data = datasets.MNIST(
        root=dataset_folder.as_posix(),
        train=False,
        download=True,
        transform=transform_func,
    )

    train_data_loader = DataLoader(
        train_data, batch_size=config.train_batch_size, shuffle=True
    )
    test_data_loader = DataLoader(
        test_data, batch_size=config.test_batch_size, shuffle=False
    )

    return train_data_loader, test_data_loader


def train_model(
    model: ANNModel,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    epoch_container,
    root_path,
):
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for epoch in stqdm(range(model.config.epochs), desc="Epochs", unit="epoch"):
        trn_corr = 0  # train correct) currently
        tst_corr = 0  # test correst

        #     # Run the training batches
        #     # with enumerate, we're actually going to keep track of what batch number we're on with B.
        if model.config.dataset == "MNIST":
            total_train = 60000 / model.config.train_batch_size

        batch_progress = stqdm(
            enumerate(train_loader), total=total_train, desc="Batches", unit="batch"
        )
        for batch, (
            x_train,
            y_train,
        ) in (
            batch_progress
        ):  # y_train=output = label, b = batches, train_loader = return back the image and its label
            batch += 1

            # Apply the model
            y_pred = model(
                x_train.view(model.config.train_batch_size, -1)
            )  # Here we flatten X_train
            train_loss = criterion(y_pred, y_train)  # calculating error difference
            # calculate the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[
                1
            ]  # check print(y_pred.data) to know data of one epoch, 1 = actual predicted value

            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            #
            #         # Update parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_acc = (trn_corr / 60000) * 100

        if model.config.dataset == "MNIST":
            epoch_container.write(
                f"epoch: {epoch:2} loss: {train_loss:10.8f} accuracy: {train_acc:7.3f}%"
            )
            train_correct.append(train_acc)
        #
        #
        #     # Update train loss & accuracy for the epoch
        train_losses.append(train_loss.item())

        #
        #     # Run the testing batches
        with torch.no_grad():  # don't update weight and bias in test data
            for x_test, y_test in test_loader:
                # Apply the model
                y_val = model(
                    x_test.view(model.config.test_batch_size, -1)
                )  # Here we flatten X_test, 500 because batch size for test data in cell above = 500

                # Calculating the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test).sum()
                #
                #     # Update test loss & accuracy for the epoch
                test_loss = criterion(y_val, y_test)

        test_losses.append(test_loss.item())
        test_acc = (tst_corr / 60000) * 100

        if model.config.dataset == "MNIST":
            test_correct.append(test_acc)

    parent_model_path = (
        root_path / "models" / model.config.dataset / model.config.model_name
    )
    parent_model_path.mkdir(parents=True, exist_ok=True)

    model_path = parent_model_path / "model.pt"
    torch.save(model, model_path.as_posix())
    epoch_container.write(f"Model saved to {model_path.as_posix()}")

    history_path = parent_model_path / "history.txt"
    save_lists_to_file(
        history_path.as_posix(), train_losses, test_losses, train_correct, test_correct
    )
    epoch_container.write(f"History saved to {history_path.as_posix()}")

    return train_losses, test_losses, train_correct, test_correct


def get_loss(config: ANNModelConfig):
    if config.criterion == "Cross Entropy Loss":
        return nn.CrossEntropyLoss()


def get_optimizer(model: ANNModel):
    if model.config.optimizer == "Adam":
        return torch.optim.Adam(model.parameters())
