from datetime import datetime
from pathlib import Path

import streamlit as st
from matplotlib import pyplot as plt

from api import logic, utils


def initialization_flow():
    with st.form("Training Parameter"):
        c1, c2 = st.columns(2)
        dataset = c1.selectbox("Select Dataset", ["MNIST"])
        model_name = c2.text_input(
            "Model Name", value=f"model-{datetime.now().isoformat()}"
        )

        c3, c4, c5 = st.columns(3)
        n_input = int(c3.number_input("Number of input", min_value=1, value=784))
        n_output = int(c4.number_input("Number of output", min_value=1, value=10))
        hidden_layer_nodes = c5.text_input("Nodes per layer", value="120,84")

        c6, c7, c8 = st.columns(3)
        train_batch_size = int(
            c6.number_input("Train Batch Size", min_value=1, value=100)
        )
        test_batch_size = int(
            c7.number_input("Test Batch Size", min_value=1, value=500)
        )
        epochs = int(c8.number_input("Epochs", min_value=1, value=10))

        c9, c10, c11 = st.columns(3)
        criterion = c9.selectbox("Loss", ["Cross Entropy Loss"])
        optimizer = c10.selectbox("Optimizer", ["Adam"])
        output_mode = c11.selectbox("Output Mode", ["Log Softmax", "Log Sigmoid"])

        if st.form_submit_button("Start Training"):
            return utils.ANNModelConfig(
                dataset=dataset,
                model_name=model_name,
                n_input=n_input,
                n_output=n_output,
                hidden_layer_nodes=hidden_layer_nodes,
                output_mode=output_mode,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                epochs=epochs,
                criterion=criterion,
                optimizer=optimizer,
            )


def train_flow(config: utils.ANNModelConfig, epoch_container):
    ### Load Dataset
    with st.spinner("Loading Dataset..."):
        root_folder = Path.cwd()
        train_loader, test_loader = logic.load_dataset(root_folder, config)

    # Choose model hyperparameter
    with st.spinner("Building Model..."):
        model = logic.ANNModel(config)

        criterion = logic.get_loss(config)
        optimizer = logic.get_optimizer(model)

    # Train model
    train_losses, test_losses, train_correct, test_correct = logic.train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        epoch_container,
        root_folder,
    )

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(train_losses, label="Train losses")
    ax_loss.plot(test_losses, label="Test losses")
    ax_loss.legend()
    st.pyplot(fig_loss)

    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(train_correct, label="Train acc")
    ax_acc.plot(test_correct, label="Test acc")
    ax_acc.legend()
    st.pyplot(fig_acc)


def main():
    st.header("Training App")

    state = initialization_flow()

    if state:
        epoch_container = st.container()
        st.button("Stop")
        train_flow(state, epoch_container)


if __name__ == "__main__":
    main()
