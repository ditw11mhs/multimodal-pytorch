from pathlib import Path

import streamlit as st
import torch
from matplotlib import pyplot as plt

from api import logic, utils


def initialization_flow():
    state = {}
    state["data_selection"] = st.selectbox("Select Dataset", ["MNIST"])
    with st.form("form"):
        root_path = Path.cwd() / "models" / state["data_selection"]

        model_path = sorted(d for d in root_path.iterdir() if d.is_dir())
        model_dict = {d.stem: d.as_posix() for d in model_path}

        model_selection = st.selectbox("Select Model", options=list(model_dict.keys()))

        state["sample_data"] = st.selectbox("Select Sample Data", options=[1, 2, 3])

        if st.form_submit_button("Inference"):
            state["model_path"] = model_dict[model_selection] + "/model.pt"
            return state


@st.cache_resource
def load_model(path):
    model = torch.load(path)
    model.eval()
    return model


def inference_flow(state: dict):
    root_path = Path.cwd()
    model = load_model(state["model_path"])

    if state["data_selection"] == "MNIST":
        sample_list = logic.load_sample(root_path)

    sample = sample_list[int(state["sample_data"]) - 1]

    with torch.no_grad():
        y_pred = model(sample[0].view(1, -1))
        prediction = torch.max(y_pred.data, 1)[1]  # check print(y_pred.data) to

    fig, ax = plt.subplots()
    ax.imshow(sample[0][0][0], cmap="gray")
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    c1.subheader(f"Ground Truth: {sample[1][0]}")
    c2.subheader(f"Prediction: {prediction[0]}")


def main():
    st.header("Inference App")
    state = initialization_flow()

    if state:
        inference_flow(state)


if __name__ == "__main__":
    main()
