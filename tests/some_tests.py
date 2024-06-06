import torch
import onnxruntime as ort
import pandas as pd

from prod_functions import get_result
from product_detection.data import load_dataset
from product_detection.config import KNOWN_CLASSES

print("Still alive, and that`s progress")

def onnx_answer_test():
    save_onnx = "weights/self_model.onnx"
    ort_session = ort.InferenceSession(save_onnx)

    input_sample = torch.randint(low=0, high=300, size=(1, 60))

    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_sample.numpy()}

    ort_outs = ort_session.run(None, ort_inputs)

    tensor_outputs = torch.tensor(ort_outs).squeeze()[-1, :]
    print(tensor_outputs.shape)
    print(tensor_outputs)
    answer_class = torch.argmax(tensor_outputs)
    print(answer_class, KNOWN_CLASSES[answer_class])

def get_result_test():
    x, y = load_dataset("datasets/testmeat.csv", min_count=1)
    to_pred = pd.Series(x)

    answer = get_result(to_pred)

    for i in range(len(answer)):
        print(to_pred[i])
        print(answer[i])
        print("-----")

if __name__ == "__main__":
    get_result_test()