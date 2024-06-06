from product_detection.models.shell import Model_Lightning_Shell
import torch

model = Model_Lightning_Shell.load_from_checkpoint("outputs/2024-06-05/21-35-15/weights/epoch_epoch=9-val_loss=0.097.ckpt")

save_onnx = "weights/self_model.onnx"
input_sample = torch.randint(low=0, high=300, size=(1, 60))
model.to_onnx(save_onnx, input_sample, export_params=True)