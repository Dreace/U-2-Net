import torch
from model import U2NETP
import os

if __name__ == "__main__":
    # 配置路径
    checkpoint_dir = "."
    model_path = os.path.join(checkpoint_dir, "u2netp.pth")
    onnx_path = os.path.join(checkpoint_dir, "u2netp.onnx")

    # 构建模型并加载权重
    model = U2NETP(3, 1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 构造 dummy 输入
    dummy_input = torch.randn(1, 3, 320, 320)

    # 导出 ONNX，只导出主输出 d0
    class U2NETP_ONNX(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            d0, *_ = self.model(x)
            return d0

    export_model = U2NETP_ONNX(model)

    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"ONNX 模型已导出到: {onnx_path}")
