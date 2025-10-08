import torch
from mmdet.apis import init_detector
import argparse

def export_to_onnx(config, checkpoint, output_file, input_shape=(640, 640), opset=11, device='cuda:0'):
    model = init_detector(config, checkpoint, device=device)
    model.eval()

    dummy_input = torch.randn(1, 3, *input_shape).to(device)

    # 3. Exportar para ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={
            "input": {0: "batch"},   
            "output": {0: "batch"}
        }
    )

    print(f"[OK] Modelo exportado para ONNX em: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportar modelo MMDetection para ONNX")
    parser.add_argument("--config", type=str, required=True, help="Arquivo de config do modelo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint treinado (.pth)")
    parser.add_argument("--output", type=str, default="model.onnx", help="Nome do arquivo ONNX de saída")
    parser.add_argument("--shape", type=int, nargs=2, default=[640, 640], help="Tamanho da entrada (H W)")
    parser.add_argument("--opset", type=int, default=11, help="Versão do ONNX opset")
    parser.add_argument("--device", type=str, default="cuda:0", help="Dispositivo de exportação (cuda:0 ou cpu)")
    args = parser.parse_args()

    export_to_onnx(args.config, args.checkpoint, args.output, tuple(args.shape), args.opset, args.device)
