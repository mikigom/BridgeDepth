import os, sys, argparse
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
import torch
from bridgedepth.bridgedepth import BridgeDepth


class BridgeDepthOnnx(BridgeDepth):
    @torch.no_grad()
    def forward(self, img1, img2):
        inputs = {'img1': img1, 'img2': img2}
        results = BridgeDepth.forward(self, inputs)
        disp = results['disp_pred']
        
        if disp.dim() == 4 and disp.shape[1] == 1:
            disp = disp.squeeze(1)
        return disp
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=f'{code_dir}/../onnx/', help='Path to save results.')
    parser.add_argument('--model_name', choices=['rvc', 'rvc_pretrain', 'eth3d_pretrain', 'middlebury_pretrain'], default='rvc_pretrain')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version (TensorRT often works best with 16-17)')
    parser.add_argument('--static', action='store_true', help='Export with fixed H,W (no dynamic axes). Recommended for TensorRT.')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)

    torch.autograd.set_grad_enabled(False)

    pretrained_model_name_or_path = args.model_name
    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path)
        pretrained_model_name_or_path = args.checkpoint_path
        model_name = os.path.splitext(os.path.basename(pretrained_model_name_or_path))[0]
    else:
        model_name = f"bridge_{args.model_name}"

    model = BridgeDepthOnnx.from_pretrained(pretrained_model_name_or_path)
    model = model.to(torch.device("cuda")).eval()

    img1 = torch.randn(1, 3, args.height, args.width).cuda().float()
    img2 = torch.randn(1, 3, args.height, args.width).cuda().float()

    opset_version = int(args.opset)
    output_file = os.path.join(args.save_dir, f"{model_name}_opset{opset_version}.onnx")
    
    print(f"try to export the ONNX (opset {opset_version})...")
    # Prepare dynamic axes only if not exporting static
    dynamic_axes = None
    if not args.static:
        dynamic_axes = {
            'left': {2: 'height', 3: 'width'},
            'right': {2: 'height', 3: 'width'},
            'disp': {1: 'height', 2: 'width'}
        }

    torch.onnx.export(
        model,
        (img1, img2),
        output_file,
        input_names=["left", "right"],
        output_names=["disp"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"success! ONNX file saved at {output_file}")