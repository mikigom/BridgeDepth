import os
import argparse
import imageio.v2 as imageio
import cv2
import numpy as np
import open3d as o3d
import torch
import time

from bridgedepth.bridgedepth import BridgeDepth
from bridgedepth.utils.logger import setup_logger
from bridgedepth.utils import visualization


def depth2xyzmap(depth: np.ndarray, K, uvs: np.array=None, zmin=0.1):
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us] 
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N, 3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


# ========================= TensorRT helpers (optional) =========================

def _trt_dims_to_tuple(dims):
    try:
        return tuple(dims)
    except Exception:
        # Fallback for older TensorRT versions
        return tuple(dims.d[:dims.nbDims])


def _torch_dtype_from_trt(trt_dtype):
    import tensorrt as trt  # import lazily to avoid hard dependency when not used
    if trt_dtype == trt.DataType.FLOAT:
        return torch.float32
    if trt_dtype == trt.DataType.HALF:
        return torch.float16
    if trt_dtype == trt.DataType.INT32:
        return torch.int32
    # Default to float32 for unsupported types
    return torch.float32


def _load_trt_engine(engine_path):
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    assert engine is not None, "Failed to deserialize TensorRT engine."
    context = engine.create_execution_context()
    assert context is not None, "Failed to create TensorRT execution context."
    return runtime, engine, context


def _trt_is_io_api(engine):
    # Newer TensorRT API with named I/O tensors
    return hasattr(engine, 'num_io_tensors') and hasattr(engine, 'get_tensor_name')


def _get_static_input_hw(engine):
    import tensorrt as trt  # type: ignore
    if _trt_is_io_api(engine):
        # Prefer named input 'left'
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        left_name = 'left' if 'left' in input_names else (input_names[0] if input_names else None)
        if left_name is None:
            return None
        shape = _trt_dims_to_tuple(engine.get_tensor_shape(left_name))
        if len(shape) == 4 and all(isinstance(d, int) and d > 0 for d in shape):
            return int(shape[2]), int(shape[3])
        return None
    else:
        # Classic bindings API
        if not hasattr(engine, 'num_bindings'):
            return None
        binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
        left_idx = engine.get_binding_index('left') if 'left' in binding_names else -1
        if left_idx == -1:
            input_bindings = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
            if not input_bindings:
                return None
            left_idx = input_bindings[0]
        shape = _trt_dims_to_tuple(engine.get_binding_shape(left_idx))
        if len(shape) == 4 and all(isinstance(d, int) and d > 0 for d in shape):
            return int(shape[2]), int(shape[3])
        return None


# Use a non-default CUDA stream for TensorRT to avoid internal synchronizations on the default stream
_TRT_STREAM = None

def _get_trt_stream():
    global _TRT_STREAM
    if _TRT_STREAM is None:
        _TRT_STREAM = torch.cuda.Stream()
    return _TRT_STREAM


@torch.no_grad()
def _infer_with_trt(engine, context, img1_np: np.ndarray, img2_np: np.ndarray):
    """
    Run TensorRT inference with PyTorch CUDA tensors as device buffers.
    img1_np/img2_np are HxWx3 uint8 or float arrays already resized to the engine-required shape if static.
    Returns disparity map as numpy array HxW (float32).
    """
    import tensorrt as trt  # type: ignore

    N = 1
    H, W = int(img1_np.shape[0]), int(img1_np.shape[1])

    if _trt_is_io_api(engine):
        # Named I/O tensors path
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        left_name = 'left' if 'left' in input_names else input_names[0]
        right_name = 'right' if 'right' in input_names else (input_names[1] if len(input_names) > 1 else input_names[0])
        disp_name = 'disp' if 'disp' in output_names else output_names[0]

        left_shape = _trt_dims_to_tuple(engine.get_tensor_shape(left_name))
        dynamic_left = any(d == -1 for d in left_shape)
        if dynamic_left:
            # Context API uses set_input_shape for IO mode
            if hasattr(context, 'set_input_shape'):
                context.set_input_shape(left_name, (N, 3, H, W))
                context.set_input_shape(right_name, (N, 3, H, W))
            else:
                # Extremely old fallbacks are unlikely here if IO API is present
                raise RuntimeError('TensorRT context missing set_input_shape for IO API')

        in_left_dtype = _torch_dtype_from_trt(engine.get_tensor_dtype(left_name))
        in_right_dtype = _torch_dtype_from_trt(engine.get_tensor_dtype(right_name))
        out_dtype = _torch_dtype_from_trt(engine.get_tensor_dtype(disp_name))

        left_t = torch.as_tensor(img1_np, device='cuda').permute(2, 0, 1).unsqueeze(0).contiguous().to(in_left_dtype)
        right_t = torch.as_tensor(img2_np, device='cuda').permute(2, 0, 1).unsqueeze(0).contiguous().to(in_right_dtype)

        # Query output shape after setting input shapes
        out_shape = _trt_dims_to_tuple(context.get_tensor_shape(disp_name))
        if len(out_shape) == 3:
            out_N, out_H, out_W = int(out_shape[0]), int(out_shape[1]), int(out_shape[2])
        elif len(out_shape) == 4 and out_shape[1] in (1, 3):
            out_N, out_H, out_W = int(out_shape[0]), int(out_shape[2]), int(out_shape[3])
        else:
            out_N, out_H, out_W = N, H, W

        out_t = torch.empty((out_N, out_H, out_W), device='cuda', dtype=out_dtype).contiguous()

        # Set device addresses
        context.set_tensor_address(left_name, int(left_t.data_ptr()))
        context.set_tensor_address(right_name, int(right_t.data_ptr()))
        context.set_tensor_address(disp_name, int(out_t.data_ptr()))

        ok = False
        if hasattr(context, 'execute_async_v3'):
            trt_stream = _get_trt_stream()
            trt_stream.wait_stream(torch.cuda.current_stream())
            ok = context.execute_async_v3(int(trt_stream.cuda_stream))
            trt_stream.synchronize()
        elif hasattr(context, 'execute_v3'):
            ok = context.execute_v3()
        else:
            raise RuntimeError('TensorRT context lacks execute_v3/execute_async_v3 for IO API')
        assert ok, "TensorRT execution failed."

        disp = out_t.float().squeeze(0).detach().cpu().numpy()
        return disp
    else:
        # Classic bindings path
        assert hasattr(engine, 'num_bindings'), 'TensorRT engine does not expose classic bindings API'
        binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
        left_idx = engine.get_binding_index('left') if 'left' in binding_names else None
        right_idx = engine.get_binding_index('right') if 'right' in binding_names else None
        disp_idx = engine.get_binding_index('disp') if 'disp' in binding_names else None
        if left_idx is None or right_idx is None or disp_idx is None:
            input_bindings = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
            output_bindings = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
            assert len(input_bindings) >= 2 and len(output_bindings) >= 1, "Unexpected number of bindings in engine."
            left_idx = input_bindings[0]
            right_idx = input_bindings[1]
            disp_idx = output_bindings[0]

        left_shape = _trt_dims_to_tuple(engine.get_binding_shape(left_idx))
        dynamic_left = any(d == -1 for d in left_shape)
        if dynamic_left:
            context.set_binding_shape(left_idx, (N, 3, H, W))
            context.set_binding_shape(right_idx, (N, 3, H, W))

        in_left_dtype = _torch_dtype_from_trt(engine.get_binding_dtype(left_idx))
        in_right_dtype = _torch_dtype_from_trt(engine.get_binding_dtype(right_idx))
        out_dtype = _torch_dtype_from_trt(engine.get_binding_dtype(disp_idx))

        left_t = torch.as_tensor(img1_np, device='cuda').permute(2, 0, 1).unsqueeze(0).contiguous().to(in_left_dtype)
        right_t = torch.as_tensor(img2_np, device='cuda').permute(2, 0, 1).unsqueeze(0).contiguous().to(in_right_dtype)

        out_shape = _trt_dims_to_tuple(context.get_binding_shape(disp_idx))
        if len(out_shape) == 3:
            out_N, out_H, out_W = int(out_shape[0]), int(out_shape[1]), int(out_shape[2])
        elif len(out_shape) == 4 and out_shape[1] == 1:
            out_N, out_H, out_W = int(out_shape[0]), int(out_shape[2]), int(out_shape[3])
        else:
            out_N, out_H, out_W = N, H, W

        out_t = torch.empty((out_N, out_H, out_W), device='cuda', dtype=out_dtype).contiguous()

        bindings = [0] * engine.num_bindings
        bindings[left_idx] = int(left_t.data_ptr())
        bindings[right_idx] = int(right_t.data_ptr())
        bindings[disp_idx] = int(out_t.data_ptr())

        ok = False
        if hasattr(context, 'execute_async_v2'):
            trt_stream = _get_trt_stream()
            trt_stream.wait_stream(torch.cuda.current_stream())
            ok = context.execute_async_v2(bindings, int(trt_stream.cuda_stream))
            trt_stream.synchronize()
        else:
            ok = context.execute_v2(bindings)
        assert ok, "TensorRT execution failed."

        disp = out_t.float().squeeze(0).detach().cpu().numpy()
        return disp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default='assets/left.png', type=str)
    parser.add_argument('--right_file', default='assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default='assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--model_name', choices=['rvc', 'rvc_pretrain', 'eth3d_pretrain', 'middlebury_pretrain'], default='rvc_pretrain')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--out_dir', default='demo_output', type=str, help='the directory to save results')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--get_pc', type=int, default=0, help='save point cloud output')
    parser.add_argument('--trt_engine', default=None, type=str, help='Path to a TensorRT engine built from the provided ONNX (e.g., trtexec --onnx=... --saveEngine=...). When set, use TensorRT for inference instead of PyTorch.')
    parser.add_argument('--fps_iters', type=int, default=100, help='number of iterations for FPS measurement on same image (0 to disable)')
    parser.add_argument('--fps_warmup', type=int, default=10, help='warmup iterations before timing')
    args = parser.parse_args()

    logger = setup_logger(name="bridgedepth")
    logger.info("Arguments: " + str(args))
    os.makedirs(args.out_dir, exist_ok=True)

    use_trt = args.trt_engine is not None

    pretrained_model_name_or_path = args.model_name
    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path)
        pretrained_model_name_or_path = args.checkpoint_path

    if not use_trt:
        model = BridgeDepth.from_pretrained(pretrained_model_name_or_path)
        model = model.to(torch.device("cuda"))
        model.eval()
    else:
        # Load TensorRT engine
        try:
            runtime, trt_engine, trt_context = _load_trt_engine(args.trt_engine)
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine at {args.trt_engine}: {e}")

    img1 = imageio.imread(args.left_file)
    img2 = imageio.imread(args.right_file)
    H, W = img1.shape[:2]

    print(H, W)
    W_MAX = 1024
    scale = W_MAX / W

    # If using a static-shape TensorRT engine, resize to its required size
    if use_trt:
        static_hw = _get_static_input_hw(trt_engine)
        if static_hw is not None:
            target_h, target_w = static_hw
            img1 = cv2.resize(img1, dsize=(int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, dsize=(int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)
            # Recompute scale using actual width change
            scale = target_w / W
        else:
            # Keep original scaling policy if engine supports dynamic shapes
            img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
    else:
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)

    H, W = img1.shape[:2]
    logger.info(f"img1: {img1.shape}")

    viz = visualization.Visualizer(img1)

    if not use_trt:
        sample = {
            'img1': torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2),
            'img2': torch.as_tensor(img2).cuda().float()[None].permute(0, 3, 1, 2),
        }
        with torch.no_grad():
            results_dict = model(sample)
        disp = results_dict['disp_pred'].clamp_min(1e-3).cpu().numpy().reshape(H, W)
    else:
        disp = _infer_with_trt(trt_engine, trt_context, img1, img2)
        disp = np.clip(disp, 1e-3, None)

    vis = viz.draw_disparity(disp, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
    # vis = viz.draw_disparity(disp, colormap='kitti', enhance=False).get_image()
    vis = np.concatenate([img1, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    logger.info(f"Outputs saved to {args.out_dir}")

    # FPS measurement on the same image pair
    if getattr(args, 'fps_iters', 0) and args.fps_iters > 0:
        num_warmup = max(0, int(args.fps_warmup))
        num_iters = int(args.fps_iters)
        if not use_trt:
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(sample)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(num_iters):
                    _ = model(sample)
                torch.cuda.synchronize()
                elapsed = time.time() - t0
        else:
            for _ in range(num_warmup):
                _ = _infer_with_trt(trt_engine, trt_context, img1, img2)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(num_iters):
                _ = _infer_with_trt(trt_engine, trt_context, img1, img2)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
        fps = num_iters / elapsed if elapsed > 0 else float('inf')
        ms_per_frame = 1000.0 / fps if fps != float('inf') else 0.0
        logger.info(f"FPS over {num_iters} iters (warmup {num_warmup}): {fps:.2f} FPS, {ms_per_frame:.2f} ms/frame")

    if args.get_pc:
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
            baseline = float(lines[1])
        K[:2] *= scale
        depth = K[0, 0] * baseline / disp
        np.save(f'{args.out_dir}/depth_meter.npy', depth)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img1.reshape(-1, 3))
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
        keep_ids = np.arange(len(np.array(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
        logger.info(f"PCL saved to {args.out_dir}")

        logger.info("Visualization point cloud. Press ESC to exit.")
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
            vis.run()
            vis.destory_window()
        except:
            logger.info(f"Maybe you are in a headless environment! Please check the results stored under {args.out_dir}!")