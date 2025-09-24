# Adapted from FoundationStereo (https://github.com/NVlabs/FoundationStereo/blob/master/scripts/run_demo.py)
import os
import argparse
import imageio.v2 as imageio
import cv2
import numpy as np
import open3d as o3d
import torch

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


def _infer_with_tensorrt(engine_path: str, img_left_hwc: np.ndarray, img_right_hwc: np.ndarray) -> np.ndarray:
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import TensorRT Python API. Please ensure TensorRT is installed.") from exc

    # Prefer PyCUDA for buffer management; fallback to CUDA Python (cuda-python)
    use_pycuda = False
    cuda = None
    cudart = None
    stream = None

    try:
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401
        use_pycuda = True
        stream = cuda.Stream()
    except Exception:
        try:
            from cuda import cudart  # type: ignore
            # Create CUDA stream
            err, s = cudart.cudaStreamCreate()
            if err != 0:
                raise RuntimeError(f"cudaStreamCreate failed with error code {err}")
            stream = s
        except Exception as exc:
            raise RuntimeError(
                "TensorRT inference requires either PyCUDA ('pycuda') or CUDA Python ('cuda-python'). "
                "Please install one of them to run a TensorRT engine."
            ) from exc

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine from: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    # Prepare inputs as NCHW float32, batch=1
    inp_left = np.transpose(img_left_hwc.astype(np.float32), (2, 0, 1))[None, ...]
    inp_right = np.transpose(img_right_hwc.astype(np.float32), (2, 0, 1))[None, ...]

    # Resolve binding indices
    def _get_binding_index_safe(name: str, default_idx: int) -> int:
        try:
            idx = engine.get_binding_index(name)
            if idx is None or idx < 0:
                return default_idx
            return int(idx)
        except Exception:
            return default_idx

    left_idx = _get_binding_index_safe('left', 0)
    right_idx = _get_binding_index_safe('right', 1)
    out_idx = _get_binding_index_safe('disp', 2)

    # Set dynamic shapes
    context.set_binding_shape(left_idx, tuple(inp_left.shape))
    context.set_binding_shape(right_idx, tuple(inp_right.shape))

    # Derive output shape and dtype
    out_shape = tuple(context.get_binding_shape(out_idx))
    if len(out_shape) == 0 or any(int(d) < 0 for d in out_shape):
        # Fallback if shape not yet inferred
        H, W = img_left_hwc.shape[:2]
        out_shape = (1, H, W)

    trt_dtype = engine.get_binding_dtype(out_idx)
    np_dtype = np.float32 if trt_dtype == trt.DataType.FLOAT else (
        np.float16 if trt_dtype == trt.DataType.HALF else np.float32
    )

    # Allocate device buffers and copy inputs
    num_bindings = engine.num_bindings
    bindings = [0] * num_bindings

    if use_pycuda:
        # PyCUDA path
        d_left = cuda.mem_alloc(inp_left.nbytes)
        d_right = cuda.mem_alloc(inp_right.nbytes)
        d_out = cuda.mem_alloc(int(np.prod(out_shape)) * np.dtype(np_dtype).itemsize)

        cuda.memcpy_htod_async(d_left, np.ascontiguousarray(inp_left), stream)
        cuda.memcpy_htod_async(d_right, np.ascontiguousarray(inp_right), stream)

        bindings[left_idx] = int(d_left)
        bindings[right_idx] = int(d_right)
        bindings[out_idx] = int(d_out)

        ok = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        out_host = np.empty(out_shape, dtype=np_dtype)
        cuda.memcpy_dtoh_async(out_host, d_out, stream)
        stream.synchronize()

        # Convert to float32 for downstream usage
        disp = out_host.astype(np.float32)[0]
        return disp
    else:
        # CUDA Python (cudart) path
        # cudaMalloc returns (err, ptr)
        err, d_left = cudart.cudaMalloc(inp_left.nbytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc for left failed with error {err}")
        err, d_right = cudart.cudaMalloc(inp_right.nbytes)
        if err != 0:
            cudart.cudaFree(d_left)
            raise RuntimeError(f"cudaMalloc for right failed with error {err}")
        out_nbytes = int(np.prod(out_shape)) * np.dtype(np_dtype).itemsize
        err, d_out = cudart.cudaMalloc(out_nbytes)
        if err != 0:
            cudart.cudaFree(d_left)
            cudart.cudaFree(d_right)
            raise RuntimeError(f"cudaMalloc for out failed with error {err}")

        # Copy H->D
        err = cudart.cudaMemcpyAsync(
            d_left,
            int(np.ascontiguousarray(inp_left).ctypes.data),
            inp_left.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )[0]
        if err != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D (left) failed with error {err}")
        err = cudart.cudaMemcpyAsync(
            d_right,
            int(np.ascontiguousarray(inp_right).ctypes.data),
            inp_right.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )[0]
        if err != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D (right) failed with error {err}")

        bindings[left_idx] = int(d_left)
        bindings[right_idx] = int(d_right)
        bindings[out_idx] = int(d_out)

        ok = context.execute_async_v2(bindings=bindings, stream_handle=stream)
        if not ok:
            cudart.cudaFree(d_left); cudart.cudaFree(d_right); cudart.cudaFree(d_out)
            raise RuntimeError("TensorRT execution failed")

        out_host = np.empty(out_shape, dtype=np_dtype)
        err = cudart.cudaMemcpyAsync(
            int(out_host.ctypes.data),
            int(d_out),
            out_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )[0]
        if err != 0:
            cudart.cudaFree(d_left); cudart.cudaFree(d_right); cudart.cudaFree(d_out)
            raise RuntimeError(f"cudaMemcpyAsync D2H failed with error {err}")
        cudart.cudaStreamSynchronize(stream)

        # Free device buffers
        cudart.cudaFree(d_left); cudart.cudaFree(d_right); cudart.cudaFree(d_out)

        disp = out_host.astype(np.float32)[0]
        return disp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default='assets/left.png', type=str)
    parser.add_argument('--right_file', default='assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default='assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--model_name', choices=['rvc', 'rvc_pretrain', 'eth3d_pretrain', 'middlebury_pretrain'], default='rvc_pretrain')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--trt_engine', default=None, type=str, help='path to TensorRT engine (.engine); if set, use TensorRT for inference')
    parser.add_argument('--out_dir', default='demo_output', type=str, help='the directory to save results')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    args = parser.parse_args()

    logger = setup_logger(name="bridgedepth")
    logger.info("Arguments: " + str(args))
    os.makedirs(args.out_dir, exist_ok=True)

    use_trt = args.trt_engine is not None
    if use_trt:
        assert os.path.exists(args.trt_engine), f"TensorRT engine not found: {args.trt_engine}"
        logger.info(f"Using TensorRT engine: {args.trt_engine}")
    else:
        pretrained_model_name_or_path = args.model_name
        if args.checkpoint_path is not None:
            assert os.path.exists(args.checkpoint_path)
            pretrained_model_name_or_path = args.checkpoint_path

        model = BridgeDepth.from_pretrained(pretrained_model_name_or_path)
        model = model.to(torch.device("cuda"))
        model.eval()

    img1 = imageio.imread(args.left_file)
    img2 = imageio.imread(args.right_file)
    H, W = img1.shape[:2]
    
    W_MAX = 1024
    scale = W_MAX / W
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
    H, W = img1.shape[:2]
    logger.info(f"img1: {img1.shape}")

    viz = visualization.Visualizer(img1)

    if use_trt:
        disp = _infer_with_tensorrt(args.trt_engine, img1, img2)
        disp = np.maximum(disp, 1e-3)
    else:
        sample = {
            'img1': torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2),
            'img2': torch.as_tensor(img2).cuda().float()[None].permute(0, 3, 1, 2),
        }
        with torch.no_grad():
            results_dict = model(sample)
        disp = results_dict['disp_pred'].clamp_min(1e-3).cpu().numpy().reshape(H, W)

    vis = viz.draw_disparity(disp, colormap=cv2.COLORMAP_TURBO, enhance=False).get_image()
    # vis = viz.draw_disparity(disp, colormap='kitti', enhance=False).get_image()
    vis = np.concatenate([img1, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    logger.info(f"Outputs saved to {args.out_dir}")

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