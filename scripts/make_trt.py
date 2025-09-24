#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _require_tensorrt():
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(
            "ERROR: Failed to import TensorRT Python API. Please ensure TensorRT is installed and available in PYTHONPATH.",
            file=sys.stderr,
        )
        raise
    return trt


def _parse_dims(dims_str: str) -> Tuple[int, ...]:
    """Parse a dimension string like '1x3x224x224' or '1,3,224,224' to a tuple of ints."""
    cleaned = dims_str.strip().lower().replace(" ", "").replace(",", "x")
    if not cleaned:
        raise ValueError("Empty shape string")
    try:
        dims = tuple(int(tok) for tok in cleaned.split("x") if tok)
    except ValueError:
        raise ValueError(f"Invalid dimension token in '{dims_str}'")
    if not dims:
        raise ValueError(f"No valid dimensions parsed from '{dims_str}'")
    if any(d <= 0 for d in dims):
        raise ValueError(f"All dimensions must be > 0 in '{dims_str}'")
    return dims


def _parse_name_dims(entries: Optional[Iterable[str]]) -> Dict[str, Tuple[int, ...]]:
    """Parse entries like ['input=1x3x224x224', 'image=8x3x256x256'] into a dict."""
    result: Dict[str, Tuple[int, ...]] = {}
    if not entries:
        return result
    for raw in entries:
        if raw is None:
            continue
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Shape entry must be of the form name=d1x...xdk, got '{raw}'"
            )
        name, shape = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing input name in '{raw}'")
        dims = _parse_dims(shape)
        result[name] = dims
    return result


def _any_dynamic(shape: Sequence[int]) -> bool:
    return any(int(d) < 0 for d in shape)


# ----------------------
# Optional ONNX simplify
# ----------------------

def _maybe_simplify_onnx(
    onnx_path: Path,
    shape_hints: Optional[Dict[str, Tuple[int, ...]]],
    verbose: bool,
) -> Path:
    """Try to simplify the ONNX to remove control-flow and unnecessary casts.
    Returns the path to the simplified model (or the original path if simplification is unavailable or fails)."""
    try:
        import onnx  # type: ignore
        from onnxsim import simplify  # type: ignore
    except Exception:
        if verbose:
            print("onnx-simplifier not available; skipping simplification.")
        return onnx_path

    try:
        model = onnx.load(str(onnx_path))
        input_shapes = None
        if shape_hints:
            # onnx-simplifier expects list[int] shapes
            input_shapes = {name: list(map(int, dims)) for name, dims in shape_hints.items()}
        simplified_model, ok = simplify(model, input_shapes=input_shapes)
        if not ok:
            if verbose:
                print("onnx-simplifier reported the model could not be simplified; proceeding with original.")
            return onnx_path
        sim_path = onnx_path.with_suffix(".sim.onnx")
        onnx.save(simplified_model, str(sim_path))
        if verbose:
            print(f"Saved simplified ONNX to: {sim_path}")
        return sim_path
    except Exception as exc:
        print(f"WARNING: ONNX simplification failed: {exc}", file=sys.stderr)
        return onnx_path


# ----------------------
# Optional trtexec fallback
# ----------------------

def _try_trtexec(
    onnx_path: Path,
    engine_path: Path,
    workspace_mib: int,
    fp16: bool,
    verbose: bool,
    min_shapes: Dict[str, Tuple[int, ...]],
    opt_shapes: Dict[str, Tuple[int, ...]],
    max_shapes: Dict[str, Tuple[int, ...]],
    plugin_paths: Sequence[Path] = (),
) -> bool:
    """Try to build using `trtexec` binary if available. Returns True on success."""
    import shutil
    import subprocess

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        if verbose:
            print("trtexec not found in PATH; skipping trtexec fallback.")
        return False

    def fmt_shape(dims: Tuple[int, ...]) -> str:
        return "x".join(str(int(d)) for d in dims)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mib}",
        "--noDataTransfers",
    ]
    if fp16:
        cmd.append("--fp16")
    if verbose:
        cmd.append("--verbose")

    if min_shapes:
        cmd.append("--minShapes=" + ",".join(f"{k}:{fmt_shape(v)}" for k, v in min_shapes.items()))
    if opt_shapes:
        cmd.append("--optShapes=" + ",".join(f"{k}:{fmt_shape(v)}" for k, v in opt_shapes.items()))
    if max_shapes:
        cmd.append("--maxShapes=" + ",".join(f"{k}:{fmt_shape(v)}" for k, v in max_shapes.items()))

    for plugin in plugin_paths:
        cmd.append(f"--plugins={plugin}")

    if verbose:
        print("Running trtexec:", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, check=False, capture_output=not verbose, text=True)
        if proc.returncode == 0 and engine_path.exists():
            return True
        if not verbose:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: trtexec failed to run: {exc}", file=sys.stderr)
    return False


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_mib: int,
    fp16: bool,
    verbose: bool,
    min_shapes: Dict[str, Tuple[int, ...]],
    opt_shapes: Dict[str, Tuple[int, ...]],
    max_shapes: Dict[str, Tuple[int, ...]],
) -> None:
    trt = _require_tensorrt()

    # Logger
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger = trt.Logger(severity)

    # Builder and Network
    builder = trt.Builder(logger)
    # Try to enable EXPLICIT_BATCH if available
    try:
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except Exception:
        explicit_batch_flag = 0
    network = builder.create_network(explicit_batch_flag)

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        print(f"ERROR: Failed to parse ONNX file '{onnx_path}'. Parser errors:", file=sys.stderr)
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise SystemExit(1)

    # Builder Config
    config = builder.create_builder_config()
    # Workspace size: prefer new API if available
    try:
        # TensorRT >= 8.6
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) * (1 << 20))
    except Exception:
        # Older TensorRT
        config.max_workspace_size = int(workspace_mib) * (1 << 20)

    # Precision flags
    if fp16:
        try:
            if getattr(builder, "platform_has_fast_fp16", True):
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("WARNING: Platform does not report fast FP16 support. Proceeding anyway.")
                config.set_flag(trt.BuilderFlag.FP16)
        except Exception:
            # Older TensorRT may not have set_flag
            try:
                config.flags = config.flags | trt.BuilderFlag.FP16
            except Exception:
                print("WARNING: Unable to enable FP16 on this TensorRT version.")

    # Optimization profile for dynamic shapes (required if any input has dynamic dims)
    need_profile = False
    input_names: List[str] = []
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        input_names.append(inp.name)
        if _any_dynamic(inp.shape):
            need_profile = True

    if need_profile:
        profile = builder.create_optimization_profile()

        # Allow a convenience: if only --opt-shape or --static-shape provided, promote to min/max
        if not min_shapes and opt_shapes:
            min_shapes = dict(opt_shapes)
        if not max_shapes and opt_shapes:
            max_shapes = dict(opt_shapes)

        # Validate provided shapes cover all dynamic inputs
        missing_min = []
        missing_opt = []
        missing_max = []
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            if _any_dynamic(inp.shape):
                if inp.name not in min_shapes:
                    missing_min.append(inp.name)
                if inp.name not in opt_shapes:
                    missing_opt.append(inp.name)
                if inp.name not in max_shapes:
                    missing_max.append(inp.name)
        if missing_min or missing_opt or missing_max:
            lines = [
                "Dynamic input shapes detected. Please specify all of --min-shape, --opt-shape and --max-shape for each dynamic input.",
                f"Inputs: {input_names}",
            ]
            if missing_min:
                lines.append(f"Missing --min-shape for: {missing_min}")
            if missing_opt:
                lines.append(f"Missing --opt-shape for: {missing_opt}")
            if missing_max:
                lines.append(f"Missing --max-shape for: {missing_max}")
            raise SystemExit("\n".join(lines))

        # Set shapes in the profile
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            name = inp.name
            if _any_dynamic(inp.shape):
                try:
                    profile.set_shape(name, min_shapes[name], opt_shapes[name], max_shapes[name])
                except Exception as exc:
                    raise SystemExit(
                        f"Failed to set optimization profile for input '{name}'. "
                        f"Provided min/opt/max: {min_shapes.get(name)}, {opt_shapes.get(name)}, {max_shapes.get(name)}.\n{exc}"
                    )

        config.add_optimization_profile(profile)

    # Build serialized engine (TensorRT >= 8) or engine then serialize (older TRT)
    serialized: Optional[bytes] = None
    try:
        host_mem = builder.build_serialized_network(network, config)
        if host_mem is None:
            raise RuntimeError("builder.build_serialized_network() returned None")
        # host_mem supports buffer protocol; cast to bytes for safety
        serialized = bytes(host_mem)
    except AttributeError:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("builder.build_engine() returned None")
        try:
            host_mem = engine.serialize()
            serialized = bytes(host_mem)
        except Exception:
            # Fallback: some versions allow writing IHostMemory directly
            host_mem = engine.serialize()
            serialized = host_mem  # type: ignore

    # Save engine
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)  # type: ignore[arg-type]


def _load_plugins(plugin_paths: Sequence[Path], verbose: bool) -> None:
    """Dynamically load TensorRT plugin libraries (.so) into the current process."""
    if not plugin_paths:
        return
    trt = _require_tensorrt()
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    try:
        # Initialize core TRT/official plugins if available
        trt.init_libnvinfer_plugins(logger, "")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import ctypes  # Local import to avoid hard dependency if unused
    except Exception:
        print("WARNING: ctypes is unavailable; cannot load plugin libraries.", file=sys.stderr)
        return
    for lib_path in plugin_paths:
        try:
            ctypes.CDLL(str(lib_path))
            if verbose:
                print(f"Loaded TensorRT plugin library: {lib_path}")
        except Exception as exc:
            print(f"WARNING: Failed to load plugin library '{lib_path}': {exc}", file=sys.stderr)


def _warn_suspect_onnx_ops(onnx_path: Path, verbose: bool) -> None:
    """Emit a warning if the ONNX graph contains ops commonly unsupported by TensorRT."""
    try:
        import onnx  # type: ignore
    except Exception:
        return
    try:
        model = onnx.load(str(onnx_path))
        ops = sorted({node.op_type for node in model.graph.node})
        suspect_set = {"If", "Loop", "Range", "Mod", "ScatterND", "GatherElements", "GridSample"}
        suspects = [op for op in ops if op in suspect_set]
        if suspects:
            print(
                "WARNING: Detected ONNX ops that TensorRT may not support natively: " + ", ".join(suspects),
                file=sys.stderr,
            )
            if verbose:
                print(
                    "Consider simplifying the graph, exporting without control-flow, or providing plugin libraries via --plugin.",
                    file=sys.stderr,
                )
    except Exception:
        # Best-effort only
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model to a TensorRT engine (.engine)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("onnx", type=Path, help="Path to ONNX model file")
    parser.add_argument(
        "--engine",
        type=Path,
        default=None,
        help="Output TensorRT engine path. Defaults to <onnx_basename>.engine",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1024,
        help="Workspace size in MiB",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision (if supported)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose TensorRT logging",
    )
    parser.add_argument(
        "--plugin",
        dest="plugins",
        action="append",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a TensorRT plugin library (.so). Repeatable.",
    )

    # Dynamic shape profiles
    parser.add_argument(
        "--min-shape",
        dest="min_shapes",
        action="append",
        default=None,
        metavar="NAME=DIMx...",
        help="Minimum shape for a dynamic input (repeatable)",
    )
    parser.add_argument(
        "--opt-shape",
        dest="opt_shapes",
        action="append",
        default=None,
        metavar="NAME=DIMx...",
        help="Optimal shape for a dynamic input (repeatable)",
    )
    parser.add_argument(
        "--max-shape",
        dest="max_shapes",
        action="append",
        default=None,
        metavar="NAME=DIMx...",
        help="Maximum shape for a dynamic input (repeatable)",
    )
    parser.add_argument(
        "--static-shape",
        dest="static_shapes",
        action="append",
        default=None,
        metavar="NAME=DIMx...",
        help="Convenience: set min/opt/max to the same shape for a given input (repeatable)",
    )

    args = parser.parse_args(argv)

    onnx_path: Path = args.onnx
    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        return 1

    engine_path: Path = args.engine if args.engine is not None else onnx_path.with_suffix(".engine")

    min_shapes = _parse_name_dims(args.min_shapes)
    opt_shapes = _parse_name_dims(args.opt_shapes)
    max_shapes = _parse_name_dims(args.max_shapes)

    # Apply static shapes if provided
    static_shapes = _parse_name_dims(args.static_shapes)
    for name, dims in static_shapes.items():
        min_shapes[name] = dims
        opt_shapes[name] = dims
        max_shapes[name] = dims

    # Optional: warn about likely unsupported ops
    _warn_suspect_onnx_ops(onnx_path, verbose=bool(args.verbose))

    # Optional: load user-provided plugin libraries
    plugin_paths: List[Path] = list(args.plugins or [])
    _load_plugins(plugin_paths, verbose=bool(args.verbose))

    # Build attempts: try original, then simplified ONNX, and toggle FP16 off if necessary. Finally try trtexec.
    attempt_descriptions: List[str] = []

    # Determine shape hints for ONNX simplifier from opt->max->min
    simplifier_shape_hints: Dict[str, Tuple[int, ...]] = {}
    for src in (opt_shapes, max_shapes, min_shapes):
        for k, v in src.items():
            simplifier_shape_hints.setdefault(k, v)

    # Attempt 1: original ONNX with requested precision
    try:
        build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            workspace_mib=int(args.workspace),
            fp16=bool(args.fp16),
            verbose=bool(args.verbose),
            min_shapes=min_shapes,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes,
        )
        print(f"Saved TensorRT engine to: {engine_path}")
        return 0
    except SystemExit as exc:
        # Propagate cleanly without stacktrace for expected argument/profile issues
        if exc.code not in (None, 0):
            return int(exc.code)
        return 0
    except Exception as exc:
        attempt_descriptions.append(f"original onnx (fp16={bool(args.fp16)}) -> {exc}")
        if args.verbose:
            print(f"First build attempt failed: {exc}")

    # Attempt 2: Simplify ONNX and retry with same precision
    sim_onnx_path = _maybe_simplify_onnx(onnx_path, simplifier_shape_hints or None, verbose=bool(args.verbose))
    if sim_onnx_path != onnx_path:
        try:
            build_engine(
                onnx_path=sim_onnx_path,
                engine_path=engine_path,
                workspace_mib=int(args.workspace),
                fp16=bool(args.fp16),
                verbose=bool(args.verbose),
                min_shapes=min_shapes,
                opt_shapes=opt_shapes,
                max_shapes=max_shapes,
            )
            print(f"Saved TensorRT engine to: {engine_path}")
            return 0
        except Exception as exc:
            attempt_descriptions.append(f"simplified onnx (fp16={bool(args.fp16)}) -> {exc}")
            if args.verbose:
                print(f"Second build attempt (simplified) failed: {exc}")

    # Attempt 3: Retry with FP32 (disable fp16)
    try:
        build_engine(
            onnx_path=sim_onnx_path if sim_onnx_path.exists() else onnx_path,
            engine_path=engine_path,
            workspace_mib=int(args.workspace),
            fp16=False,
            verbose=bool(args.verbose),
            min_shapes=min_shapes,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes,
        )
        print(f"Saved TensorRT engine to: {engine_path}")
        return 0
    except Exception as exc:
        attempt_descriptions.append("retry with fp16=False -> %s" % exc)
        if args.verbose:
            print(f"Third build attempt (fp32) failed: {exc}")

    # Attempt 4: trtexec fallback
    if _try_trtexec(
        onnx_path=sim_onnx_path if sim_onnx_path.exists() else onnx_path,
        engine_path=engine_path,
        workspace_mib=int(args.workspace),
        fp16=bool(args.fp16),
        verbose=bool(args.verbose),
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        plugin_paths=plugin_paths,
    ):
        print(f"Saved TensorRT engine to: {engine_path}")
        return 0

    print("ERROR: Engine build failed after retries:")
    for desc in attempt_descriptions:
        print(f" - {desc}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
