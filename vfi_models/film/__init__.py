import os
import torch
from comfy.model_management import get_torch_device, soft_empty_cache
import bisect
import numpy as np
import typing
from vfi_utils import InterpolationStateList, load_file_from_github_release, preprocess_frames, postprocess_frames
import pathlib
import gc

from .film_arch import Interpolator

MODEL_TYPE = pathlib.Path(__file__).parent.name
DEVICE = get_torch_device()

_ALIAS_ANALYSIS_ERROR_SNIPPETS = (
    "alias_analysis.cpp",
    "Types must be strictly equal if you are replacing aliasing information",
)


def _is_jit_alias_analysis_failure(exc: RuntimeError) -> bool:
    msg = str(exc)
    return all(snippet in msg for snippet in _ALIAS_ANALYSIS_ERROR_SNIPPETS)


class _FilmModelRunner:
    """
    Prefer the existing TorchScript path when it works.
    If the current PyTorch JIT trips the known alias-analysis internal assert,
    rebuild the same FILM weights into the eager Interpolator implementation and
    continue with that backend for the rest of the run.
    """

    def __init__(self, model_path: str):
        self._script_model = torch.jit.load(model_path, map_location="cpu")
        self._script_model.eval()
        self._cpu_state_dict = {
            key: value.detach().cpu()
            for key, value in self._script_model.state_dict().items()
        }
        self._script_model = self._script_model.to(DEVICE)
        self._eager_model = None

    def _build_eager_model(self):
        eager = Interpolator()
        eager.load_state_dict(self._cpu_state_dict, strict=True)
        eager.eval()

        self._cpu_state_dict = None

        old_script_model = self._script_model
        self._script_model = None
        del old_script_model
        gc.collect()
        soft_empty_cache()

        eager = eager.to(DEVICE)

        compile_eager = os.environ.get("COMFYUI_FILM_COMPILE_EAGER", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if compile_eager and hasattr(torch, "compile"):
            try:
                eager = torch.compile(eager, dynamic=False)
                print("FILM VFI: compiled eager fallback enabled")
            except Exception as exc:
                print(f"FILM VFI: eager fallback compile skipped: {exc}")

        self._eager_model = eager
        return eager

    def __call__(self, x0, x1, dt):
        if self._eager_model is not None:
            return self._eager_model(x0, x1, dt)

        try:
            return self._script_model(x0, x1, dt)
        except RuntimeError as exc:
            if not _is_jit_alias_analysis_failure(exc):
                raise
            print("FILM VFI: TorchScript alias-analysis crash detected; switching to eager FILM backend for this run.")
            return self._build_eager_model()(x0, x1, dt)


def inference(model, img_batch_1, img_batch_2, inter_frames):
    results = [
        img_batch_1,
        img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i].to(DEVICE)
        x1 = results[end_i].to(DEVICE)
        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).float())
        del remains[step]

    return [tensor.flip(0) for tensor in results]

class FILM_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["film_net_fp32.pt"], ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"        

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        interpolation_states = optional_interpolation_states
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        model = _FilmModelRunner(model_path)

        frames = preprocess_frames(frames)
        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []
        
        if type(multiplier) == int:
            multipliers = [multiplier] * len(frames)
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (len(frames) - len(multipliers) - 1)
        for frame_itr in range(len(frames) - 1): # Skip the final frame since there are no frames after it
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
                continue
            #Ensure that input frames are in fp32 - the same dtype as model
            frame_0 = frames[frame_itr:frame_itr+1].to(DEVICE).float()
            frame_1 = frames[frame_itr+1:frame_itr+2].to(DEVICE).float()
            relust = inference(model, frame_0, frame_1, multipliers[frame_itr] - 1)
            output_frames.extend([frame.detach().cpu().to(dtype=torch.float32) for frame in relust[:-1]])

            number_of_frames_processed_since_last_cleared_cuda_cache += 1
            # Try to avoid a memory overflow by clearing cuda cache regularly
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                print("Comfy-VFI: Clearing cache...", end = ' ')
                soft_empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0
                print("Done cache clearing")
            gc.collect()
        
        output_frames.append(frames[-1:].to(dtype=torch.float32)) # Append final frame
        output_frames = [frame.cpu() for frame in output_frames] #Ensure all frames are in cpu
        out = torch.cat(output_frames, dim=0)
        # clear cache for courtesy
        print("Comfy-VFI: Final clearing cache...", end = ' ')
        soft_empty_cache()
        print("Done cache clearing")
        return (postprocess_frames(out), )
