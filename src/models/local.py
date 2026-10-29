"""Optional Transformers adapter for decoder-only chat VLMs."""

import time
from pathlib import Path

from src.utils import image_digest
from src.models.preprocessing import grid_report, processor_report, tensor_digest, validate_processor_config
from src.models.protocols import text_sha256


class LocalVLM:
    def __init__(self, config, inference):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise ImportError('Install the local extra: pip install -e ".[local]"') from error
        self.torch = torch
        self.inference = inference
        self.config = config
        self.last_generation = None
        model_source = config["name"]
        resolved_revision = None
        if config.get("local_files_only", False) and not Path(model_source).is_dir():
            from huggingface_hub import snapshot_download
            # Pass the pinned snapshot directory to every nested loader. Some processor
            # sub-loaders otherwise probe main despite the outer revision/local-only arguments.
            model_source = snapshot_download(model_source, revision=config.get("revision", "main"), local_files_only=True)
            resolved_revision = Path(model_source).name
        kwargs = {"revision": config.get("revision", "main"), "trust_remote_code": False,
                  "local_files_only": config.get("local_files_only", False)}
        dtype_name = config.get("dtype", "auto")
        dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        processor_options = {key: config[key] for key in ("min_pixels", "max_pixels") if key in config}
        if "processor" in config:
            contract = config["processor"]
            validate_processor_config(contract)
            processor_options.update(contract["parameters"])
            processor_options["use_fast"] = contract["use_fast"]
            processor_options["size"] = {"shortest_edge": config["min_pixels"], "longest_edge": config["max_pixels"]}
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_source, **processor_options, **kwargs)
        self.preprocessing = processor_report(self.processor, config)
        model_options = {"attn_implementation": config.get("attn_implementation", "sdpa")}
        if config.get("quantization") == "nf4":
            from transformers import BitsAndBytesConfig
            model_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if dtype == "auto" else dtype,
            )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_source, device_map=config.get("device_map", "auto"),
            torch_dtype=dtype, **kwargs, **model_options,
        ).eval()
        if self.model.config.is_encoder_decoder:
            raise ValueError("This adapter supports decoder-only chat VLMs")
        self.load_report = {
            "load_seconds": time.perf_counter() - started,
            "model_name": config["name"], "resolved_revision": resolved_revision or getattr(self.model.config, "_commit_hash", None),
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_map": {str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
            "dtype": str(self.model.dtype), "quantization": config.get("quantization", "none"),
            "model_memory_bytes": self.model.get_memory_footprint(),
            "processor": self.preprocessing,
        }

    def prepare_inputs(self, request):
        """CPU preprocessing, preserving request order and recording the actual packed tensor."""
        content = [{"type": "image"} for _ in request.images]
        content.append({"type": "text", "text": request.prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        self.last_generation["rendered_prompt_sha256"] = text_sha256(text)
        self.last_generation["prompt_sha256"] = text_sha256(request.prompt)
        kwargs = {"text": [text], "return_tensors": "pt", "padding": False, "truncation": False}
        if request.images:
            kwargs["images"] = list(request.images)
            if "processor" in self.config:
                kwargs["images_kwargs"] = {"device": "cpu"}
        inputs = self.processor(**kwargs)
        grid = inputs.get("image_grid_thw")
        grids = grid.detach().cpu().tolist() if grid is not None else []
        if (grid is not None or "processor" in self.config) and len(grids) != len(request.images):
            raise ValueError("Processor image count does not match model request")
        self.last_generation.update(image_grid_thw=grids, input_tokens=inputs["input_ids"].shape[1])
        if "processor" in self.config:
            count = int((inputs["input_ids"] == self.processor.image_token_id).sum())
            self.last_generation.update(grid_report(grids, len(request.images), self.config, count))
            pixels = inputs.get("pixel_values")
            expected_rows = sum(entry["patches"] for entry in self.last_generation["images"])
            if (pixels is None and expected_rows) or (pixels is not None and len(pixels) != expected_rows):
                raise ValueError("Packed pixel tensor does not match image grids")
            offset = 0
            for entry in self.last_generation["images"]:
                end = offset + entry["patches"]
                entry["pixel_values_sha256"] = tensor_digest(pixels[offset:end])
                offset = end
        return inputs

    def _memory(self):
        if not self.torch.cuda.is_available():
            return {"peak_allocated_bytes": None, "peak_reserved_bytes": None, "cuda_memory_by_device": {}}
        devices = {}
        for i in range(self.torch.cuda.device_count()):
            devices[str(i)] = {"peak_allocated_bytes": self.torch.cuda.max_memory_allocated(i),
                               "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(i)}
        return {"cuda_memory_by_device": devices,
                "peak_allocated_bytes": sum(d["peak_allocated_bytes"] for d in devices.values()),
                "peak_reserved_bytes": sum(d["peak_reserved_bytes"] for d in devices.values())}

    def _synchronize(self):
        for i in range(self.torch.cuda.device_count()):
            self.torch.cuda.synchronize(i)

    def generate(self, request):
        from transformers import set_seed

        # Reset before any fallible work so errors cannot inherit the previous request's telemetry.
        self.last_generation = {
            "status": "running", "stage": "initialization", "sample_id": request.sample_id,
            "processor_fingerprint": self.preprocessing["fingerprint"],
            "n_images": len(request.images), "image_sizes": [list(im.size) for im in request.images],
            "image_hashes": [image_digest(im) for im in request.images],
            "image_modes": [im.mode for im in request.images],
        }
        started = time.perf_counter()
        try:
            set_seed(request.seed)
            self._synchronize()
            for i in range(self.torch.cuda.device_count()):
                self.torch.cuda.reset_peak_memory_stats(i)
            self.last_generation["stage"] = "preprocessing"
            phase = time.perf_counter()
            inputs = self.prepare_inputs(request)
            self.last_generation["preprocessing_seconds"] = time.perf_counter() - phase
            self.last_generation["stage"] = "device_transfer"
            phase = time.perf_counter()
            inputs = inputs.to(self.model.device)
            self._synchronize()
            self.last_generation["transfer_seconds"] = time.perf_counter() - phase
            self.last_generation["stage"] = "generation"
            phase = time.perf_counter()
            raw = self._generate(inputs)
            self._synchronize()
            self.last_generation["generation_seconds"] = time.perf_counter() - phase
            self.last_generation.update(status="ok", stage="complete")
            return raw
        except Exception as error:
            self.last_generation.update(status="error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            self.last_generation["seconds"] = time.perf_counter() - started
            self.last_generation.update(self._memory())

    def _generate(self, inputs):
        temperature = self.inference.get("temperature", 0)
        generation = {"max_new_tokens": self.inference.get("max_new_tokens", 256),
                      "do_sample": temperature > 0}
        if temperature > 0:
            generation["temperature"] = temperature
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        generated = output[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        self.last_generation.update(generated_tokens=generated.shape[1],
                                    hit_token_limit=generated.shape[1] >= generation["max_new_tokens"])
        return raw
