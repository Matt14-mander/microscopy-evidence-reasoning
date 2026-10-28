"""Optional Transformers adapter for decoder-only chat VLMs."""

import time

from src.utils import image_digest


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
        kwargs = {"revision": config.get("revision", "main"), "trust_remote_code": False,
                  "local_files_only": config.get("local_files_only", False)}
        dtype_name = config.get("dtype", "auto")
        dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        processor_options = {key: config[key] for key in ("min_pixels", "max_pixels") if key in config}
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(config["name"], **processor_options, **kwargs)
        model_options = {"attn_implementation": config.get("attn_implementation", "sdpa")}
        if config.get("quantization") == "nf4":
            from transformers import BitsAndBytesConfig
            model_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if dtype == "auto" else dtype,
            )
        self.model = AutoModelForImageTextToText.from_pretrained(
            config["name"], device_map=config.get("device_map", "auto"),
            torch_dtype=dtype, **kwargs, **model_options,
        ).eval()
        if self.model.config.is_encoder_decoder:
            raise ValueError("This adapter supports decoder-only chat VLMs")
        self.load_report = {
            "load_seconds": time.perf_counter() - started,
            "model_name": config["name"], "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_map": {str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
            "dtype": str(self.model.dtype), "quantization": config.get("quantization", "none"),
            "model_memory_bytes": self.model.get_memory_footprint(),
            "processor": {key: getattr(self.processor.image_processor, key, None)
                          for key in ("min_pixels", "max_pixels", "size", "patch_size", "merge_size")},
        }

    def generate(self, request):
        from transformers import set_seed

        set_seed(request.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        content = [{"type": "image"} for _ in request.images]
        content.append({"type": "text", "text": request.prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        kwargs = {"text": [text], "return_tensors": "pt"}
        if request.images:
            kwargs["images"] = list(request.images)
        inputs = self.processor(**kwargs).to(self.model.device)
        grid = inputs.get("image_grid_thw")
        if grid is not None and len(grid) != len(request.images):
            raise ValueError("Processor image count does not match model request")
        temperature = self.inference.get("temperature", 0)
        generation = {"max_new_tokens": self.inference.get("max_new_tokens", 256),
                      "do_sample": temperature > 0}
        if temperature > 0:
            generation["temperature"] = temperature
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        generated = output[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.last_generation = {
            "seconds": time.perf_counter() - started, "n_images": len(request.images),
            "image_sizes": [list(image.size) for image in request.images],
            "image_hashes": [image_digest(image) for image in request.images],
            "image_grid_thw": grid.detach().cpu().tolist() if grid is not None else [],
            "input_tokens": inputs["input_ids"].shape[1], "generated_tokens": generated.shape[1],
            "hit_token_limit": generated.shape[1] >= generation["max_new_tokens"],
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated() if self.torch.cuda.is_available() else None,
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved() if self.torch.cuda.is_available() else None,
        }
        return raw
