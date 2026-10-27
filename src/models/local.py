"""Optional Transformers adapter for decoder-only chat VLMs."""


class LocalVLM:
    def __init__(self, config, inference):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise ImportError('Install the local extra: pip install -e ".[local]"') from error
        self.torch = torch
        self.inference = inference
        kwargs = {"revision": config.get("revision", "main"), "trust_remote_code": False}
        self.processor = AutoProcessor.from_pretrained(config["name"], **kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config["name"], device_map=config.get("device_map", "auto"),
            torch_dtype=config.get("dtype", "auto"), **kwargs,
        ).eval()
        if self.model.config.is_encoder_decoder:
            raise ValueError("This adapter supports decoder-only chat VLMs")

    def generate(self, request):
        from transformers import set_seed

        set_seed(request.seed)
        content = [{"type": "image"} for _ in request.images]
        content.append({"type": "text", "text": request.prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        kwargs = {"text": [text], "return_tensors": "pt"}
        if request.images:
            kwargs["images"] = list(request.images)
        inputs = self.processor(**kwargs).to(self.model.device)
        temperature = self.inference.get("temperature", 0)
        generation = {"max_new_tokens": self.inference.get("max_new_tokens", 256),
                      "do_sample": temperature > 0}
        if temperature > 0:
            generation["temperature"] = temperature
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]
