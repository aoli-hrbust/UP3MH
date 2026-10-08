import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHashHead(nn.Module):
    """Predict a hash-space Gaussian with a bounded diagonal variance."""

    def __init__(self, input_dim, nbit, variance_min, variance_max):
        super().__init__()
        self.variance_min = float(variance_min)
        self.variance_max = float(variance_max)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(input_dim, nbit)
        self.variance_logit = nn.Linear(input_dim, nbit)
        nn.init.zeros_(self.variance_logit.weight)
        nn.init.zeros_(self.variance_logit.bias)

    def forward(self, feature):
        hidden = self.shared(feature)
        mean = torch.tanh(self.mean(hidden))
        variance = self.variance_min + (
            self.variance_max - self.variance_min
        ) * torch.sigmoid(self.variance_logit(hidden))
        return mean, variance


class BidirectionalPrototypePrompt(nn.Module):
    """Prompt each modality with routed prototype mean and variance."""

    def __init__(self, feature_dim, nbit, gate_init=-1.5):
        super().__init__()
        # encoded feature + routed mean + normalized routed variance
        input_dim = feature_dim + 2 * nbit
        self.image_to_text = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim * 2),
        )
        self.text_to_image = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim * 2),
        )
        self.image_to_text_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.text_to_image_gate = nn.Parameter(torch.tensor(float(gate_init)))
        for branch in (self.image_to_text, self.text_to_image):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    @staticmethod
    def _inject(target, parameters, gate, strength):
        scale, shift = parameters.chunk(2, dim=-1)
        delta = 0.5 * torch.tanh(scale) * target + torch.tanh(shift)
        return target + strength * torch.sigmoid(gate) * delta, delta

    def forward(
        self,
        image_feature,
        text_feature,
        image_dynamic_prototype,
        text_dynamic_prototype,
        image_dynamic_variance_condition,
        text_dynamic_variance_condition,
        strength,
        enable_image_to_text=True,
        enable_text_to_image=True,
    ):
        image_context = F.layer_norm(image_feature, (image_feature.shape[-1],))
        text_context = F.layer_norm(text_feature, (text_feature.shape[-1],))

        if enable_image_to_text:
            parameters = self.image_to_text(
                torch.cat(
                    [
                        image_context,
                        F.normalize(image_dynamic_prototype, dim=-1),
                        image_dynamic_variance_condition,
                    ],
                    dim=-1,
                )
            )
            prompted_text, image_to_text_delta = self._inject(
                text_feature, parameters, self.image_to_text_gate, strength
            )
        else:
            prompted_text = text_feature
            image_to_text_delta = torch.zeros_like(text_feature)

        if enable_text_to_image:
            parameters = self.text_to_image(
                torch.cat(
                    [
                        text_context,
                        F.normalize(text_dynamic_prototype, dim=-1),
                        text_dynamic_variance_condition,
                    ],
                    dim=-1,
                )
            )
            prompted_image, text_to_image_delta = self._inject(
                image_feature, parameters, self.text_to_image_gate, strength
            )
        else:
            prompted_image = image_feature
            text_to_image_delta = torch.zeros_like(image_feature)

        return {
            "image_feature": prompted_image,
            "text_feature": prompted_text,
            "image_to_text_delta": image_to_text_delta,
            "text_to_image_delta": text_to_image_delta,
        }


class DimensionGatedFusion(nn.Module):
    """Fuse modalities with a learnable reliability gate for each dimension."""

    def __init__(self, feature_dim, nbit, dropout=0.1):
        super().__init__()
        interaction_dim = feature_dim * 4
        self.gate = nn.Sequential(
            nn.Linear(interaction_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Linear(interaction_dim + feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hash = nn.Sequential(nn.Linear(feature_dim, nbit), nn.Tanh())

    def forward(self, image_feature, text_feature):
        interaction = torch.cat(
            [
                image_feature,
                text_feature,
                torch.abs(image_feature - text_feature),
                image_feature * text_feature,
            ],
            dim=-1,
        )
        gate = self.gate(interaction)
        mixed = gate * image_feature + (1.0 - gate) * text_feature
        fused = mixed + self.refine(torch.cat([interaction, mixed], dim=-1))
        return fused, self.hash(fused), gate


class MultiBranchHashRefiner(nn.Module):
    """Refine the hash with modality, prototype, and variance evidence."""

    def __init__(self, nbit, dropout=0.1, gate_init=-2.0):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(nbit * 7, nbit * 2),
            nn.LayerNorm(nbit * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nbit * 2, nbit),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(
        self,
        base_code,
        image_mean,
        text_mean,
        dynamic_prototype,
        dynamic_variance_condition,
        strength,
    ):
        interaction = torch.cat(
            [
                base_code,
                image_mean,
                text_mean,
                torch.abs(image_mean - text_mean),
                image_mean * text_mean,
                dynamic_prototype,
                dynamic_variance_condition,
            ],
            dim=-1,
        )
        delta = torch.tanh(self.refine(interaction))
        code = torch.tanh(
            base_code + strength * torch.sigmoid(self.gate) * delta
        )
        return code, delta


class PCH(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_dim = int(args.image_dim)
        self.text_dim = int(args.text_dim)
        self.common_dim = int(args.common_dim)
        self.nbit = int(args.nbit)
        self.classes = int(args.classes)
        self.K = int(args.K)
        if self.K not in (1, 2, 4, 6, 8, 10, 12):
            raise ValueError("Prototype sensitivity supports K in {1,2,4,6,8,10,12}.")
        self.route_topk = min(int(args.route_topk), self.classes * self.K)
        self.proto_temperature = float(args.proto_temperature)
        self.prototype_variance_min = float(args.prototype_variance_min)
        self.prototype_variance_max = float(args.prototype_variance_max)
        self.w2_mean_weight = float(args.prototype_w2_mean_weight)
        self.w2_variance_weight = float(args.prototype_w2_variance_weight)
        self.enable_proto_interaction = bool(args.enable_proto_interaction)
        self.enable_image_to_text_prompt = bool(args.enable_image_to_text_prompt)
        self.enable_text_to_image_prompt = bool(args.enable_text_to_image_prompt)
        self.disable_router_proto = bool(args.disable_router_proto)
        self.input_layernorm = bool(args.input_layernorm)
        self.prototype_ema_momentum = float(args.prototype_ema_momentum)
        self.enable_hash_refiner = bool(args.enable_hash_refiner)

        self.image_encoder = nn.Sequential(
            nn.Linear(self.image_dim, self.common_dim),
            nn.LayerNorm(self.common_dim),
            nn.GELU(),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.common_dim),
            nn.LayerNorm(self.common_dim),
            nn.GELU(),
        )
        self.image_distribution = GaussianHashHead(
            self.common_dim,
            self.nbit,
            self.prototype_variance_min,
            self.prototype_variance_max,
        )
        self.text_distribution = GaussianHashHead(
            self.common_dim,
            self.nbit,
            self.prototype_variance_min,
            self.prototype_variance_max,
        )
        # Compact bank: [class, 2 sub-centers, hash bits]. It is never stacked
        # per sample, avoiding the V25 proto/proto_bank dimensional explosion.
        self.prototype_mean = nn.Parameter(
            F.normalize(torch.randn(self.classes, self.K, self.nbit), dim=-1)
        )
        self.prototype_logvar = nn.Parameter(
            torch.zeros(self.classes, self.K, self.nbit)
        )
        self.register_buffer(
            "prototype_ema_mean", self.prototype_mean.detach().clone()
        )
        self.register_buffer(
            "prototype_ema_variance",
            torch.full(
                (self.classes, self.K, self.nbit),
                (self.prototype_variance_min + self.prototype_variance_max) / 2.0,
            ),
        )
        self.register_buffer("prototype_ema_initialized", torch.tensor(False))
        self.register_buffer("prototype_interaction_strength", torch.tensor(1.0))
        # Retained for checkpoint/result compatibility with V25.
        self.register_buffer("prototype_distance_scale", torch.tensor(1.0))

        self.proto_prompt = BidirectionalPrototypePrompt(
            self.common_dim,
            self.nbit,
            gate_init=float(args.interaction_gate_init),
        )
        self.fusion = DimensionGatedFusion(
            self.common_dim, self.nbit, dropout=float(args.dropout)
        )
        self.hash_refiner = MultiBranchHashRefiner(
            self.nbit,
            dropout=float(args.dropout),
            gate_init=float(args.hash_refine_gate_init),
        )
        self.hash_classifier = nn.Linear(self.nbit, self.classes)

    def get_prototype_variance(self):
        return self.prototype_variance_min + (
            self.prototype_variance_max - self.prototype_variance_min
        ) * torch.sigmoid(self.prototype_logvar)

    def normalize_variance_condition(self, variance):
        """Map bounded variance to [0, 1] for downstream conditioning."""
        variance_range = max(
            self.prototype_variance_max - self.prototype_variance_min,
            1e-12,
        )
        return (
            (variance - self.prototype_variance_min) / variance_range
        ).clamp(0.0, 1.0)




    def _encode_modalities(self, image, text):
        if self.input_layernorm:
            image = F.layer_norm(image, (self.image_dim,))
            text = F.layer_norm(text, (self.text_dim,))
        return self.image_encoder(image), self.text_encoder(text)


    def _routing_prototypes(self):
        if not self.training and bool(self.prototype_ema_initialized):
            return self.prototype_ema_mean, self.prototype_ema_variance
        return F.normalize(self.prototype_mean, dim=-1), self.get_prototype_variance()

    def route_distribution(
        self, sample_mean, sample_variance, oracle_label=None
    ):
        prototype_mean, prototype_variance = self._routing_prototypes()
        sample_mean = F.normalize(sample_mean, dim=-1)
        mean_distance = (
            sample_mean[:, None, None, :] - prototype_mean[None]
        ).square().mean(dim=-1)
        std_distance = (
            sample_variance.clamp_min(1e-6).sqrt()[:, None, None, :]
            - prototype_variance.clamp_min(1e-6).sqrt()[None]
        ).square().mean(dim=-1)
        distance = (
            self.w2_mean_weight * mean_distance
            + self.w2_variance_weight * std_distance
        )
        logits = -distance / self.proto_temperature
        flat_logits = logits.flatten(start_dim=1)

        # Oracle routing is deliberately isolated from the normal forward path.
        # It is an evaluation-only diagnostic: ground-truth positive classes
        # restrict the candidate prototypes, while feature similarity still
        # chooses/weights their sub-centers.
        if oracle_label is not None:
            if self.training:
                raise ValueError("Oracle routing is evaluation-only.")
            positive_class = oracle_label.to(
                device=flat_logits.device, dtype=torch.bool
            )
            if positive_class.ndim != 2 or positive_class.shape[1] != self.classes:
                raise ValueError(
                    "oracle_label must have shape [batch_size, classes]."
                )
            positive_center = positive_class[:, :, None].expand(
                -1, -1, self.K
            ).reshape_as(flat_logits)
            has_positive = positive_center.any(dim=1, keepdim=True)
            positive_center = torch.where(
                has_positive, positive_center, torch.ones_like(positive_center)
            )
            oracle_logits = flat_logits.masked_fill(
                ~positive_center, float("-inf")
            )
            flat_weight = F.softmax(oracle_logits, dim=1)
        # Soft routing keeps all Cx2 centers differentiable during training.
        # Label-free evaluation uses similarity-ranked sparse selection.
        elif self.training:
            flat_weight = F.softmax(flat_logits, dim=1)
        else:
            top_value, top_index = torch.topk(
                flat_logits, k=self.route_topk, dim=1
            )
            top_weight = F.softmax(top_value, dim=1)
            flat_weight = torch.zeros_like(flat_logits)
            flat_weight.scatter_(1, top_index, top_weight)
        weight = flat_weight.view(-1, self.classes, self.K)

        dynamic_mean = torch.einsum("bck,ckd->bd", weight, prototype_mean)
        dynamic_second = torch.einsum(
            "bck,ckd->bd",
            weight,
            prototype_variance + prototype_mean.square(),
        )
        dynamic_variance = (
            dynamic_second - dynamic_mean.square()
        ).clamp(self.prototype_variance_min, self.prototype_variance_max)
        class_score = torch.logsumexp(logits, dim=2)
        class_weight = weight.sum(dim=2)
        confidence = F.softmax(class_score, dim=1).amax(dim=1, keepdim=True)
        return {
            "logits": logits,
            "class_score": class_score,
            "class_weight": class_weight,
            "subcenter_weight": weight,
            "dynamic_prototype": dynamic_mean,
            "dynamic_prototype_variance": dynamic_variance,
            "distance": distance,
            "confidence": confidence,
        }

    def forward(self, image, text, label=None, oracle_label=None):
        del label
        image_feature, text_feature = self._encode_modalities(image, text)
        image_mean, image_variance = self.image_distribution(image_feature)
        text_mean, text_variance = self.text_distribution(text_feature)
        image_route = self.route_distribution(
            image_mean, image_variance, oracle_label=oracle_label
        )
        text_route = self.route_distribution(
            text_mean, text_variance, oracle_label=oracle_label
        )

        prototype_active = (
            not self.disable_router_proto and self.enable_proto_interaction
        )
        if prototype_active:
            # Confidence is detached only in the amplitude gate; route logits
            # still receive direct gradients from L_clu and through prototypes.
            mean_confidence = 0.5 * (
                image_route["confidence"] + text_route["confidence"]
            )
            # Keep a non-vanishing prompt path early in training; confidence
            # modulates its amplitude without being able to shut it off.
            prompt_strength = self.prototype_interaction_strength * (
                0.5 + 0.5 * mean_confidence
            ).detach()
            image_variance_condition = self.normalize_variance_condition(
                image_route["dynamic_prototype_variance"]
            )
            text_variance_condition = self.normalize_variance_condition(
                text_route["dynamic_prototype_variance"]
            )
            prompted = self.proto_prompt(
                image_feature,
                text_feature,
                image_route["dynamic_prototype"],
                text_route["dynamic_prototype"],
                image_variance_condition,
                text_variance_condition,
                prompt_strength,
                self.enable_image_to_text_prompt,
                self.enable_text_to_image_prompt,
            )
        else:
            prompted = {
                "image_feature": image_feature,
                "text_feature": text_feature,
                "image_to_text_delta": torch.zeros_like(text_feature),
                "text_to_image_delta": torch.zeros_like(image_feature),
            }

        fused_feature, base_code, fusion_gate = self.fusion(
            prompted["image_feature"], prompted["text_feature"]
        )
        probability_mean = 0.5 * (image_mean + text_mean)
        probability_variance = 0.5 * (
            image_variance + text_variance
        ) + 0.25 * (image_mean - text_mean).square()
        route_weight = 0.5 * (
            image_route["class_weight"] + text_route["class_weight"]
        )
        dynamic_prototype = 0.5 * (
            image_route["dynamic_prototype"]
            + text_route["dynamic_prototype"]
        )
        dynamic_variance = 0.5 * (
            image_route["dynamic_prototype_variance"]
            + text_route["dynamic_prototype_variance"]
        )
        dynamic_variance_condition = self.normalize_variance_condition(
            dynamic_variance
        )
        if self.enable_hash_refiner:
            code, hash_refine_delta = self.hash_refiner(
                base_code,
                image_mean,
                text_mean,
                dynamic_prototype,
                dynamic_variance_condition,
                self.prototype_interaction_strength,
            )
        else:
            code = base_code
            hash_refine_delta = torch.zeros_like(base_code)
        hash_logits = self.hash_classifier(code)

        return {
            "code": code,
            "hash_logits": hash_logits,
            "hash_refine_delta": hash_refine_delta,
            "base_code": base_code,
            "coarse_base_code": base_code,
            "fused_code": fused_feature,
            "fusion_gate": fusion_gate,
            "image_mean": image_mean,
            "image_variance": image_variance,
            "text_mean": text_mean,
            "text_variance": text_variance,
            "probability_mean": probability_mean,
            "probability_variance": probability_variance,
            "routing_probability_variance": probability_variance,
            "image_route_logits": image_route["logits"],
            "text_route_logits": text_route["logits"],
            "image_class_score": image_route["class_score"],
            "text_class_score": text_route["class_score"],
            "combined_router_logits": 0.5 * (
                image_route["class_score"] + text_route["class_score"]
            ),
            "router_logits": 0.5 * (
                image_route["class_score"] + text_route["class_score"]
            ),
            "route_weight": route_weight,
            "subcenter_weight": 0.5 * (
                image_route["subcenter_weight"]
                + text_route["subcenter_weight"]
            ),
            "dynamic_prototype": dynamic_prototype,
            "dynamic_prototype_variance": dynamic_variance,
            "dynamic_variance_condition": dynamic_variance_condition,
            "image_dynamic_prototype": image_route["dynamic_prototype"],
            "text_dynamic_prototype": text_route["dynamic_prototype"],
            "prototype_mean": self.prototype_mean,
            "prototype_variance": self.get_prototype_variance(),
            "route_confidence": 0.5 * (
                image_route["confidence"] + text_route["confidence"]
            ),
            "probability_route_confidence": 0.5 * (
                image_route["confidence"] + text_route["confidence"]
            ),
            "dynamic_w2_distance": 0.5 * (
                image_route["distance"].mean(dim=(1, 2), keepdim=True)
                + text_route["distance"].mean(dim=(1, 2), keepdim=True)
            ),
            "calibrated_probability_distance": 0.5 * (
                image_route["distance"].mean(dim=(1, 2), keepdim=True)
                + text_route["distance"].mean(dim=(1, 2), keepdim=True)
            ),
            "prompted_image_feature": prompted["image_feature"],
            "prompted_text_feature": prompted["text_feature"],
            "image_to_text_prompt": prompted["image_to_text_delta"],
            "text_to_image_prompt": prompted["text_to_image_delta"],
            "prototype_interaction_strength": self.prototype_interaction_strength,
        }
