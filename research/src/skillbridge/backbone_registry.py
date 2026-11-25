from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Literal

BackboneRole = Literal[
    "primary",
    "compatibility",
    "external_validation",
    "attachment_feasibility",
    "post_training_reference",
]
BackboneRuntime = Literal["native", "compatibility", "attachment_only", "reference_only"]
RouteTiming = Literal["per_flow_step", "discrete_step", "cached_conditioner", "none"]


@dataclass(frozen=True, slots=True)
class BackboneProfile:
    name: str
    role: BackboneRole
    runtime: BackboneRuntime
    architecture: str
    repository_url: str
    pinned_commit: str
    transformer_path: str | None
    route_timing: RouteTiming
    required_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BackboneRepositoryReport:
    name: str
    repository: Path
    expected_commit: str
    actual_commit: str | None
    missing_files: tuple[str, ...]
    commit_matches: bool

    @property
    def ready(self) -> bool:
        return not self.missing_files and self.commit_matches


_PROFILES = {
    "janusflow": BackboneProfile(
        name="janusflow",
        role="primary",
        runtime="native",
        architecture="shared_ar_rectified_flow",
        repository_url="https://github.com/deepseek-ai/Janus.git",
        pinned_commit="1daa72fa409002d40931bd7b36a9280362469ead",
        transformer_path="language_model.model",
        route_timing="per_flow_step",
        required_files=(
            "janus/janusflow/models/modeling_vlm.py",
            "janus/janusflow/models/uvit.py",
        ),
    ),
    "dimoo": BackboneProfile(
        name="dimoo",
        role="compatibility",
        runtime="compatibility",
        architecture="shared_ar_discrete_diffusion",
        repository_url="https://github.com/Alpha-VLLM/Lumina-DiMOO.git",
        pinned_commit="81344e60545ae8d70f11039d3450717d974f831b",
        transformer_path="model.transformer",
        route_timing="discrete_step",
        required_files=(),
    ),
    "blip3o": BackboneProfile(
        name="blip3o",
        role="compatibility",
        runtime="compatibility",
        architecture="qwen_semantic_queries_plus_diffusion",
        repository_url="https://github.com/JiuhaiChen/BLIP3o.git",
        pinned_commit="798b206c07c2425269931bbf1f1e5b81875eedbb",
        transformer_path="model.layers",
        route_timing="cached_conditioner",
        required_files=(
            "blip3o/model/language_model/blip3o_qwen.py",
            "blip3o/model/language_model/blip3o_qwen_inference.py",
            "blip3o/model/multimodal_decoder/builder.py",
        ),
    ),
    "uniworld_v1": BackboneProfile(
        name="uniworld_v1",
        role="compatibility",
        runtime="compatibility",
        architecture="ar_conditioner_plus_flux_flow",
        repository_url="https://github.com/PKU-YuanGroup/UniWorld-V1.git",
        pinned_commit="2124cd6845c78dd2a9cd5a535e45f8a17e626a63",
        transformer_path="model",
        route_timing="cached_conditioner",
        required_files=(
            "UniWorld-V1/univa/models/qwen2p5vl/modeling_univa_qwen2p5vl.py",
            "UniWorld-V1/univa/models/modeling_univa_denoise_tower.py",
            "UniWorld-V1/train_denoiser.py",
        ),
    ),
    "edit_r1": BackboneProfile(
        name="edit_r1",
        role="post_training_reference",
        runtime="reference_only",
        architecture="flow_policy_optimization_with_mllm_reward",
        repository_url="https://github.com/PKU-YuanGroup/Edit-R1.git",
        pinned_commit="4217f251a5abd3ca9ef2f9b4eee851a14f744f84",
        transformer_path=None,
        route_timing="none",
        required_files=(
            "scripts/train_nft_qwen_image_edit.py",
            "reward_server/reward_server.py",
        ),
    ),
}


def backbone_profile(name: str) -> BackboneProfile:
    try:
        return _PROFILES[name]
    except KeyError as error:
        msg = f"unknown backbone profile: {name}"
        raise ValueError(msg) from error


def require_executable_backbone(name: str) -> BackboneProfile:
    profile = backbone_profile(name)
    if profile.runtime not in ("native", "compatibility"):
        label = profile.runtime.replace("_", "-")
        msg = f"{name} is {label}; this repository does not expose an end-to-end runtime"
        raise NotImplementedError(msg)
    return profile


def validate_backbone_repository(
    name: str,
    repository: Path,
    *,
    verify_commit: bool = True,
) -> BackboneRepositoryReport:
    profile = backbone_profile(name)
    repo = repository.resolve()
    missing = tuple(path for path in profile.required_files if not (repo / path).is_file())
    actual_commit: str | None = None
    commit_matches = True
    if verify_commit:
        git_binary = which("git")
        if git_binary is None:
            msg = "git executable is required for revision verification"
            raise FileNotFoundError(msg)
        result = subprocess.run(  # noqa: S603
            (git_binary, "-C", str(repo), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            actual_commit = result.stdout.strip()
        commit_matches = bool(profile.pinned_commit) and actual_commit == profile.pinned_commit
    return BackboneRepositoryReport(
        name=name,
        repository=repo,
        expected_commit=profile.pinned_commit,
        actual_commit=actual_commit,
        missing_files=missing,
        commit_matches=commit_matches,
    )
