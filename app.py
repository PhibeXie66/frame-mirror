from __future__ import annotations

import base64
import json
import math
import os
import random
import re
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from scipy import ndimage
from sentence_transformers import SentenceTransformer


APP_DIR = Path(__file__).resolve().parent
for env_candidate in (
    APP_DIR / ".env",
    APP_DIR.parent / "Cloudflare.env",
    Path(r"E:\简历\movie_agent\Cloudflare.env"),
):
    if env_candidate.is_file():
        load_dotenv(env_candidate, override=False)

STATIC_DIR = APP_DIR / "static"
DATASET_DIR = Path(
    os.getenv(
        "POSTER_DATASET_DIR",
        r"E:\简历\movie_agent\impawards_posters",
    )
)
IMAGE_DIR = DATASET_DIR / "images"
ANALYSIS_DIR = DATASET_DIR / "clip_analysis"
MANIFEST_PATH = ANALYSIS_DIR / "clip_manifest.csv"
EMBEDDINGS_PATH = ANALYSIS_DIR / "poster_embeddings.npy"
PROGRESS_PATH = ANALYSIS_DIR / "embedding_progress.npy"
TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
UPLOAD_DIR = Path(os.getenv("PRODUCT_UPLOAD_DIR", str(DATASET_DIR / "product_uploads")))
GENERATION_DIR = Path(os.getenv("GENERATION_DIR", str(DATASET_DIR / "generated_ads")))
JOB_STATE_DIR = GENERATION_DIR / "job_state"
MOTION_DIR = GENERATION_DIR / "motion"
REFERENCE_UPLOAD_DIR = Path(
    os.getenv("REFERENCE_UPLOAD_DIR", str(DATASET_DIR / "agent_references"))
)
PRODUCT_CUTOUT_DIR = Path(
    os.getenv("PRODUCT_CUTOUT_DIR", str(DATASET_DIR / "product_cutouts_sam"))
)
SAM_MODEL_NAME = os.getenv("SAM_MODEL_NAME", "sam_b.pt")
SAM_DEVICE = os.getenv("SAM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
SAM_CUTOUT_VERSION = "v9-general"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
CLOUDFLARE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-2-klein-4b",
)
CLOUDFLARE_AGENT_MODEL = os.getenv(
    "CLOUDFLARE_AGENT_MODEL",
    "@cf/meta/llama-3.1-8b-instruct",
)

PRODUCT_CATEGORIES = [
    ("香水", "a premium perfume bottle product photograph"),
    ("护肤品", "a luxury skincare cosmetic product photograph"),
    ("彩妆", "a lipstick or makeup cosmetic product photograph"),
    ("手表", "a luxury wristwatch product photograph"),
    ("珠宝首饰", "fine jewelry product photograph"),
    ("手袋", "a designer handbag product photograph"),
    ("鞋履", "fashion sneakers or shoes product photograph"),
    ("服装", "fashion clothing product photograph"),
    ("眼镜", "eyewear or sunglasses product photograph"),
    ("酒类饮品", "a premium wine or spirits bottle product photograph"),
    ("食品包装", "a packaged food product photograph"),
    ("咖啡茶饮", "a coffee or tea beverage product photograph"),
    ("手机数码", "a smartphone consumer electronics product photograph"),
    ("电脑", "a laptop computer product photograph"),
    ("耳机音响", "headphones or audio speaker product photograph"),
    ("相机", "a camera product photograph"),
    ("家居用品", "a furniture or home decor product photograph"),
    ("汽车", "an automobile product photograph"),
]

REFERENCE_CONCEPT_GROUPS = {
    "scene": [
        "a surreal dreamlike world",
        "an open outdoor landscape",
        "a cinematic urban street",
        "a cosmic space environment",
        "a botanical nature environment",
        "a desert landscape",
        "an ocean or underwater world",
        "an abstract graphic environment",
        "an architectural interior",
        "a minimal studio backdrop",
    ],
    "composition": [
        "a centered symmetrical composition",
        "a dynamic diagonal composition",
        "a layered collage composition",
        "a large silhouette framing the subject",
        "a split-panel composition",
        "a radial composition",
        "a dramatic close-up crop",
        "a wide negative-space composition",
    ],
    "motif": [
        "bold geometric shapes",
        "cinematic light beams and glow",
        "smoke and atmospheric haze",
        "floating particles and fragments",
        "reflections and mirror surfaces",
        "paper collage and torn textures",
        "organic flowing forms",
        "dramatic graphic shadows",
        "grainy printed poster texture",
        "neon color blocks",
    ],
}

VARIANT_CREATIVE_ROUTES = [
    "Surreal editorial transformation: express the reference motif through atmosphere, landscape and edge framing while leaving the broad center empty.",
    "Graphic poster construction: use flat patterns, layering and scale contrast around the perimeter; do not introduce a central focal object.",
    "Cinematic environmental narrative: create an open, story-rich setting with depth, clear scenery and an unoccupied central foreground.",
    "Material and light experiment: reinterpret textures, reflections, shadows and color relationships as continuous background surfaces only.",
]

BACKGROUND_OBJECT_BAN = (
    "CRITICAL EMPTY-BACKPLATE RULE: this output is only an unoccupied environmental plate, never a finished product ad. "
    "The central 60 percent must show only continuous scenery, ground, atmosphere, light or abstract background texture and must remain completely empty. "
    "Do not create any central hero object, focal prop, isolated object, display object or object-shaped silhouette. "
    "Absolutely no merchandise, substitute product, package, packaging, box, carton, bottle, jar, tube, can, bag, shoe, clothing, watch, jewelry, cosmetic, "
    "electronic device, furniture, vehicle, sign, statue, plinth, pedestal, container, framed object or freestanding geometric block. "
    "No people, animals, faces, hands, logos, brands, letters or text. All creative motifs must remain environmental, non-object background patterns or edge framing. "
    "If any idea could look like a sellable item, omit it and leave that area empty. The real merchandise is composited later by a separate local layer."
)


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=240)
    top_k: int = Field(default=16, ge=1, le=40)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)


class ProductSpec(BaseModel):
    upload_id: str | None = None
    name: str = Field(default="", max_length=100)
    category: str = Field(default="", max_length=100)
    description: str = Field(default="", max_length=600)


class ReferenceSpec(BaseModel):
    vector_index: int
    title: str = Field(max_length=240)
    year: int | str | None = None
    similarity: float | None = None
    detail_url: str | None = None
    poster_url: str | None = None


class BriefRequest(BaseModel):
    style_query: str = Field(min_length=1, max_length=400)
    product: ProductSpec
    references: list[ReferenceSpec] = Field(min_length=1, max_length=4)


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=20, max_length=2400)
    style_query: str = Field(default="", max_length=400)
    product: ProductSpec
    references: list[ReferenceSpec] = Field(default_factory=list, max_length=4)
    variant_count: int = Field(default=4, ge=1, le=4)
    width: int = Field(default=1024, ge=512, le=1920)
    height: int = Field(default=1365, ge=512, le=1920)


class TypographySpec(BaseModel):
    enabled: bool = False
    headline: str = Field(default="", max_length=100)
    subtitle: str = Field(default="", max_length=160)
    position: str = Field(default="top", pattern="^(top|bottom)$")
    style: str = Field(default="editorial", pattern="^(editorial|modern|minimal)$")
    color: str = Field(default="#F4E7C1", pattern=r"^#[0-9A-Fa-f]{6}$")


class RevisionRequest(BaseModel):
    source_index: int = Field(default=0, ge=0, le=3)
    instruction: str = Field(min_length=2, max_length=3200)
    variant_count: int = Field(default=2, ge=1, le=4)
    typography: TypographySpec | None = None
    product_scale_action: str = Field(default="none", pattern="^(none|larger|smaller)$")


class LayoutRequest(BaseModel):
    source_index: int = Field(default=0, ge=0, le=3)
    scale: float = Field(default=0.62, ge=0.30, le=0.90)
    offset_x: float = Field(default=0.0, ge=-0.35, le=0.35)
    offset_y: float = Field(default=0.0, ge=-0.35, le=0.35)


class MotionRequest(BaseModel):
    source_index: int = Field(default=0, ge=0, le=3)
    duration: int = Field(default=4, ge=3, le=8)
    style: str = Field(default="push", pattern="^(push|pull|drift|breathe)$")
    format: str = Field(default="mp4", pattern="^(mp4|gif)$")
    prompt: str = Field(default="", max_length=1000)


class MotionPlanRequest(BaseModel):
    message: str = Field(min_length=2, max_length=1000)


class AgentMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=1200)


class AgentTurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1200)
    history: list[AgentMessage] = Field(default_factory=list, max_length=16)
    current_prompt: str = Field(default="", max_length=3000)
    product_name: str = Field(default="商品", max_length=120)
    product_category: str = Field(default="", max_length=120)
    reference_id: str | None = Field(default=None, max_length=64)


class PosterSearchEngine:
    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.manifest = pd.DataFrame()
        self.embeddings: np.ndarray | None = None
        self.valid_indices = np.array([], dtype=np.int64)
        self.model: SentenceTransformer | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_lock = threading.Lock()
        self.index_to_row: dict[int, int] = {}

    def load(self) -> None:
        try:
            for path in (MANIFEST_PATH, EMBEDDINGS_PATH, PROGRESS_PATH, IMAGE_DIR):
                if not path.exists():
                    raise FileNotFoundError(f"缺少数据文件：{path}")

            manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
            embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
            progress = np.load(PROGRESS_PATH)

            if len(manifest) != len(embeddings) or len(progress) != len(manifest):
                raise RuntimeError("清单、向量和进度文件的行数不一致。")

            valid = np.flatnonzero(progress.astype(bool))
            if not len(valid):
                raise RuntimeError("没有已完成的 CLIP 图片向量。")

            print(f"Loading multilingual CLIP text model on {self.device}...")
            model = SentenceTransformer(TEXT_MODEL_NAME, device=self.device)

            self.manifest = manifest
            self.embeddings = embeddings
            self.valid_indices = valid
            self.model = model
            self.index_to_row = {
                int(row.vector_index): int(position)
                for position, row in manifest.iterrows()
            }
            self.ready = True
            self.error = None
            print(f"Poster search ready: {len(valid):,} vectors")
        except Exception as exc:
            self.ready = False
            self.error = str(exc)
            print(f"Poster search failed to load: {exc}")

    def status(self) -> dict[str, Any]:
        years = pd.to_numeric(self.manifest.get("year"), errors="coerce")
        finite_years = years.dropna().astype(int) if years is not None else pd.Series(dtype=int)
        return {
            "ready": self.ready,
            "error": self.error,
            "total": int(len(self.valid_indices)),
            "device": self.device,
            "year_min": int(finite_years.min()) if len(finite_years) else None,
            "year_max": int(finite_years.max()) if len(finite_years) else None,
            "dataset": str(DATASET_DIR),
        }

    def encode_query(self, query: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("文本模型尚未载入。")
        prompts = [
            query,
            f"一张{query}风格的电影海报",
            f"电影海报设计，{query}",
        ]
        with self.model_lock:
            vectors = self.model.encode(
                prompts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
        vector = vectors.mean(axis=0)
        vector /= np.linalg.norm(vector) + 1e-12
        return vector

    def search(self, request: SearchRequest) -> dict[str, Any]:
        if not self.ready or self.embeddings is None:
            raise RuntimeError(self.error or "搜索引擎尚未准备完成。")

        started = time.perf_counter()
        query_vector = self.encode_query(request.query.strip())
        candidates = self.valid_indices

        if request.year_from is not None or request.year_to is not None:
            years = pd.to_numeric(self.manifest.iloc[candidates]["year"], errors="coerce").to_numpy()
            mask = np.ones(len(candidates), dtype=bool)
            if request.year_from is not None:
                mask &= years >= request.year_from
            if request.year_to is not None:
                mask &= years <= request.year_to
            candidates = candidates[mask]

        if not len(candidates):
            return {"query": request.query, "elapsed_ms": 0, "total": 0, "items": []}

        matrix = np.asarray(self.embeddings[candidates], dtype="float32")
        scores = matrix @ query_vector
        k = min(request.top_k, len(candidates))
        local = np.argpartition(-scores, k - 1)[:k]
        local = local[np.argsort(-scores[local])]
        positions = candidates[local]

        items: list[dict[str, Any]] = []
        for rank, (position, score) in enumerate(zip(positions, scores[local]), start=1):
            row = self.manifest.iloc[int(position)]
            vector_index = int(row.get("vector_index", position))
            items.append(
                {
                    "rank": rank,
                    "vector_index": vector_index,
                    "similarity": round(float(score), 5),
                    "title": json_value(row.get("title")) or Path(str(row["file_name"])).stem,
                    "year": json_value(row.get("year")),
                    "variant": json_value(row.get("variant")),
                    "width": json_value(row.get("width")),
                    "height": json_value(row.get("height")),
                    "file_name": str(row["file_name"]),
                    "detail_url": json_value(row.get("detail_url")),
                    "poster_url": f"/api/posters/{vector_index}",
                }
            )

        return {
            "query": request.query,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "total": len(items),
            "items": items,
        }

    def poster_path(self, vector_index: int) -> Path:
        if vector_index not in self.index_to_row:
            raise KeyError(vector_index)
        row = self.manifest.iloc[self.index_to_row[vector_index]]
        path = Path(str(row["full_path"])).resolve()
        image_root = IMAGE_DIR.resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise FileNotFoundError(path)
        return path


class ProductAnalyzer:
    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None
        self.model_lock = threading.Lock()
        self.load_lock = threading.Lock()
        self.category_vectors: np.ndarray | None = None

    def ensure_loaded(self) -> None:
        if self.model is not None and self.category_vectors is not None:
            return
        with self.load_lock:
            if self.model is None:
                print(f"Loading CLIP image model on {engine.device}...")
                self.model = SentenceTransformer(IMAGE_MODEL_NAME, device=engine.device)
            if engine.model is None:
                raise RuntimeError("中文 CLIP 文本模型尚未载入。")
            prompts = [prompt for _, prompt in PRODUCT_CATEGORIES]
            with engine.model_lock:
                self.category_vectors = engine.model.encode(
                    prompts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype("float32")

    def analyze(self, image_path: Path) -> list[dict[str, Any]]:
        self.ensure_loaded()
        if self.model is None or self.category_vectors is None:
            raise RuntimeError("商品识别模型尚未准备完成。")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            with self.model_lock:
                vector = self.model.encode(
                    image,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype("float32")
        scores = self.category_vectors @ vector
        order = np.argsort(-scores)[:3]
        return [
            {
                "category": PRODUCT_CATEGORIES[int(index)][0],
                "confidence": round(float(scores[int(index)]), 4),
            }
            for index in order
        ]

    def analyze_reference_language(self, image_paths: list[Path]) -> dict[str, list[str]]:
        if not image_paths:
            return {}
        self.ensure_loaded()
        if self.model is None or engine.model is None:
            raise RuntimeError("CLIP 视觉语言分析模型尚未准备完成。")

        images: list[Image.Image] = []
        for path in image_paths[:4]:
            with Image.open(path) as source:
                preview = source.convert("RGB")
                preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
                images.append(preview.copy())
        with self.model_lock:
            image_vectors = self.model.encode(
                images,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")

        labels = [
            label
            for group_labels in REFERENCE_CONCEPT_GROUPS.values()
            for label in group_labels
        ]
        with engine.model_lock:
            text_vectors = engine.model.encode(
                labels,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
        similarities = text_vectors @ image_vectors.T
        combined_scores = similarities.max(axis=1) * 0.7 + similarities.mean(axis=1) * 0.3

        result: dict[str, list[str]] = {}
        offset = 0
        for group_name, group_labels in REFERENCE_CONCEPT_GROUPS.items():
            group_scores = combined_scores[offset : offset + len(group_labels)]
            top_indices = np.argsort(-group_scores)[:2]
            result[group_name] = [group_labels[int(index)] for index in top_indices]
            offset += len(group_labels)
        return result


class CloudflareImageGenerator:
    def __init__(self) -> None:
        self.provider = os.getenv("IMAGE_PROVIDER", "").strip().lower()
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        self.model = CLOUDFLARE_MODEL

    def configuration_problem(self) -> str | None:
        if self.provider != "cloudflare":
            return "IMAGE_PROVIDER 尚未设置为 cloudflare。"
        if not self.account_id or "@" in self.account_id:
            return "CLOUDFLARE_ACCOUNT_ID 缺失或仍是邮箱；请填写 Cloudflare 控制台中的 Account ID。"
        if not self.token:
            return "CLOUDFLARE_API_TOKEN 尚未配置。"
        return None

    def status(self) -> dict[str, Any]:
        problem = self.configuration_problem()
        return {
            "image_generation": problem is None,
            "provider": "cloudflare",
            "model": self.model,
            "configured": problem is None,
            "message": problem or "Cloudflare 图片生成引擎已配置。",
            "video_generation": False,
        }

    @staticmethod
    def prepare_reference(path: Path) -> bytes:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((511, 511), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, "JPEG", quality=91, optimize=True)
            return buffer.getvalue()

    def generate(
        self,
        prompt: str,
        input_paths: list[Path],
        output_path: Path,
        width: int,
        height: int,
        seed: int,
    ) -> None:
        problem = self.configuration_problem()
        if problem:
            raise RuntimeError(problem)

        endpoint = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model}"
        )
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, path in enumerate(input_paths[:4]):
            files.append(
                (
                    f"input_image_{index}",
                    (f"reference-{index}.jpg", self.prepare_reference(path), "image/jpeg"),
                )
            )
        data = {
            "prompt": prompt,
            "width": str(width),
            "height": str(height),
            "seed": str(seed),
            "guidance": "4.0",
        }
        try:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.token}"},
                    data=data,
                    files=files,
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"无法连接 Cloudflare 图片服务：{exc}") from exc

        if response.status_code >= 400:
            message = f"Cloudflare 返回 HTTP {response.status_code}。"
            try:
                payload = response.json()
                errors = payload.get("errors") or []
                if errors and errors[0].get("message"):
                    message = str(errors[0]["message"])
            except ValueError:
                pass
            raise RuntimeError(message)

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = response.json()
            result = payload.get("result") or {}
            encoded = result.get("image")
            if not encoded:
                raise RuntimeError("Cloudflare 响应中没有生成图片。")
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            raw = base64.b64decode(encoded)
        else:
            raw = response.content

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(BytesIO(raw)) as generated:
                generated.convert("RGB").save(output_path, "JPEG", quality=95, optimize=True)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError("图片服务返回了无法识别的文件。") from exc


class CloudflareCreativeAgent:
    def __init__(self) -> None:
        self.provider = os.getenv("IMAGE_PROVIDER", "").strip().lower()
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        self.model = CLOUDFLARE_AGENT_MODEL

    def configuration_problem(self) -> str | None:
        if self.provider != "cloudflare":
            return "IMAGE_PROVIDER 尚未设置为 cloudflare。"
        if not self.account_id or "@" in self.account_id:
            return "CLOUDFLARE_ACCOUNT_ID 配置无效。"
        if not self.token:
            return "CLOUDFLARE_API_TOKEN 尚未配置。"
        return None

    @staticmethod
    def response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["clarify", "ready"]},
                "assistant_message": {"type": "string"},
                "questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                "generation_prompt": {"type": "string"},
                "product_scale_action": {
                    "type": "string",
                    "enum": ["none", "larger", "smaller"],
                },
                "reference_usage": {"type": "string"},
                "typography": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "headline": {"type": "string"},
                        "subtitle": {"type": "string"},
                        "position": {"type": "string", "enum": ["top", "bottom"]},
                        "style": {
                            "type": "string",
                            "enum": ["editorial", "modern", "minimal"],
                        },
                        "color": {"type": "string"},
                    },
                    "required": [
                        "enabled",
                        "headline",
                        "subtitle",
                        "position",
                        "style",
                        "color",
                    ],
                },
            },
            "required": [
                "status",
                "assistant_message",
                "questions",
                "generation_prompt",
                "product_scale_action",
                "reference_usage",
                "typography",
            ],
        }

    @staticmethod
    def typography_from_request(text: str) -> dict[str, Any]:
        headline = ""
        headline_patterns = (
            r"(?:写着|写上|写|加上文字|标题(?:为|是)?|大标题(?:为|是)?)[：:\s]*[“\"「『]?([^”\"」』，。！？\n]{2,60})",
            r"[“\"「『]([^”\"」』\n]{2,60})[”\"」』](?:作为|做成)?(?:大标题|标题)",
        )
        for pattern in headline_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                headline = match.group(1).strip(" ：:，,。.!！?？")
                break

        subtitle = ""
        subtitle_match = re.search(
            r"(?:副标题|小标题)(?:为|是|写着|写上)?[：:\s]*[“\"「『]?([^”\"」』，。！？\n]{2,80})",
            text,
            flags=re.IGNORECASE,
        )
        if subtitle_match:
            subtitle = subtitle_match.group(1).strip(" ：:，,。.!！?？")

        position = "bottom" if any(word in text for word in ("底部", "下方", "正下方")) else "top"
        style = "editorial"
        if any(word in text.lower() for word in ("现代", "无衬线", "modern", "sans")):
            style = "modern"
        elif any(word in text.lower() for word in ("极简", "简洁", "minimal")):
            style = "minimal"

        color = "#F4E7C1"
        color_rules = (
            (("白色", "白字", "纯白"), "#FFFFFF"),
            (("蓝色", "蓝字"), "#9EB8FF"),
            (("红色", "红字"), "#E45745"),
            (("黑色", "黑字"), "#111111"),
            (("金色", "金字", "香槟金"), "#E8C77A"),
        )
        for words, value in color_rules:
            if any(word in text for word in words):
                color = value
                break

        return {
            "enabled": bool(headline or subtitle),
            "headline": headline,
            "subtitle": subtitle,
            "position": position,
            "style": style,
            "color": color,
        }

    @staticmethod
    def fallback_plan(request: AgentTurnRequest, reference_summary: str) -> dict[str, Any]:
        user_context = " ".join(
            [
                message.content
                for message in request.history[-12:]
                if message.role == "user"
            ]
            + [request.message]
        )
        typography = CloudflareCreativeAgent.typography_from_request(user_context)
        outdoor_words = ("室外", "户外", "outside", "outdoor")
        outdoor_details = (
            "半室外",
            "窗户",
            "露台",
            "街道",
            "庭院",
            "屋顶",
            "海边",
            "森林",
            "城市",
            "完整室外",
        )
        if any(word in request.message.lower() for word in outdoor_words) and not any(
            word in user_context for word in outdoor_details
        ):
            return {
                "status": "clarify",
                "assistant_message": "可以把黑色室内改成更有呼吸感的室外环境。为了避免我理解错，我想先确认室外与光线的具体程度。",
                "questions": [
                    "你想要完整室外、半室外露台，还是通过大窗户看到室外？",
                    "希望是白天自然光、日落金光，还是夜晚城市灯光？",
                ],
                "generation_prompt": "",
                "product_scale_action": "none",
                "reference_usage": reference_summary,
                "typography": typography,
            }
        if request.reference_id and not any(
            word in request.message for word in ("色彩", "光线", "构图", "材质", "建筑", "元素", "氛围")
        ):
            return {
                "status": "clarify",
                "assistant_message": "参考图已经收到。为了避免直接照搬，我需要知道你真正想借用哪一种视觉语言。",
                "questions": ["你想参考它的色彩、光线、构图、空间材质，还是某个具体元素？"],
                "generation_prompt": "",
                "product_scale_action": "none",
                "reference_usage": reference_summary,
                "typography": typography,
            }
        scale_action = "none"
        if any(word in user_context for word in ("放大", "大一点", "larger", "bigger")):
            scale_action = "larger"
        elif any(word in user_context for word in ("缩小", "小一点", "smaller")):
            scale_action = "smaller"
        reference_clause = f" Reference analysis: {reference_summary}." if reference_summary else ""
        prompt = (
            f"Revise the advertising background according to this confirmed creative direction: {user_context}."
            f"{reference_clause} Preserve the original product pixels, silhouette, proportions, labels, logos and "
            "material details exactly. Do not redraw, deform, duplicate or remove the product. Change only the "
            "background, environmental lighting and scene styling. Maintain a premium cinematic commercial layout, "
            "clear subject separation and intentional negative space. Do not add readable text or watermarks."
        )
        if typography["enabled"]:
            prompt += (
                f" Reserve clean {typography['position']} negative space for a locally typeset headline. "
                "Do not render the headline or any other text inside the generated image."
            )
        return {
            "status": "ready",
            "assistant_message": "信息已经足够。我整理了一版可编辑的生成方案，请确认后再生成。",
            "questions": [],
            "generation_prompt": prompt,
            "product_scale_action": scale_action,
            "reference_usage": reference_summary,
            "typography": typography,
        }

    def turn(self, request: AgentTurnRequest, reference_summary: str = "") -> dict[str, Any]:
        fallback = self.fallback_plan(request, reference_summary)
        if self.configuration_problem():
            return {**fallback, "engine": "local-fallback"}

        system_prompt = (
            "你是高级商业广告的 Creative Director Agent。你的任务不是立即生成图片，而是通过对话澄清用户真正想改变的视觉内容。"
            "当用户描述有关键歧义时，status 必须是 clarify，并只问 1-3 个最有价值的问题；不要问已经回答过的问题。"
            "当环境、时间/光线和核心元素已经足够明确时，status 必须是 ready，并输出一段可以直接交给图片生成模型的英文 generation_prompt。"
            "永远保护商品：商品原图像素、轮廓、比例、品牌标签和材质不得重绘、变形、复制或删除。只允许背景、环境光和氛围变化；"
            "商品只可通过 product_scale_action 指示等比例放大或缩小。参考图只能提取用户明确指定的色彩、光线、构图或元素，不得要求复制人物、Logo或完整画面。"
            "如果用户要求海报写字，必须把用户指定的标题和副标题逐字保存在 typography 中，不得改写、翻译或润色；"
            "generation_prompt 只要求在 typography.position 预留干净留白，并明确禁止图片模型生成任何文字。文字将由本地排版层精确添加。"
            "assistant_message 和 questions 使用简体中文，generation_prompt 使用英文。"
        )
        context = {
            "product_name": request.product_name,
            "product_category": request.product_category,
            "current_prompt": request.current_prompt,
            "reference_visual_analysis": reference_summary,
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": "当前创作上下文：" + json.dumps(context, ensure_ascii=False)},
        ]
        messages.extend(
            {"role": message.role, "content": message.content}
            for message in request.history[-12:]
        )
        messages.append({"role": "user", "content": request.message})
        endpoint = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model}"
        )
        payload = {
            "messages": messages,
            "max_tokens": 1100,
            "temperature": 0.35,
            "response_format": {
                "type": "json_schema",
                "json_schema": self.response_schema(),
            },
        }
        try:
            with httpx.Client(timeout=90.0) as client:
                response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            if response.status_code >= 400:
                return {**fallback, "engine": "local-fallback"}
            result = (response.json().get("result") or {}).get("response")
            if isinstance(result, str):
                result = json.loads(result)
            if not isinstance(result, dict) or result.get("status") not in {"clarify", "ready"}:
                return {**fallback, "engine": "local-fallback"}
            result["questions"] = [str(item) for item in result.get("questions", [])[:3]]
            model_typography = result.get("typography")
            if fallback["typography"]["enabled"]:
                result["typography"] = fallback["typography"]
            else:
                try:
                    result["typography"] = TypographySpec.model_validate(
                        model_typography or {}
                    ).model_dump()
                except ValueError:
                    result["typography"] = fallback["typography"]
            if (
                result.get("status") == "clarify"
                and fallback.get("status") == "ready"
                and fallback.get("typography", {}).get("enabled")
            ):
                fallback["assistant_message"] = (
                    "文字内容、位置和视觉气质已经明确。我已保留标题原文，并整理成可确认的精确排版方案。"
                )
                return {**fallback, "engine": "cloudflare-agent-typography-guardrail"}
            if (
                result.get("status") == "clarify"
                and fallback.get("status") == "ready"
                and len(request.history) >= 2
            ):
                fallback["assistant_message"] = (
                    "空间、光线和核心元素已经明确。我整理了一版可编辑的生成方案，请确认后再生成。"
                )
                return {**fallback, "engine": "cloudflare-agent-guardrail"}
            result["engine"] = "cloudflare-agent"
            return result
        except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError):
            return {**fallback, "engine": "local-fallback"}


def extract_reference_palette(paths: list[Path], color_count: int = 6) -> list[str]:
    swatches: list[tuple[int, int, int]] = []
    for path in paths:
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((240, 240), Image.Resampling.LANCZOS)
                quantized = image.quantize(colors=color_count, method=Image.Quantize.MEDIANCUT)
                palette = quantized.getpalette() or []
                colors = quantized.getcolors() or []
                for _, index in sorted(colors, reverse=True)[:color_count]:
                    offset = index * 3
                    rgb = tuple(palette[offset : offset + 3])
                    if len(rgb) == 3:
                        swatches.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
        except (UnidentifiedImageError, OSError):
            continue

    unique: list[tuple[int, int, int]] = []
    for color in swatches:
        if all(sum(abs(a - b) for a, b in zip(color, existing)) > 70 for existing in unique):
            unique.append(color)
        if len(unique) >= color_count:
            break
    return [f"#{red:02X}{green:02X}{blue:02X}" for red, green, blue in unique]


_SAM_MODEL: Any | None = None
_SAM_MODEL_LOCK = threading.Lock()
_MATTING_SESSION: Any | None = None
_MATTING_LOCK = threading.Lock()


def threshold_product_cutout(path: Path) -> Image.Image:
    """Fallback for environments where SAM cannot be loaded."""
    with Image.open(path) as source:
        if "A" in source.getbands():
            rgba_source = source.convert("RGBA")
            alpha_extrema = rgba_source.getchannel("A").getextrema()
            if alpha_extrema[0] < 250:
                bbox = rgba_source.getbbox()
                return rgba_source.crop(bbox) if bbox else rgba_source
        image = source.convert("RGB")
    pixels = np.asarray(image, dtype=np.float32)
    distance_from_white = np.sqrt(np.sum((255 - pixels) ** 2, axis=2))
    brightness = pixels.mean(axis=2)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    foreground = (distance_from_white > 30) & ((chroma > 28) | (brightness < 175))
    foreground = ndimage.binary_opening(foreground, iterations=1)
    foreground = ndimage.binary_closing(foreground, iterations=2)
    labels, count = ndimage.label(foreground)
    if count:
        sizes = ndimage.sum(foreground, labels, range(1, count + 1))
        largest = int(np.argmax(sizes)) + 1
        foreground = labels == largest
    foreground = ndimage.binary_fill_holes(foreground)
    foreground = ndimage.binary_erosion(foreground, iterations=1)
    alpha = ndimage.gaussian_filter(foreground.astype(np.float32), sigma=0.75)
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    bbox = rgba.getbbox()
    if not bbox:
        return image.convert("RGBA")
    return rgba.crop(bbox)


def rough_product_box(image: Image.Image) -> list[int]:
    """Estimate a loose product box from the border background for SAM prompting."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    border_size = max(2, min(height, width) // 40)
    border = np.concatenate(
        (
            rgb[:border_size].reshape(-1, 3),
            rgb[-border_size:].reshape(-1, 3),
            rgb[:, :border_size].reshape(-1, 3),
            rgb[:, -border_size:].reshape(-1, 3),
        ),
        axis=0,
    )
    background_color = np.median(border, axis=0)
    distance = np.sqrt(np.sum((rgb - background_color) ** 2, axis=2))
    foreground = distance > 22
    foreground = ndimage.binary_opening(foreground, iterations=1)
    foreground = ndimage.binary_closing(foreground, iterations=2)

    labels, count = ndimage.label(foreground)
    selected = np.zeros_like(foreground, dtype=bool)
    selected_area = 0
    min_component = max(12, int(height * width * 0.00035))
    center_x = width / 2
    center_y = height / 2
    if count:
        objects = ndimage.find_objects(labels)
        for label_index, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            component = labels == label_index
            area = int(component.sum())
            if area < min_component:
                continue
            y_slice, x_slice = slices
            component_x = (x_slice.start + x_slice.stop) / 2
            component_y = (y_slice.start + y_slice.stop) / 2
            near_subject = (
                abs(component_x - center_x) < width * 0.42
                and abs(component_y - center_y) < height * 0.46
            )
            if near_subject and area > selected_area:
                selected = component
                selected_area = area

    coordinates = np.argwhere(selected if selected.any() else foreground)
    if not len(coordinates):
        return [0, 0, width - 1, height - 1]
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    pad_x = max(4, int((x_max - x_min + 1) * 0.12))
    pad_y = max(4, int((y_max - y_min + 1) * 0.14))
    return [
        max(0, int(x_min) - pad_x),
        max(0, int(y_min) - pad_y),
        min(width - 1, int(x_max) + pad_x),
        min(height - 1, int(y_max) + pad_y),
    ]


def get_sam_model() -> Any:
    global _SAM_MODEL
    if _SAM_MODEL is None:
        with _SAM_MODEL_LOCK:
            if _SAM_MODEL is None:
                from ultralytics import SAM

                _SAM_MODEL = SAM(SAM_MODEL_NAME)
    return _SAM_MODEL


def get_matting_session() -> Any:
    global _MATTING_SESSION
    if _MATTING_SESSION is None:
        with _MATTING_LOCK:
            if _MATTING_SESSION is None:
                from rembg import new_session

                _MATTING_SESSION = new_session("u2net")
    return _MATTING_SESSION


def sam_guided_box(path: Path, image: Image.Image) -> list[int]:
    """Use a rough border box as the prompt, then tighten it with SAM."""
    rough_box = rough_product_box(image)
    model = get_sam_model()
    with _SAM_MODEL_LOCK:
        results = model(
            str(path),
            bboxes=[rough_box],
            device=SAM_DEVICE,
            verbose=False,
        )
    if not results or results[0].masks is None or not len(results[0].masks.data):
        return rough_box
    mask = results[0].masks.data[0].detach().float().cpu().numpy()
    mask_image = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), mode="L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.BILINEAR)
    coordinates = np.argwhere(np.asarray(mask_image) > 96)
    if not len(coordinates):
        return rough_box
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    pad_x = max(4, int((x_max - x_min + 1) * 0.04))
    pad_y = max(4, int((y_max - y_min + 1) * 0.04))
    width, height = image.size
    return [
        max(rough_box[0], int(x_min) - pad_x),
        max(rough_box[1], int(y_min) - pad_y),
        min(rough_box[2], int(x_max) + pad_x, width - 1),
        min(rough_box[3], int(y_max) + pad_y, height - 1),
    ]


def hybrid_product_cutout(path: Path, category: str = "") -> Image.Image:
    """SAM locates the product; U2Net produces a cleaner product alpha matte."""
    PRODUCT_CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"{path.stem}-{SAM_CUTOUT_VERSION}-"
        f"{SAM_MODEL_NAME.replace('.', '_')}.png"
    )
    cache_path = PRODUCT_CUTOUT_DIR / cache_name
    if cache_path.is_file() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        with Image.open(cache_path) as cached:
            return cached.convert("RGBA")

    with Image.open(path) as source:
        image = source.convert("RGB")
    box = sam_guided_box(path, image)
    crop = image.crop((box[0], box[1], box[2] + 1, box[3] + 1))

    from rembg import remove

    matting_session = get_matting_session()
    with _MATTING_LOCK:
        result = remove(crop, session=matting_session, alpha_matting=False).convert("RGBA")

    bbox = result.getbbox()
    if not bbox:
        raise RuntimeError("透明背景模型返回了空蒙版。")
    result = result.crop(bbox)
    result.save(cache_path, "PNG", optimize=True)
    return result


def sam_product_cutout(path: Path) -> Image.Image:
    PRODUCT_CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = f"{path.stem}-{SAM_CUTOUT_VERSION}-{SAM_MODEL_NAME.replace('.', '_')}.png"
    cache_path = PRODUCT_CUTOUT_DIR / cache_name
    if cache_path.is_file() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        with Image.open(cache_path) as cached:
            return cached.convert("RGBA")

    with Image.open(path) as source:
        image = source.convert("RGB")
    box = rough_product_box(image)
    model = get_sam_model()
    with _SAM_MODEL_LOCK:
        results = model(
            str(path),
            bboxes=[box],
            device=SAM_DEVICE,
            verbose=False,
        )
    if not results or results[0].masks is None or not len(results[0].masks.data):
        raise RuntimeError("SAM 没有返回商品蒙版。")

    masks = results[0].masks.data.detach().float().cpu().numpy()
    mask = masks[np.argmax(masks.reshape(masks.shape[0], -1).sum(axis=1))]
    mask_image = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), mode="L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.BILINEAR)
    mask_array = np.asarray(mask_image, dtype=np.float32) / 255.0
    mask_binary = ndimage.binary_closing(mask_array > 0.45, iterations=2)
    mask_binary = ndimage.binary_fill_holes(mask_binary)
    silhouette_alpha = ndimage.gaussian_filter(mask_binary.astype(np.float32), sigma=0.75)

    # SAM extracts the silhouette, but transparent glass still contains the old
    # background color. Estimate per-pixel opacity from the border background,
    # then reverse the white compositing to remove halos and white-filled glass.
    rgb = np.asarray(image, dtype=np.float32)
    height, width = rgb.shape[:2]
    border_size = max(2, min(height, width) // 40)
    border = np.concatenate(
        (
            rgb[:border_size].reshape(-1, 3),
            rgb[-border_size:].reshape(-1, 3),
            rgb[:, :border_size].reshape(-1, 3),
            rgb[:, -border_size:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(border, axis=0)
    color_distance = np.sqrt(np.sum((rgb - background) ** 2, axis=2))
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    darkness = np.maximum(0, background.mean() - rgb.mean(axis=2))
    material_alpha = np.maximum.reduce(
        (
            np.clip((color_distance - 12) / 62, 0, 1),
            np.clip((chroma - 5) / 58, 0, 1),
            np.clip((darkness - 8) / 72, 0, 1),
        )
    )
    material_alpha = ndimage.gaussian_filter(material_alpha, sigma=0.45)

    # Preserve the opaque cap/neck. Light metallic caps are close to white and
    # would otherwise be mistaken for transparent glass. The central upper
    # component avoids detached labels or decorations near the image edges.
    x1, y1, x2, y2 = box
    box_width = x2 - x1 + 1
    box_height = y2 - y1 + 1
    center_x = (x1 + x2) / 2
    yy, xx = np.ogrid[:height, :width]
    upper_center = (
        (xx >= center_x - box_width * 0.29)
        & (xx <= center_x + box_width * 0.29)
        & (yy >= y1)
        & (yy <= y1 + box_height * 0.34)
    )
    cap_candidates = upper_center & (
        (color_distance > 16) | (chroma > 9) | (darkness > 11)
    )
    cap_candidates = ndimage.binary_closing(cap_candidates, iterations=3)
    cap_labels, cap_count = ndimage.label(cap_candidates)
    cap_mask = np.zeros_like(cap_candidates)
    if cap_count:
        cap_sizes = ndimage.sum(cap_candidates, cap_labels, range(1, cap_count + 1))
        cap_mask = cap_labels == (int(np.argmax(cap_sizes)) + 1)
        cap_mask = ndimage.binary_fill_holes(cap_mask)
        cap_mask = ndimage.binary_dilation(cap_mask, iterations=1)
    material_alpha = np.maximum(material_alpha, cap_mask.astype(np.float32) * 0.96)
    final_alpha = np.clip(silhouette_alpha * material_alpha, 0, 1)
    final_alpha[final_alpha < 0.12] = 0

    safe_alpha = np.maximum(final_alpha[..., None], 0.06)
    foreground_rgb = background + (rgb - background) / safe_alpha
    foreground_rgb = np.clip(foreground_rgb, 0, 255).astype(np.uint8)
    rgba_array = np.dstack(
        (foreground_rgb, np.clip(final_alpha * 255, 0, 255).astype(np.uint8))
    )
    rgba = Image.fromarray(rgba_array, mode="RGBA")
    bbox = rgba.getbbox()
    if not bbox:
        raise RuntimeError("SAM 返回了空蒙版。")
    cutout = rgba.crop(bbox)
    cutout.save(cache_path, "PNG", optimize=True)
    return cutout


def product_cutout(path: Path, category: str = "") -> Image.Image:
    with Image.open(path) as source:
        if "A" in source.getbands():
            rgba_source = source.convert("RGBA")
            if rgba_source.getchannel("A").getextrema()[0] < 250:
                bbox = rgba_source.getbbox()
                return rgba_source.crop(bbox) if bbox else rgba_source
    try:
        return hybrid_product_cutout(path, category)
    except Exception as exc:
        print(f"SAM + matting 扣图失败，尝试 SAM 单模型：{exc}")
        try:
            return sam_product_cutout(path)
        except Exception as sam_exc:
            print(f"SAM 扣图失败，已回退到基础算法：{sam_exc}")
            return threshold_product_cutout(path)


def load_poster_font(style: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = {
        "editorial": (
            Path(r"C:\Windows\Fonts\STZHONGS.TTF"),
            Path(r"C:\Windows\Fonts\simsun.ttc"),
        ),
        "modern": (
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
        ),
        "minimal": (
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simfang.ttf"),
        ),
    }
    for font_path in font_candidates.get(style, font_candidates["editorial"]):
        if font_path.is_file():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def fit_poster_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    style: str,
    max_width: int,
    start_size: int,
    minimum_size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for size in range(start_size, minimum_size - 1, -2):
        font = load_poster_font(style, size)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=max(1, size // 55))
        if bbox[2] - bbox[0] <= max_width:
            return font
    return load_poster_font(style, minimum_size)


def apply_typography(
    image: Image.Image,
    typography: TypographySpec | dict[str, Any] | None,
) -> Image.Image:
    if not typography:
        return image.convert("RGBA")
    if isinstance(typography, dict):
        try:
            spec = TypographySpec.model_validate(typography)
        except ValueError:
            return image.convert("RGBA")
    else:
        spec = typography
    if not spec.enabled or not (spec.headline.strip() or spec.subtitle.strip()):
        return image.convert("RGBA")

    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = spec.color
    max_width = int(canvas.width * 0.86)
    headline = spec.headline.strip()
    subtitle = spec.subtitle.strip()
    headline_font = fit_poster_font(
        draw,
        headline,
        spec.style,
        max_width,
        start_size=max(54, int(canvas.width * 0.105)),
        minimum_size=max(30, int(canvas.width * 0.042)),
    )
    headline_bbox = draw.textbbox((0, 0), headline, font=headline_font, stroke_width=2)
    headline_height = headline_bbox[3] - headline_bbox[1]
    subtitle_font = load_poster_font("modern", max(20, int(canvas.width * 0.025)))
    subtitle_height = 0
    if subtitle:
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font, stroke_width=1)
        subtitle_height = subtitle_bbox[3] - subtitle_bbox[1]

    block_height = headline_height + (int(canvas.height * 0.018) + subtitle_height if subtitle else 0)
    if spec.position == "bottom":
        top = int(canvas.height * 0.92) - block_height
    else:
        top = int(canvas.height * 0.055)
    top = max(int(canvas.height * 0.025), min(top, canvas.height - block_height - 20))

    # Image generators often imitate poster lettering even when instructed not to.
    # Clean the local title zone first so exact user text never competes with fake glyphs.
    band_padding = int(canvas.height * 0.025)
    band_top = max(0, top - band_padding)
    band_bottom = min(canvas.height, top + block_height + band_padding * 2)
    if band_bottom > band_top:
        title_band = canvas.crop((0, band_top, canvas.width, band_bottom))
        title_band = title_band.filter(ImageFilter.GaussianBlur(max(8, canvas.width // 70)))
        shade = Image.new("RGBA", title_band.size, (3, 7, 14, 155))
        title_band = Image.alpha_composite(title_band, shade)
        canvas.paste(title_band, (0, band_top))

    center_x = canvas.width // 2
    stroke_width = max(1, canvas.width // 420)
    draw.text(
        (center_x, top),
        headline,
        font=headline_font,
        fill=color,
        anchor="ma",
        align="center",
        stroke_width=stroke_width,
        stroke_fill=(8, 12, 20, 185),
    )
    if subtitle:
        subtitle_y = top + headline_height + int(canvas.height * 0.018)
        draw.text(
            (center_x, subtitle_y),
            subtitle,
            font=subtitle_font,
            fill=color,
            anchor="ma",
            align="center",
            stroke_width=1,
            stroke_fill=(8, 12, 20, 170),
        )
    return Image.alpha_composite(canvas, overlay)


def render_product_composite(
    background_path: Path,
    product_path: Path,
    product_scale: float = 0.62,
    product_category: str = "",
    typography: TypographySpec | dict[str, Any] | None = None,
    product_offset_x: float = 0.0,
    product_offset_y: float = 0.0,
) -> Image.Image:
    with Image.open(background_path) as source:
        background = source.convert("RGBA")
    cutout = product_cutout(product_path, product_category)
    max_width = int(background.width * product_scale)
    max_height = int(background.height * 0.72)
    ratio = min(max_width / cutout.width, max_height / cutout.height)
    new_size = (max(1, int(cutout.width * ratio)), max(1, int(cutout.height * ratio)))
    cutout = cutout.resize(new_size, Image.Resampling.LANCZOS)

    x = (background.width - cutout.width) // 2 + int(background.width * product_offset_x)
    y = int(background.height * 0.86) - cutout.height + int(background.height * product_offset_y)
    x = max(0, min(background.width - cutout.width, x))
    y = max(0, min(background.height - cutout.height, y))

    alpha = cutout.getchannel("A")
    shadow = Image.new("RGBA", background.size, (0, 0, 0, 0))
    shadow_mask = Image.new("L", background.size, 0)
    shadow_mask.paste(alpha, (x + 18, y + 24))
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(22))
    shadow_layer = Image.new("RGBA", background.size, (0, 0, 0, 150))
    shadow_layer.putalpha(shadow_mask.point(lambda value: int(value * 0.55)))
    shadow.alpha_composite(shadow_layer)
    background.alpha_composite(shadow)
    background.alpha_composite(cutout, (x, y))
    return apply_typography(background, typography).convert("RGB")


def composite_product(
    background_path: Path,
    product_path: Path,
    output_path: Path,
    product_scale: float = 0.62,
    product_category: str = "",
    typography: TypographySpec | dict[str, Any] | None = None,
    product_offset_x: float = 0.0,
    product_offset_y: float = 0.0,
) -> None:
    background = render_product_composite(
        background_path=background_path,
        product_path=product_path,
        product_scale=product_scale,
        product_category=product_category,
        typography=typography,
        product_offset_x=product_offset_x,
        product_offset_y=product_offset_y,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(output_path, "JPEG", quality=95, optimize=True)


def creative_palette(query: str) -> list[str]:
    palette = ["曜石黑", "电光蓝", "冰川白"]
    rules = [
        (("金", "奢华", "复古"), "香槟金"),
        (("红", "戏剧", "热烈"), "深绯红"),
        (("绿", "自然", "森林"), "苔藓绿"),
        (("紫", "霓虹", "未来"), "紫外光"),
        (("粉", "柔和", "浪漫"), "雾粉"),
    ]
    for keywords, color in rules:
        if any(keyword in query for keyword in keywords):
            palette.append(color)
    return palette[:4]


def background_only_instruction(instruction: str) -> str:
    cleaned = re.sub(
        r"Product scale action\s*:\s*(?:none|larger|smaller)\s*\.?",
        "",
        instruction,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:把|让|将|保持)?(?:商品|主体物品|主体商品|产品|鞋子|香水瓶).{0,18}?"
        r"(?:放大|缩小|大一点|小一点|往上|往下|向上|向下|向左|向右|移动|调整位置).{0,18}?"
        r"(?=[，。,.!！?？；;\n]|$)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:在|于)?(?:上面|顶部|下方|底部|画面上方|画面下方).{0,10}?"
        r"(?:写|写着|添加文字|加入标题).{0,80}?(?=[。.!！?？\n]|$)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ，,。.")


def product_negative_terms(product: ProductSpec) -> str:
    product_text = f"{product.category} {product.name}".lower()
    rules = (
        (("鞋", "sneaker", "shoe", "footwear"), "shoes, sneakers or footwear"),
        (("香水", "perfume"), "perfume bottles or cosmetic bottles"),
        (("手表", "watch"), "watches or watch-like objects"),
        (("包", "handbag", "bag"), "bags or handbags"),
        (("手机", "phone"), "phones or electronic devices"),
        (("汽车", "car"), "cars or vehicles"),
    )
    for keywords, phrase in rules:
        if any(keyword in product_text for keyword in keywords):
            return phrase
    return "the subject product or any product-like object"


def background_safe_concepts(values: list[str], fallback: str) -> list[str]:
    """Remove reference concepts that encourage the model to invent a central subject."""
    unsafe = ("silhouette", "close-up", "portrait", "character", "person", "face", "subject")
    safe = [value for value in values if not any(word in value.lower() for word in unsafe)]
    return safe or [fallback]


def build_brief(request: BriefRequest) -> dict[str, Any]:
    product_name = request.product.name.strip() or request.product.category.strip() or "主体商品"
    category = request.product.category.strip() or "待确认商品"
    reference_titles = [item.title for item in request.references]
    reference_line = "、".join(reference_titles)
    description = request.product.description.strip()
    prompt = (
        f"为{product_name}创作一张高级商业广告海报。商品必须保持真实外形、品牌结构和材质细节，"
        f"作为唯一清晰主体；整体视觉方向：{request.style_query.strip()}。"
        f"参考电影海报《{reference_line}》的构图秩序、色彩关系、光影气氛和字体节奏，"
        "但不要复制电影人物、片名、商标或受版权保护的具体画面。"
        "画面具有电影级布光、明确视觉焦点、充足广告文案留白、精致材质与高端杂志质感。"
    )
    if description:
        prompt += f"商品补充信息：{description}。"
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "product": {
            "upload_id": request.product.upload_id,
            "name": product_name,
            "category": category,
            "description": description,
        },
        "direction": request.style_query.strip(),
        "palette": creative_palette(request.style_query),
        "references": [item.model_dump() for item in request.references],
        "composition": [
            "商品占据主视觉，轮廓与标签清晰可辨",
            "提取参考海报的构图、色彩、光影与排版节奏",
            "保留品牌标题、卖点与行动文案的安全留白",
            "仅借鉴视觉语言，不复制电影人物与具体版权元素",
        ],
        "generation_prompt": prompt,
        "generation_ready": False,
        "generation_note": "创意方案已完成；连接图片生成引擎后即可生成最终海报。",
    }


engine = PosterSearchEngine()
product_analyzer = ProductAnalyzer()
image_generator = CloudflareImageGenerator()
creative_agent = CloudflareCreativeAgent()
generation_jobs: dict[str, dict[str, Any]] = {}
generation_jobs_lock = threading.Lock()
motion_jobs: dict[str, dict[str, Any]] = {}
motion_jobs_lock = threading.Lock()


def uploaded_product_path(upload_id: str | None) -> Path:
    if not upload_id or len(upload_id) != 32 or any(char not in "0123456789abcdef" for char in upload_id):
        raise FileNotFoundError("Invalid product upload id")
    for suffix in (".png", ".jpg"):
        path = (UPLOAD_DIR / f"{upload_id}{suffix}").resolve()
        if path.is_relative_to(UPLOAD_DIR.resolve()) and path.is_file():
            return path
    raise FileNotFoundError(upload_id)


def latest_uploaded_product() -> Path | None:
    if not UPLOAD_DIR.is_dir():
        return None
    candidates = [
        path
        for path in UPLOAD_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg"}
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def agent_reference_path(reference_id: str | None) -> Path:
    if not reference_id or not re.fullmatch(r"[0-9a-f]{32}", reference_id):
        raise FileNotFoundError("Invalid agent reference id")
    path = (REFERENCE_UPLOAD_DIR / f"{reference_id}.jpg").resolve()
    if not path.is_relative_to(REFERENCE_UPLOAD_DIR.resolve()) or not path.is_file():
        raise FileNotFoundError(reference_id)
    return path


def reference_visual_summary(path: Path) -> dict[str, Any]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        preview = image.copy()
        preview.thumbnail((480, 480), Image.Resampling.LANCZOS)
    rgb = np.asarray(preview, dtype=np.float32)
    hsv = np.asarray(preview.convert("HSV"), dtype=np.float32)
    brightness = float(rgb.mean() / 255)
    saturation = float(hsv[..., 1].mean() / 255)
    orientation = "vertical" if height > width * 1.12 else "horizontal" if width > height * 1.12 else "square"
    light_label = "bright" if brightness > 0.68 else "dark" if brightness < 0.34 else "mid-tone"
    saturation_label = "vivid" if saturation > 0.48 else "muted" if saturation < 0.22 else "balanced"
    palette = extract_reference_palette([path], color_count=5)
    return {
        "orientation": orientation,
        "lighting": light_label,
        "saturation": saturation_label,
        "palette": palette,
        "summary": (
            f"{orientation} composition, {light_label} lighting, {saturation_label} color, "
            f"dominant palette {', '.join(palette) if palette else 'not available'}"
        ),
    }


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in job.items() if not key.startswith("_")}
    editable_indices: list[int] = []
    generation_root = GENERATION_DIR.resolve()
    for index, raw_path in enumerate((job.get("_context") or {}).get("background_paths") or []):
        try:
            path = Path(raw_path).resolve()
            if path.is_relative_to(generation_root) and path.is_file():
                editable_indices.append(index)
        except (OSError, TypeError):
            continue
    public["layout_editable_indices"] = editable_indices
    public["layout_editable"] = bool(editable_indices)
    return public


def job_state_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise FileNotFoundError("Invalid job id")
    return JOB_STATE_DIR / f"{job_id}.json"


def persist_generation_job(job: dict[str, Any]) -> None:
    JOB_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = job_state_path(str(job["job_id"]))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_persisted_job(job_id: str) -> dict[str, Any] | None:
    try:
        path = job_state_path(job_id)
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def update_job(job_id: str, **changes: Any) -> None:
    snapshot: dict[str, Any] | None = None
    with generation_jobs_lock:
        if job_id in generation_jobs:
            generation_jobs[job_id].update(changes)
            if generation_jobs[job_id].get("status") in {"completed", "failed"}:
                snapshot = dict(generation_jobs[job_id])
    if snapshot is not None:
        persist_generation_job(snapshot)


def update_motion_job(job_id: str, **changes: Any) -> None:
    with motion_jobs_lock:
        if job_id in motion_jobs:
            motion_jobs[job_id].update(changes)


def motion_plan_from_text(text: str, default_style: str = "push") -> dict[str, Any]:
    content = text.strip().lower()
    style = default_style
    if any(word in content for word in ("拉远", "后退", "zoom out", "pull back")):
        style = "pull"
    elif any(word in content for word in ("横移", "向左", "向右", "漂移", "pan", "drift")):
        style = "drift"
    elif any(word in content for word in ("呼吸", "脉冲", "breath", "pulse")):
        style = "breathe"
    elif any(word in content for word in ("推进", "靠近", "放大", "zoom in", "push")):
        style = "push"

    duration = 5
    duration_match = re.search(r"([3-8])\s*(?:秒|s|second)", content)
    if duration_match:
        duration = int(duration_match.group(1))
    effects: list[str] = []
    effect_rules = (
        ("particles", ("粒子", "尘埃", "灰尘", "雪花", "星光", "火花")),
        ("fog", ("烟雾", "雾气", "云雾", "薄雾", "smoke", "fog", "mist")),
        ("light", ("扫光", "光线", "光影", "闪光", "霓虹", "light", "glow")),
        ("wave", ("海浪", "波浪", "水面", "涟漪", "wave", "water", "ripple")),
        ("float", ("悬浮", "漂浮", "浮动", "轻轻上下", "float", "hover")),
    )
    for effect, keywords in effect_rules:
        if any(keyword in content for keyword in keywords):
            effects.append(effect)
    if not effects:
        effects = ["light", "particles"]
    intensity = "high" if any(word in content for word in ("强烈", "明显", "快速", "dramatic")) else "low" if any(word in content for word in ("轻微", "克制", "缓慢", "subtle")) else "medium"
    labels = {
        "push": "镜头缓慢推进",
        "pull": "镜头逐渐拉远",
        "drift": "镜头横向漂移",
        "breathe": "镜头电影呼吸",
        "particles": "环境粒子",
        "fog": "流动雾气",
        "light": "动态扫光",
        "wave": "局部波纹",
        "float": "轻微悬浮",
    }
    summary = "、".join([labels[style], *[labels[item] for item in effects]])
    return {
        "status": "ready",
        "assistant_message": f"我会保留海报主体和构图，采用{summary}，生成一段{duration}秒的本地2.5D动态广告。",
        "questions": [],
        "style": style,
        "duration": duration,
        "effects": effects,
        "intensity": intensity,
        "summary": summary,
        "prompt": text.strip(),
        "engine": "local-motion-director",
    }


def render_motion_asset(
    source_path: Path,
    output_path: Path,
    request: MotionRequest,
    motion_job_id: str,
) -> None:
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGB")

    max_side = 960 if request.format == "mp4" else 720
    if source.width <= source.height:
        out_height = max_side
        out_width = max(2, round(max_side * source.width / source.height))
    else:
        out_width = max_side
        out_height = max(2, round(max_side * source.height / source.width))
    out_width -= out_width % 2
    out_height -= out_height % 2
    base = ImageOps.fit(source, (out_width, out_height), method=Image.Resampling.LANCZOS)
    plan = motion_plan_from_text(request.prompt, request.style)
    motion_style = str(plan["style"])
    effects = set(plan["effects"])
    intensity = {"low": 0.6, "medium": 1.0, "high": 1.45}.get(str(plan["intensity"]), 1.0)

    fps = 24 if request.format == "mp4" else 15
    frame_total = request.duration * fps
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_width}x{out_height}", "-r", str(fps), "-i", "-", "-an",
    ]
    if request.format == "mp4":
        command.extend([
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ])
    else:
        command.extend([
            "-filter_complex",
            "[0:v]fps=12,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop", "0", str(output_path),
        ])

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    rng = np.random.default_rng(2026)
    grid_y, grid_x = np.mgrid[0:out_height, 0:out_width]
    particle_count = max(24, round(70 * intensity))
    particle_x = rng.uniform(0, out_width, particle_count)
    particle_y = rng.uniform(0, out_height, particle_count)
    particle_speed = rng.uniform(12, 42, particle_count) * intensity
    particle_size = rng.integers(1, 4, particle_count)
    try:
        if process.stdin is None:
            raise RuntimeError("无法启动本地视频编码器。")
        for frame_index in range(frame_total):
            t = frame_index / max(1, frame_total - 1)
            ease = t * t * (3 - 2 * t)
            if motion_style == "push":
                zoom = 1.0 + 0.085 * ease
                offset_x, offset_y = 0.0, -0.012 * ease
            elif motion_style == "pull":
                zoom = 1.085 - 0.075 * ease
                offset_x, offset_y = 0.0, -0.006 * (1 - ease)
            elif motion_style == "drift":
                zoom = 1.075
                offset_x = (t - 0.5) * 0.05
                offset_y = math.sin(t * math.pi * 2) * 0.008
            else:
                pulse = math.sin(t * math.pi)
                zoom = 1.025 + 0.025 * pulse
                offset_x, offset_y = 0.0, -0.008 * pulse
            if "float" in effects:
                offset_y += math.sin(t * math.pi * 2) * 0.006 * intensity

            scaled_size = (max(out_width, round(out_width * zoom)), max(out_height, round(out_height * zoom)))
            scaled = base.resize(scaled_size, Image.Resampling.LANCZOS)
            max_x = max(0, scaled.width - out_width)
            max_y = max(0, scaled.height - out_height)
            left = int(max_x / 2 + offset_x * out_width)
            top = int(max_y / 2 + offset_y * out_height)
            left = max(0, min(max_x, left))
            top = max(0, min(max_y, top))
            frame = scaled.crop((left, top, left + out_width, top + out_height))
            pixels = np.asarray(frame, dtype=np.float32)
            if motion_style == "breathe":
                pixels *= 0.985 + 0.045 * math.sin(t * math.pi)
            if "wave" in effects:
                displaced = pixels.copy()
                wave_strength = 4.0 * intensity
                for row in range(out_height // 2, out_height):
                    fade = (row - out_height / 2) / (out_height / 2)
                    shift = round(math.sin(row / 26 + t * math.pi * 2) * wave_strength * fade)
                    displaced[row] = np.roll(pixels[row], shift, axis=0)
                pixels = displaced
            if "fog" in effects:
                fog_x = (-0.25 + 1.5 * t) * out_width
                fog_y = out_height * (0.58 + 0.08 * math.sin(t * math.pi * 2))
                fog = np.exp(-(((grid_x - fog_x) / (out_width * 0.5)) ** 2 + ((grid_y - fog_y) / (out_height * 0.16)) ** 2))
                pixels = pixels * (1 - fog[:, :, None] * 0.16 * intensity) + 215 * fog[:, :, None] * 0.16 * intensity
            if "light" in effects:
                sweep_center = (-0.25 + 1.5 * t) * out_width
                diagonal = grid_x - sweep_center - (grid_y - out_height / 2) * 0.22
                light = np.exp(-(diagonal ** 2) / (2 * (out_width * 0.13) ** 2))
                pixels += light[:, :, None] * 28 * intensity
            if "particles" in effects:
                for px, py, speed, size in zip(particle_x, particle_y, particle_speed, particle_size):
                    x_pos = int(px)
                    y_pos = int((py - speed * request.duration * t) % out_height)
                    radius = int(size)
                    y0, y1 = max(0, y_pos - radius), min(out_height, y_pos + radius + 1)
                    x0, x1 = max(0, x_pos - radius), min(out_width, x_pos + radius + 1)
                    pixels[y0:y1, x0:x1] += 34 * intensity
            grain = rng.normal(0.0, 0.9, pixels.shape[:2])[:, :, None]
            pixels = np.clip(pixels + grain, 0, 255).astype(np.uint8)
            process.stdin.write(pixels.tobytes())
            if frame_index % max(1, fps // 3) == 0:
                progress = 8 + round((frame_index + 1) / frame_total * 84)
                update_motion_job(
                    motion_job_id,
                    progress=progress,
                    message=f"正在渲染动态帧 {frame_index + 1} / {frame_total}",
                )
        process.stdin.close()
        return_code = process.wait(timeout=120)
        error_text = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if return_code != 0:
            raise RuntimeError(error_text.strip() or "FFmpeg 视频编码失败。")
    except Exception:
        if process.poll() is None:
            process.kill()
        raise


def run_motion_job(motion_job_id: str) -> None:
    with motion_jobs_lock:
        job = dict(motion_jobs[motion_job_id])
    request = MotionRequest.model_validate(job["request"])
    source_path = Path(job["source_path"])
    output_path = Path(job["output_path"])
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        update_motion_job(motion_job_id, status="rendering", progress=5, message="正在准备电影感运动轨迹")
        render_motion_asset(source_path, output_path, request, motion_job_id)
        update_motion_job(
            motion_job_id,
            status="completed",
            progress=100,
            message="动态广告已经生成完成。",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
    except Exception as exc:
        update_motion_job(
            motion_job_id,
            status="failed",
            progress=100,
            message="动态广告生成失败。",
            error=str(exc),
        )


def run_generation_job(job_id: str) -> None:
    with generation_jobs_lock:
        context = generation_jobs[job_id]["_context"]
    request = GenerationRequest.model_validate(context["request"])
    mode = context.get("mode", "initial")
    typography_data = context.get("typography")
    typography = None
    if typography_data:
        try:
            typography = TypographySpec.model_validate(typography_data)
        except ValueError:
            typography = None
    try:
        product_path = uploaded_product_path(request.product.upload_id)
        reference_paths = [engine.poster_path(item.vector_index) for item in request.references[:3]]
        palette_paths = list(reference_paths)
        revision_instruction = ""
        if mode == "revision":
            source_path = Path(context["source_path"])
            if not source_path.is_file():
                raise RuntimeError("找不到要修改的上一版图片。")
            palette_paths.insert(0, source_path)
            revision_instruction = background_only_instruction(context["instruction"])

        palette = extract_reference_palette(palette_paths)
        palette_text = ", ".join(palette) if palette else "deep black, cobalt blue, cool white"
        direction = request.style_query.strip() or "premium cinematic luxury advertising"
        update_job(
            job_id,
            status="analyzing",
            progress=4,
            message="正在提取参考海报的场景、构图和创意元素",
        )
        visual_language = product_analyzer.analyze_reference_language(reference_paths)
        scene_concepts = background_safe_concepts(
            visual_language.get("scene", []),
            "an original cinematic environment",
        )
        composition_concepts = background_safe_concepts(
            visual_language.get("composition", []),
            "a wide negative-space composition",
        )
        if "a wide negative-space composition" not in composition_concepts:
            composition_concepts.append("a wide negative-space composition")
        motif_concepts = background_safe_concepts(
            visual_language.get("motif", []),
            "cinematic light and environmental texture",
        )
        scene_language = ", ".join(scene_concepts)
        composition_language = ", ".join(composition_concepts)
        motif_language = ", ".join(motif_concepts)
        visual_language = {
            "scene": scene_concepts,
            "composition": composition_concepts,
            "motif": motif_concepts,
        }
        combined_direction = f"{direction} {revision_instruction}".lower()
        indoor_requested = any(
            word in combined_direction
            for word in ("室内", "房间", "展厅", "走廊", "interior", "indoor", "room", "showroom")
        )
        environment_rule = (
            "An interior is allowed because the user explicitly requested one, but it must still inherit the reference motifs and composition."
            if indoor_requested
            else "Do not use a generic room, apartment, showroom, gallery, corridor or empty studio architecture. Prefer an open, surreal, graphic, outdoor or hybrid conceptual world."
        )
        typography_space = ""
        if typography and typography.enabled:
            typography_space = (
                f" Reserve a calm, high-contrast negative-space area at the {typography.position} for later local "
                "typography. Do not render, imitate or hint at any letters, glyphs or words."
            )
        background_prompt = (
            "Create an original, concept-driven cinematic EMPTY BACKPLATE. It must read as scenery without a subject, not as a finished advertisement. "
            f"{BACKGROUND_OBJECT_BAN} "
            f"User art direction: {direction}. Requested revision: {revision_instruction or 'none'}. "
            "The selected movie posters have already been analyzed locally and are visual-language references only. Transform "
            f"their strongest scene language ({scene_language}), composition rhythm ({composition_language}), and creative "
            f"motifs ({motif_language}). Use the sampled palette {palette_text}. Do not merely place the product in a luxury room. "
            "Build a distinctive visual idea using the outer edges, distant depth, lighting, terrain and continuous surfaces. "
            "Keep the entire central placement area visually open, quiet and physically unoccupied. Translate the references into a new "
            "commercial concept; do not reproduce identifiable movie characters, faces, actors, exact poster layouts, titles, "
            f"logos or copyrighted scenes. {environment_rule} "
            f"Do not depict {product_negative_terms(request.product)} anywhere in the generated background. "
            "Keep the upper title zone visually quiet and free of typographic shapes. "
            "Vertical cinematic poster backplate with premium editorial lighting and clear depth. "
            f"Final compliance reminder: {BACKGROUND_OBJECT_BAN}"
            f"{typography_space}"
        )

        product_scale = float(context.get("product_scale", 0.62))

        outputs: list[dict[str, Any]] = []
        background_paths: list[str] = []
        context["background_paths"] = background_paths
        for index in range(request.variant_count):
            progress = 8 + round(index / max(request.variant_count, 1) * 82)
            update_job(
                job_id,
                status="generating",
                progress=progress,
                message=f"正在生成第 {index + 1} / {request.variant_count} 张电影感背景",
            )
            output_name = f"{job_id}-{index + 1}.jpg"
            output_path = GENERATION_DIR / output_name
            background_path = GENERATION_DIR / f"{job_id}-{index + 1}-background.jpg"
            creative_route = VARIANT_CREATIVE_ROUTES[index % len(VARIANT_CREATIVE_ROUTES)]
            variant_prompt = (
                f"{background_prompt} CREATIVE ROUTE {index + 1}: {creative_route} "
                "Make this proposal visibly different from the other routes while preserving the shared reference DNA."
            )
            image_generator.generate(
                prompt=variant_prompt,
                input_paths=[],
                output_path=background_path,
                width=request.width,
                height=request.height,
                seed=random.randint(1, 2_147_483_647),
            )
            update_job(
                job_id,
                progress=min(96, progress + 10),
                message=f"正在融合原始商品与第 {index + 1} 张背景",
            )
            composite_product(
                background_path=background_path,
                product_path=product_path,
                output_path=output_path,
                product_scale=product_scale,
                product_category=request.product.category,
                typography=typography,
            )
            background_paths.append(str(background_path.resolve()))
            outputs.append(
                {
                    "index": index,
                    "image_url": f"/api/generated/{output_name}",
                    "download_url": f"/api/generated/{output_name}?download=1",
                }
            )

        update_job(
            job_id,
            status="completed",
            progress=100,
            message="海报方案已经生成完成。",
            outputs=outputs,
            creative_elements=visual_language,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
    except Exception as exc:
        error_message = str(exc)
        if "flagged" in error_message.lower():
            error_message = (
                "Cloudflare 内容过滤器拦截了这组输入。请换一张不含敏感人物的参考海报，"
                "或在风格描述中强调只借鉴色彩、光线和构图后重试。"
            )
        update_job(job_id, status="failed", progress=100, message=error_message, error=error_message)


def create_generation_job(request: GenerationRequest, context_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    context = {"request": request.model_dump(), "mode": "initial"}
    if context_extra:
        context.update(context_extra)
    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "生成任务已进入队列。",
        "outputs": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parent_job_id": context.get("parent_job_id"),
        "revision_instruction": context.get("instruction"),
        "_context": context,
    }
    with generation_jobs_lock:
        generation_jobs[job_id] = job
    persist_generation_job(job)
    threading.Thread(target=run_generation_job, args=(job_id,), daemon=True).start()
    return public_job(job)


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.load()
    yield


app = FastAPI(
    title="帧镜 AI — 电影感商业广告生成 Agent",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return engine.status()


@app.post("/api/search")
def api_search(request: SearchRequest) -> dict[str, Any]:
    try:
        return engine.search(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/product/analyze")
def api_product_analyze(
    file: UploadFile = File(...),
    product_name: str = Form(default=""),
    description: str = Form(default=""),
) -> dict[str, Any]:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="仅支持 JPG、PNG 或 WebP 图片。")
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 20 MB。")
    if not raw:
        raise HTTPException(status_code=400, detail="上传的图片为空。")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    image_path = UPLOAD_DIR / f"{upload_id}.png"
    try:
        with Image.open(BytesIO(raw)) as source:
            source.verify()
        with Image.open(BytesIO(raw)) as source:
            image = source.convert("RGBA" if "A" in source.getbands() else "RGB")
            image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
            image.save(image_path, "PNG", optimize=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="无法读取这张图片。") from exc

    try:
        candidates = product_analyzer.analyze(image_path)
    except RuntimeError as exc:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    predicted = candidates[0]["category"]
    return {
        "upload_id": upload_id,
        "image_url": f"/api/products/{upload_id}",
        "name": product_name.strip() or predicted,
        "category": predicted,
        "description": description.strip(),
        "candidates": candidates,
        "note": "识别结果是视觉分类建议，你可以直接修改商品名称与描述。",
    }


@app.get("/api/products/{upload_id}")
def api_product_image(upload_id: str) -> FileResponse:
    try:
        path = uploaded_product_path(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Product image not found")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/agent/reference")
def api_agent_reference(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if (
        file.content_type not in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
        and suffix not in {".jpg", ".jpeg", ".png", ".webp"}
    ):
        raise HTTPException(status_code=415, detail="参考图仅支持 JPG、PNG 或 WebP。")
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="参考图不能超过 20 MB。")
    if not raw:
        raise HTTPException(status_code=400, detail="上传的参考图为空。")

    REFERENCE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    reference_id = uuid.uuid4().hex
    path = REFERENCE_UPLOAD_DIR / f"{reference_id}.jpg"
    try:
        with Image.open(BytesIO(raw)) as source:
            source.verify()
        with Image.open(BytesIO(raw)) as source:
            image = source.convert("RGB")
            image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
            image.save(path, "JPEG", quality=92, optimize=True)
    except (UnidentifiedImageError, OSError) as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="无法读取这张参考图。") from exc
    analysis = reference_visual_summary(path)
    return {
        "reference_id": reference_id,
        "image_url": f"/api/agent/references/{reference_id}",
        "analysis": analysis,
        "note": "参考图仅提取视觉特征；请继续告诉 Agent 想借用哪些元素。",
    }


@app.get("/api/agent/references/{reference_id}")
def api_agent_reference_image(reference_id: str) -> FileResponse:
    try:
        path = agent_reference_path(reference_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Reference image not found") from exc
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/agent/turn")
def api_agent_turn(request: AgentTurnRequest) -> dict[str, Any]:
    reference_summary = ""
    if request.reference_id:
        try:
            reference_summary = reference_visual_summary(
                agent_reference_path(request.reference_id)
            )["summary"]
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail="参考图已经不存在，请重新上传。") from exc
    return creative_agent.turn(request, reference_summary)


@app.post("/api/brief")
def api_brief(request: BriefRequest) -> dict[str, Any]:
    return build_brief(request)


@app.get("/api/generation/status")
def api_generation_status() -> dict[str, Any]:
    return image_generator.status()


@app.post("/api/generate")
def api_generate(request: GenerationRequest) -> dict[str, Any]:
    problem = image_generator.configuration_problem()
    if problem:
        raise HTTPException(status_code=503, detail=problem)
    try:
        uploaded_product_path(request.product.upload_id)
        for reference in request.references[:3]:
            engine.poster_path(reference.vector_index)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=400, detail="商品图或海报参考已经不存在。") from exc
    return create_generation_job(request)


@app.get("/api/jobs/{job_id}")
def api_generation_job(job_id: str) -> dict[str, Any]:
    with generation_jobs_lock:
        job = generation_jobs.get(job_id)
    if job is None:
        job = load_persisted_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found")
        with generation_jobs_lock:
            generation_jobs[job_id] = job
    with generation_jobs_lock:
        return public_job(job)


@app.post("/api/jobs/{job_id}/revise")
def api_revise_generation(job_id: str, request: RevisionRequest) -> dict[str, Any]:
    problem = image_generator.configuration_problem()
    if problem:
        raise HTTPException(status_code=503, detail=problem)
    with generation_jobs_lock:
        parent = generation_jobs.get(job_id)
    if parent is None:
        parent = load_persisted_job(job_id)
        if parent is not None:
            with generation_jobs_lock:
                generation_jobs[job_id] = parent
    with generation_jobs_lock:
        if parent is None:
            raise HTTPException(status_code=404, detail="Generation job not found")
        if parent["status"] != "completed":
            raise HTTPException(status_code=409, detail="上一轮生成尚未完成。")
        outputs = parent.get("outputs") or []
        if request.source_index >= len(outputs):
            raise HTTPException(status_code=400, detail="请选择有效的海报版本。")
        source_name = Path(outputs[request.source_index]["image_url"]).name
        source_path = (GENERATION_DIR / source_name).resolve()
        if not source_path.is_relative_to(GENERATION_DIR.resolve()) or not source_path.is_file():
            raise HTTPException(status_code=404, detail="找不到所选海报版本。")
        parent_context = parent.get("_context") or {}
        if parent_context.get("request"):
            base_request = GenerationRequest.model_validate(parent_context["request"])
        else:
            recovered_product = latest_uploaded_product()
            if recovered_product is None:
                raise HTTPException(
                    status_code=409,
                    detail="旧版本缺少商品上下文，请重新上传商品后再修改。",
                )
            base_request = GenerationRequest(
                prompt=request.instruction,
                style_query=request.instruction,
                product=ProductSpec(
                    upload_id=recovered_product.stem,
                    name="商品",
                    category="",
                ),
                references=[],
                variant_count=request.variant_count,
            )
    revised_request = base_request.model_copy(update={"variant_count": request.variant_count})
    parent_scale = float((parent.get("layout") or {}).get("scale", parent_context.get("product_scale", 0.62)))
    scale_map = {
        "none": parent_scale,
        "larger": min(0.90, parent_scale + 0.10),
        "smaller": max(0.30, parent_scale - 0.10),
    }
    typography_data = (
        request.typography.model_dump()
        if request.typography
        else parent_context.get("typography")
    )
    return create_generation_job(
        revised_request,
        {
            "mode": "revision",
            "source_path": str(source_path),
            "instruction": request.instruction.strip(),
            "parent_job_id": job_id,
            "typography": typography_data,
            "product_scale": scale_map[request.product_scale_action],
        },
    )


@app.post("/api/jobs/{job_id}/layout")
def api_adjust_product_layout(job_id: str, request: LayoutRequest) -> dict[str, Any]:
    with generation_jobs_lock:
        parent = generation_jobs.get(job_id)
    if parent is None:
        parent = load_persisted_job(job_id)
        if parent is not None:
            with generation_jobs_lock:
                generation_jobs[job_id] = parent
    if parent is None:
        raise HTTPException(status_code=404, detail="Generation job not found")
    if parent.get("status") != "completed":
        raise HTTPException(status_code=409, detail="当前版本尚未生成完成。")

    outputs = parent.get("outputs") or []
    if request.source_index >= len(outputs):
        raise HTTPException(status_code=400, detail="请选择有效的海报版本。")
    parent_context = parent.get("_context") or {}
    if not parent_context.get("request"):
        raise HTTPException(status_code=409, detail="这个旧版本缺少商品图层信息，请先重新生成一版海报。")
    base_request = GenerationRequest.model_validate(parent_context["request"])

    background_paths = parent_context.get("background_paths") or []
    background_path: Path | None = None
    if request.source_index < len(background_paths):
        candidate = Path(background_paths[request.source_index]).resolve()
        if candidate.is_relative_to(GENERATION_DIR.resolve()) and candidate.is_file():
            background_path = candidate
    if background_path is None:
        candidate = (GENERATION_DIR / f"{job_id}-{request.source_index + 1}-background.jpg").resolve()
        if candidate.is_relative_to(GENERATION_DIR.resolve()) and candidate.is_file():
            background_path = candidate
    if background_path is None:
        raise HTTPException(
            status_code=409,
            detail="这个旧版本没有保存纯背景层。请先用新版流程重新生成，再手动调整商品。",
        )

    try:
        product_path = uploaded_product_path(base_request.product.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="找不到这个版本使用的商品原图。") from exc

    layout_job_id = uuid.uuid4().hex
    output_name = f"{layout_job_id}-1.jpg"
    output_path = GENERATION_DIR / output_name
    typography_data = parent_context.get("typography")
    composite_product(
        background_path=background_path,
        product_path=product_path,
        output_path=output_path,
        product_scale=request.scale,
        product_category=base_request.product.category,
        typography=typography_data,
        product_offset_x=request.offset_x,
        product_offset_y=request.offset_y,
    )
    layout = {
        "scale": request.scale,
        "offset_x": request.offset_x,
        "offset_y": request.offset_y,
    }
    layout_context = {
        "request": base_request.model_copy(update={"variant_count": 1}).model_dump(),
        "mode": "layout",
        "parent_job_id": job_id,
        "background_paths": [str(background_path)],
        "typography": typography_data,
        "product_scale": request.scale,
        "manual_layout": layout,
    }
    layout_job = {
        "job_id": layout_job_id,
        "status": "completed",
        "progress": 100,
        "message": "商品大小与位置已经在本地调整完成。",
        "outputs": [
            {
                "index": 0,
                "image_url": f"/api/generated/{output_name}",
                "download_url": f"/api/generated/{output_name}?download=1",
            }
        ],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parent_job_id": job_id,
        "revision_instruction": "手动调整商品大小与位置",
        "layout": layout,
        "_context": layout_context,
    }
    with generation_jobs_lock:
        generation_jobs[layout_job_id] = layout_job
    persist_generation_job(layout_job)
    return public_job(layout_job)


@app.post("/api/jobs/{job_id}/layout-preview")
def api_preview_product_layout(job_id: str, request: LayoutRequest) -> Response:
    """Render a non-persisted layout preview for live slider feedback."""
    with generation_jobs_lock:
        parent = generation_jobs.get(job_id)
    if parent is None:
        parent = load_persisted_job(job_id)
        if parent is not None:
            with generation_jobs_lock:
                generation_jobs[job_id] = parent
    if parent is None:
        raise HTTPException(status_code=404, detail="Generation job not found")
    if parent.get("status") != "completed":
        raise HTTPException(status_code=409, detail="当前版本尚未生成完成。")

    outputs = parent.get("outputs") or []
    if request.source_index >= len(outputs):
        raise HTTPException(status_code=400, detail="请选择有效的海报版本。")
    parent_context = parent.get("_context") or {}
    if not parent_context.get("request"):
        raise HTTPException(status_code=409, detail="这个旧版本缺少商品图层信息。")
    base_request = GenerationRequest.model_validate(parent_context["request"])

    background_paths = parent_context.get("background_paths") or []
    background_path: Path | None = None
    if request.source_index < len(background_paths):
        candidate = Path(background_paths[request.source_index]).resolve()
        if candidate.is_relative_to(GENERATION_DIR.resolve()) and candidate.is_file():
            background_path = candidate
    if background_path is None:
        candidate = (GENERATION_DIR / f"{job_id}-{request.source_index + 1}-background.jpg").resolve()
        if candidate.is_relative_to(GENERATION_DIR.resolve()) and candidate.is_file():
            background_path = candidate
    if background_path is None:
        raise HTTPException(status_code=409, detail="这个旧版本没有保存纯背景层。")

    try:
        product_path = uploaded_product_path(base_request.product.upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="找不到这个版本使用的商品原图。") from exc

    preview = render_product_composite(
        background_path=background_path,
        product_path=product_path,
        product_scale=request.scale,
        product_category=base_request.product.category,
        typography=parent_context.get("typography"),
        product_offset_x=request.offset_x,
        product_offset_y=request.offset_y,
    )
    payload = BytesIO()
    preview.save(payload, "JPEG", quality=88, optimize=True)
    return Response(
        content=payload.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


def public_motion_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"source_path", "output_path", "request"}}


@app.post("/api/motion/plan")
def api_motion_plan(request: MotionPlanRequest) -> dict[str, Any]:
    return motion_plan_from_text(request.message)


@app.post("/api/jobs/{job_id}/motion")
def api_create_motion(job_id: str, request: MotionRequest) -> dict[str, Any]:
    with generation_jobs_lock:
        parent = generation_jobs.get(job_id)
    if parent is None:
        parent = load_persisted_job(job_id)
        if parent is not None:
            with generation_jobs_lock:
                generation_jobs[job_id] = parent
    if parent is None:
        raise HTTPException(status_code=404, detail="Generation job not found")
    if parent.get("status") != "completed":
        raise HTTPException(status_code=409, detail="请等待海报生成完成后再制作动态版本。")
    outputs = parent.get("outputs") or []
    if request.source_index >= len(outputs):
        raise HTTPException(status_code=400, detail="请选择有效的海报版本。")

    image_url = str(outputs[request.source_index].get("image_url", ""))
    source_name = Path(image_url.split("?", 1)[0]).name
    if not re.fullmatch(r"[0-9a-f]{32}-[1-9][0-9]*\.jpg", source_name):
        raise HTTPException(status_code=409, detail="找不到这张海报对应的本地图片。")
    source_path = (GENERATION_DIR / source_name).resolve()
    if not source_path.is_relative_to(GENERATION_DIR.resolve()) or not source_path.is_file():
        raise HTTPException(status_code=404, detail="找不到这张海报的原始文件。")

    motion_job_id = uuid.uuid4().hex
    output_name = f"{motion_job_id}.{request.format}"
    output_path = (MOTION_DIR / output_name).resolve()
    plan = motion_plan_from_text(request.prompt, request.style)
    motion_job = {
        "job_id": motion_job_id,
        "source_job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "动态广告任务已进入队列。",
        "format": request.format,
        "style": plan["style"],
        "duration": request.duration,
        "plan": plan,
        "file_url": f"/api/motion/{output_name}",
        "download_url": f"/api/motion/{output_name}?download=1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "request": request.model_dump(),
    }
    with motion_jobs_lock:
        motion_jobs[motion_job_id] = motion_job
    threading.Thread(target=run_motion_job, args=(motion_job_id,), daemon=True).start()
    return public_motion_job(motion_job)


@app.get("/api/motion-jobs/{motion_job_id}")
def api_motion_job(motion_job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", motion_job_id):
        raise HTTPException(status_code=404, detail="Motion job not found")
    with motion_jobs_lock:
        job = motion_jobs.get(motion_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Motion job not found")
        return public_motion_job(dict(job))


@app.get("/api/motion/{file_name}")
def api_motion_file(file_name: str, download: int = 0) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{32}\.(?:mp4|gif)", file_name):
        raise HTTPException(status_code=404, detail="Motion file not found")
    path = (MOTION_DIR / file_name).resolve()
    if not path.is_relative_to(MOTION_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Motion file not found")
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "image/gif"
    headers = {"Cache-Control": "private, max-age=3600"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="frame-mirror-{file_name}"'
    return FileResponse(path, media_type=media_type, headers=headers)


@app.get("/api/generated/recent")
def api_recent_generated_images() -> dict[str, Any]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    if GENERATION_DIR.is_dir():
        for path in GENERATION_DIR.glob("*.jpg"):
            match = re.fullmatch(r"([0-9a-f]{32})-([1-9][0-9]*)\.jpg", path.name)
            if not match:
                continue
            job_id, index_text = match.groups()
            groups.setdefault(job_id, []).append((int(index_text), path))
    if not groups:
        return {"job": None}

    latest_job_id = max(
        groups,
        key=lambda job_id: max(path.stat().st_mtime for _, path in groups[job_id]),
    )
    persisted = load_persisted_job(latest_job_id)
    if persisted is not None:
        persisted["restored"] = True
        persisted["message"] = "已从本地恢复最近生成的广告海报，可继续交给 Agent 修改。"
        with generation_jobs_lock:
            generation_jobs[latest_job_id] = persisted
        return {"job": public_job(persisted)}
    files = sorted(groups[latest_job_id], key=lambda item: item[0])
    outputs = [
        {
            "index": index - 1,
            "image_url": f"/api/generated/{path.name}",
            "download_url": f"/api/generated/{path.name}?download=1",
        }
        for index, path in files
    ]
    return {
        "job": {
            "job_id": latest_job_id,
            "status": "completed",
            "progress": 100,
            "message": "已从本地恢复最近生成的广告海报。",
            "outputs": outputs,
            "restored": True,
        }
    }


@app.get("/api/generated/{file_name}")
def api_generated_image(file_name: str, download: int = 0) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{32}-[1-9][0-9]*\.jpg", file_name):
        raise HTTPException(status_code=404, detail="Generated image not found")
    path = (GENERATION_DIR / file_name).resolve()
    if not path.is_relative_to(GENERATION_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Generated image not found")
    headers = {"Cache-Control": "private, max-age=3600"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
    return FileResponse(path, media_type="image/jpeg", headers=headers)


@app.get("/api/posters/{vector_index}")
def api_poster(vector_index: int) -> FileResponse:
    try:
        path = engine.poster_path(vector_index)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Poster not found") from exc
    return FileResponse(
        path,
        headers={"Cache-Control": "public, max-age=86400"},
    )


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

