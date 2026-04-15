"""ImageToolkit — image analysis toolkit for the agent."""

from __future__ import annotations

import json
import re
from typing import Any

from ...answer import Evidence
from ...retrieval.sqlite_fts import SQLiteFTSBackend
from ..base import Tool, Toolkit, ToolParam
from .media import ImageInfo, crop_image, load_image
from .workspace import ImageWorkspace

_IMAGE_STRATEGY_PROMPT = """\
## Available Tools

You are analyzing images. Work methodically: explore, verify, conclude.

### Vision tools (use these to SEE images)

- **vision_query(prompt, image_id=, image_ids=)** — ask a question about 1+ images.
  Use for targeted checks: OCR, object detection, spatial relations, counts.

- **vision_query_batched(queries)** — ask DIFFERENT questions about DIFFERENT images
  in ONE fast request. **Always prefer this for 3+ images.**
  ```python
  vision_query_batched([
      {"prompt": "What objects are visible?", "image_id": "img_0"},
      {"prompt": "What objects are visible?", "image_id": "img_1"},
      {"prompt": "What objects are visible?", "image_id": "img_2"},
  ])
  ```

- **compare_images(image_id_1, image_id_2, prompt=)** — compare two images directly.
  Use for differences, before/after, or similarity checks.

### Discovery tools

- **list_images()** — see all loaded images with IDs and dimensions.
- **search_images(query)** — keyword search over auto-generated captions.
  Returns ranked matches. Use to narrow candidates before vision queries.
- **get_image_info(image_id)** — metadata for one image.
- **crop_region(image_id, region_description)** — zoom into a region for detail.

### Synthesis tool

- **llm_query(prompt)** — text-only reasoning over your observations.

## Workflow (for many images)

1. **list_images()** — understand what's available.
2. **search_images(keywords)** — narrow to candidates (faster than querying all).
3. **vision_query_batched([...])** — verify candidates visually in ONE call.
4. **done(answer)** — return your answer with evidence.

## Workflow (for few images, 1-3)

1. **list_images()** — see the images.
2. **vision_query(prompt)** — analyze directly.
3. **done(answer)** — return answer.

## Rules

- ALWAYS use vision tools before concluding. Never guess from filenames alone.
- Use **vision_query_batched** when checking 3+ images — it's 5x faster.
- Cite `image_id` for every claim.
- Quote text verbatim when readable; say "unclear" if not.
- One code block per response. Print results.
"""

_CAPTION_PROMPT = (
    "Create a dense retrieval caption for this image. Include key subjects, actions, "
    "setting, visible text/OCR, spatial layout, dominant colors, and distinctive details. "
    "If text is unreadable, explicitly say text is unclear."
)


def _record_vision_budget(llm_client: Any, budget: Any) -> None:
    """Record vision call cost in the budget tracker."""
    if budget is None:
        return
    usage = getattr(llm_client, "last_usage", None)
    if usage:
        cost = getattr(llm_client, "last_cost_usd", None) or 0.0
        model = getattr(llm_client, "model", None)
        budget.record(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            model=str(model) if model else "vision",
        )


class ImageToolkit(Toolkit):
    """Image analysis toolkit with vision query, crop, search, and compare tools."""

    def __init__(
        self,
        workspace_dir: str = "./sanjaya_artifacts",
        max_vision_images: int = 8,
        max_crop_dim: int = 1024,
    ):
        self.workspace_dir = workspace_dir
        self.max_vision_images = max_vision_images
        self.max_crop_dim = max_crop_dim

        # State initialized during setup()
        self._images: dict[str, ImageInfo] = {}
        self._workspace: ImageWorkspace | None = None
        self._question: str | None = None

        # Injected by Agent.use()
        self._llm_client: Any = None
        self._tracer: Any = None
        self._budget: Any = None
        self._captioner: Any = None

        # Tracking
        self._queried_images: set[str] = set()
        self._crops: dict[str, dict[str, Any]] = {}
        self._crop_counter: int = 0

        # Lazy captioning state
        self._captions: dict[str, str] = {}
        self._captions_indexed: bool = False
        self._fts: SQLiteFTSBackend | None = None

    def setup(self, context: dict[str, Any]) -> None:
        """Load images and create workspace."""
        image_paths = context.get("image")
        if not image_paths:
            return

        self._question = context.get("question")

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Load each image and assign unique IDs
        existing_ids: set[str] = set()
        for path in image_paths:
            try:
                info = load_image(path)
            except (FileNotFoundError, ValueError, ImportError) as e:
                # Log but don't crash — skip unloadable images
                from rich.console import Console
                Console().print(f"[yellow]Skipping image {path}: {e}[/]")
                continue

            # Deduplicate image_id (same pattern as DocumentToolkit)
            base_id = info.image_id
            image_id = base_id
            counter = 1
            while image_id in existing_ids:
                image_id = f"{base_id}_{counter}"
                counter += 1
            info.image_id = image_id
            existing_ids.add(image_id)

            self._images[image_id] = info

        # Create workspace for crops/normalized images
        if self._images:
            self._workspace = ImageWorkspace(base_dir=self.workspace_dir)

    def teardown(self) -> None:
        pass

    def tools(self) -> list[Tool]:
        return [
            self._make_list_images_tool(),
            self._make_get_image_info_tool(),
            self._make_vision_query_tool(),
            self._make_vision_query_batched_tool(),
            self._make_crop_region_tool(),
            self._make_search_images_tool(),
            self._make_compare_images_tool(),
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "images_loaded": len(self._images),
            "images_queried": sorted(self._queried_images),
            "crops_made": len(self._crops),
            "captions_cached": len(self._captions),
            "images": {
                image_id: {
                    "width": info.width,
                    "height": info.height,
                    "format": info.format,
                }
                for image_id, info in self._images.items()
            },
        }

    def build_evidence(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for image_id in sorted(self._queried_images):
            info = self._images.get(image_id)
            if not info:
                continue

            # Collect crops for this image
            image_crops = [
                c for c in self._crops.values()
                if c.get("image_id") == image_id
            ]

            evidence.append(
                Evidence(
                    source=f"image:{image_id}",
                    rationale=f"Analyzed image {info.path} ({info.width}x{info.height} {info.format})",
                    artifacts={
                        "path": info.path,
                        "width": info.width,
                        "height": info.height,
                        "crops": [c.get("crop_path") for c in image_crops],
                    },
                )
            )
        return evidence

    def prompt_section(self) -> str | None:
        if not self._images:
            return None

        if self._prompt_config is not None and self._prompt_config.image_strategy:
            base = self._prompt_config.image_strategy
        else:
            base = _IMAGE_STRATEGY_PROMPT

        parts = [base]

        image_summaries = []
        for info in self._images.values():
            image_summaries.append(
                f"{info.image_id} ({info.width}x{info.height} {info.format})"
            )
        parts.append(
            f"\n{len(self._images)} image(s) loaded: {', '.join(image_summaries)}"
        )

        return "\n".join(parts)

    # ── Tool factories ──────────────────────────────────────

    def _make_list_images_tool(self) -> Tool:
        toolkit = self

        def _list_images() -> list[dict]:
            """List all loaded images with their dimensions and format."""
            return [
                {
                    "image_id": info.image_id,
                    "width": info.width,
                    "height": info.height,
                    "format": info.format,
                    "file_size_kb": round(info.file_size_bytes / 1024, 1),
                }
                for info in toolkit._images.values()
            ]

        return Tool(
            name="list_images",
            description="List all loaded images with dimensions, format, and file size.",
            fn=_list_images,
            parameters={},
            return_type="list[dict]",
        )

    def _make_get_image_info_tool(self) -> Tool:
        toolkit = self

        def _get_image_info(image_id: str) -> dict:
            """Get detailed metadata for a single image, including EXIF if available."""
            info = toolkit._images.get(image_id)
            if info is None:
                return {
                    "error": f"Unknown image_id: {image_id}. "
                    "Use list_images() to see available images."
                }
            
            from pathlib import Path
            filename = Path(info.path).name
            
            result = {
                "image_id": info.image_id,
                "filename": filename,
                "path": info.path,
                "width": info.width,
                "height": info.height,
                "format": info.format,
                "mode": info.mode,
                "file_size_kb": round(info.file_size_bytes / 1024, 1),
            }
            
            # Try to extract EXIF data (date, GPS, camera)
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS, GPSTAGS
                
                with Image.open(info.path) as img:
                    exif_data = img._getexif()
                    if exif_data:
                        exif_info = {}
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag == "DateTimeOriginal":
                                exif_info["date_taken"] = str(value)
                            elif tag == "DateTime":
                                exif_info["date_modified"] = str(value)
                            elif tag == "Make":
                                exif_info["camera_make"] = str(value)
                            elif tag == "Model":
                                exif_info["camera_model"] = str(value)
                            elif tag == "GPSInfo":
                                # Parse GPS coordinates
                                gps = {}
                                for gps_tag_id, gps_value in value.items():
                                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                                    gps[gps_tag] = gps_value
                                if "GPSLatitude" in gps and "GPSLongitude" in gps:
                                    def to_decimal(coords, ref):
                                        d, m, s = coords
                                        decimal = float(d) + float(m)/60 + float(s)/3600
                                        if ref in ['S', 'W']:
                                            decimal = -decimal
                                        return round(decimal, 6)
                                    try:
                                        lat = to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
                                        lon = to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
                                        exif_info["gps_coordinates"] = {"lat": lat, "lon": lon}
                                    except Exception:
                                        pass
                        if exif_info:
                            result["exif"] = exif_info
            except Exception:
                pass  # EXIF extraction failed, continue without it
            
            return result

        return Tool(
            name="get_image_info",
            description=(
                "Get detailed metadata for an image: filename, path, dimensions, format, "
                "and EXIF data (date taken, GPS coordinates, camera) if available."
            ),
            fn=_get_image_info,
            parameters={
                "image_id": ToolParam(
                    name="image_id",
                    type_hint="str",
                    description="Image ID from list_images().",
                ),
            },
            return_type="dict",
        )

    def _make_vision_query_tool(self) -> Tool:
        toolkit = self

        def _vision_query(
            prompt: str,
            image_id: str | None = None,
            image_ids: list[str] | None = None,
        ) -> str:
            """Query a vision model about one or more images.

            If neither image_id nor image_ids is given and only one image
            is loaded, that image is used automatically.

            Args:
                prompt: What to ask about the visual content.
                image_id: Single image ID from list_images().
                image_ids: Multiple image IDs to send together.
            """
            if toolkit._llm_client is None:
                raise RuntimeError("Vision model not configured.")

            # Resolve which images to send
            ids_to_send: list[str] = []
            if image_ids:
                ids_to_send = list(image_ids)
            elif image_id:
                ids_to_send = [image_id]
            elif len(toolkit._images) == 1:
                ids_to_send = list(toolkit._images.keys())
            else:
                raise ValueError(
                    "Specify image_id or image_ids. "
                    f"Available: {list(toolkit._images.keys())}"
                )

            # Validate and collect paths
            frame_paths: list[str] = []
            for iid in ids_to_send:
                info = toolkit._images.get(iid)
                if info is None:
                    raise ValueError(
                        f"Unknown image_id: {iid}. "
                        f"Available: {list(toolkit._images.keys())}"
                    )
                frame_paths.append(info.path)
                toolkit._queried_images.add(iid)

            # Cap number of images
            if len(frame_paths) > toolkit.max_vision_images:
                frame_paths = frame_paths[:toolkit.max_vision_images]

            tracer = toolkit._tracer
            vision_model = getattr(toolkit._llm_client, "vision_model", "unknown")
            model_label = (
                vision_model
                if isinstance(vision_model, str)
                else getattr(vision_model, "model_name", "unknown")
            )

            if tracer:
                with tracer._span(
                    "sanjaya.sub_llm_call.image_vision",
                    model=model_label,
                    prompt_chars=len(prompt),
                    n_images=len(frame_paths),
                    image_ids=ids_to_send,
                ) as ctx:
                    result = toolkit._llm_client.vision_completion(
                        prompt=prompt,
                        frame_paths=frame_paths,
                    )
                    ctx.record_response(result)
                    metadata = toolkit._llm_client.last_call_metadata
                    if metadata:
                        ctx.record(
                            cost_usd=metadata.cost_usd,
                            duration_seconds=metadata.duration_seconds,
                        )
                    _record_vision_budget(toolkit._llm_client, toolkit._budget)
                    return result
            else:
                result = toolkit._llm_client.vision_completion(
                    prompt=prompt,
                    frame_paths=frame_paths,
                )
                _record_vision_budget(toolkit._llm_client, toolkit._budget)
                return result

        return Tool(
            name="vision_query",
            description=(
                "Query a vision model about one or more images. "
                "If only one image is loaded, it is used automatically."
            ),
            fn=_vision_query,
            parameters={
                "prompt": ToolParam(
                    name="prompt",
                    type_hint="str",
                    description="What to ask about the visual content.",
                ),
                "image_id": ToolParam(
                    name="image_id",
                    type_hint="str | None",
                    default=None,
                    description="Single image ID from list_images().",
                ),
                "image_ids": ToolParam(
                    name="image_ids",
                    type_hint="list[str] | None",
                    default=None,
                    description="Multiple image IDs to send together.",
                ),
            },
            return_type="str",
        )

    def _make_vision_query_batched_tool(self) -> Tool:
        toolkit = self

        def _vision_query_batched(queries: list[dict]) -> list[str]:
            """Run multiple vision queries in a single batched request.

            Much faster than sequential vision_query() calls when you need
            to ask different questions about different images.

            Args:
                queries: List of dicts, each with:
                    - prompt (str): Question to ask
                    - image_id (str, optional): Single image ID
                    - image_ids (list[str], optional): Multiple image IDs
            """
            if toolkit._llm_client is None:
                raise RuntimeError("Vision model not configured.")

            batch: list[dict] = []

            for q in queries:
                prompt = q.get("prompt", "Describe this image.")
                image_id = q.get("image_id")
                image_ids = q.get("image_ids")

                # Resolve which images to send for this query
                ids_to_send: list[str] = []
                if image_ids:
                    ids_to_send = list(image_ids)
                elif image_id:
                    ids_to_send = [image_id]
                elif len(toolkit._images) == 1:
                    ids_to_send = list(toolkit._images.keys())

                # Collect paths
                frame_paths: list[str] = []
                for iid in ids_to_send:
                    info = toolkit._images.get(iid)
                    if info is not None:
                        frame_paths.append(info.path)
                        toolkit._queried_images.add(iid)

                # Cap images per query
                if len(frame_paths) > toolkit.max_vision_images:
                    frame_paths = frame_paths[:toolkit.max_vision_images]

                batch.append({
                    "prompt": prompt,
                    "frame_paths": frame_paths if frame_paths else None,
                    "clip_paths": None,
                })

            tracer = toolkit._tracer
            vision_model = getattr(toolkit._llm_client, "vision_model", "unknown")
            model_label = (
                vision_model
                if isinstance(vision_model, str)
                else getattr(vision_model, "model_name", "unknown")
            )

            if tracer:
                with tracer._span(
                    "sanjaya.sub_llm_call.image_vision_batched",
                    model=model_label,
                    n_queries=len(batch),
                    batched=True,
                ) as ctx:
                    results = toolkit._llm_client.vision_completion_batched(batch)
                    ctx.record(n_results=len(results))
                    _record_vision_budget(toolkit._llm_client, toolkit._budget)
                    return results
            else:
                results = toolkit._llm_client.vision_completion_batched(batch)
                _record_vision_budget(toolkit._llm_client, toolkit._budget)
                return results

        return Tool(
            name="vision_query_batched",
            description=(
                "Run multiple vision queries in a single batched request. "
                "Much faster than sequential vision_query() calls. "
                "Pass a list of {prompt, image_id} dicts."
            ),
            fn=_vision_query_batched,
            parameters={
                "queries": ToolParam(
                    name="queries",
                    type_hint="list[dict]",
                    description=(
                        "List of query dicts, each with: "
                        "prompt (str), image_id (str, optional), image_ids (list[str], optional)."
                    ),
                ),
            },
            return_type="list[str]",
        )

    # ── Rich tools (crop, search, compare) ──────────────────

    def _make_crop_region_tool(self) -> Tool:
        toolkit = self

        def _crop_region(image_id: str, region_description: str) -> dict:
            """Crop a described region from an image for closer inspection.

            Uses the vision model to locate the region by description,
            then crops and saves it. The cropped image can be passed to
            vision_query() for detailed analysis.

            Args:
                image_id: Image ID from list_images().
                region_description: Natural language description of the region to crop.
            """
            if toolkit._llm_client is None:
                raise RuntimeError("Vision model not configured.")

            info = toolkit._images.get(image_id)
            if info is None:
                return {
                    "error": f"Unknown image_id: {image_id}. "
                    "Use list_images() to see available images."
                }

            if toolkit._workspace is None:
                raise RuntimeError("Workspace not initialized.")

            # Ask vision model for bounding box
            bbox_prompt = (
                f'Look at this image and find the region described as: "{region_description}"\n'
                f"Return the bounding box as JSON: "
                f'{{"x": <left>, "y": <top>, "x2": <right>, "y2": <bottom>}}\n'
                f"Coordinates are in pixels. The image is {info.width}x{info.height}.\n"
                f"Return ONLY the JSON, no other text."
            )

            bbox_response = toolkit._llm_client.vision_completion(
                prompt=bbox_prompt,
                frame_paths=[info.path],
            )
            _record_vision_budget(toolkit._llm_client, toolkit._budget)
            toolkit._queried_images.add(image_id)

            # Parse bbox JSON from response
            bbox = _parse_bbox_response(bbox_response, info.width, info.height)
            if bbox is None:
                # Retry once with a more explicit prompt
                retry_prompt = (
                    f"I need the bounding box for: {region_description}\n"
                    f"Image dimensions: {info.width}x{info.height} pixels.\n"
                    f"Reply with ONLY a JSON object like: "
                    f'{{"x": 100, "y": 200, "x2": 400, "y2": 500}}\n'
                    f"No explanation, just the JSON."
                )
                bbox_response = toolkit._llm_client.vision_completion(
                    prompt=retry_prompt,
                    frame_paths=[info.path],
                )
                _record_vision_budget(toolkit._llm_client, toolkit._budget)
                bbox = _parse_bbox_response(bbox_response, info.width, info.height)

            if bbox is None:
                return {
                    "error": "Could not determine bounding box from vision model response.",
                    "raw_response": bbox_response[:500],
                }

            # Crop and save
            toolkit._crop_counter += 1
            crop_out = toolkit._workspace.crop_path(image_id, toolkit._crop_counter)
            crop_image(info.path, bbox, str(crop_out))

            crop_id = f"{image_id}_crop_{toolkit._crop_counter:03d}"
            crop_info = {
                "crop_id": crop_id,
                "image_id": image_id,
                "crop_path": str(crop_out),
                "bbox": {"x": bbox[0], "y": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                "region": region_description,
            }
            toolkit._crops[crop_id] = crop_info

            # Register crop as a queryable image so vision_query(image_id=crop_id) works
            from PIL import Image as _PILImage

            cropped_img = _PILImage.open(str(crop_out))
            toolkit._images[crop_id] = ImageInfo(
                path=str(crop_out),
                image_id=crop_id,
                width=cropped_img.size[0],
                height=cropped_img.size[1],
                format="JPEG",
                mode=cropped_img.mode,
                file_size_bytes=crop_out.stat().st_size,
            )

            return crop_info

        return Tool(
            name="crop_region",
            description=(
                "Crop a described region from an image for closer inspection. "
                "Uses the vision model to locate the region, then crops it. "
                "The cropped image path can be used with vision_query(frame_paths=[crop_path])."
            ),
            fn=_crop_region,
            parameters={
                "image_id": ToolParam(
                    name="image_id",
                    type_hint="str",
                    description="Image ID from list_images().",
                ),
                "region_description": ToolParam(
                    name="region_description",
                    type_hint="str",
                    description="Natural language description of the region to crop.",
                ),
            },
            return_type="dict",
        )

    def _make_search_images_tool(self) -> Tool:
        toolkit = self

        def _search_images(query: str, top_k: int = 5) -> list[dict]:
            """Search across image captions by keyword.

            On first call, all images are captioned via the vision model
            and indexed for search. Subsequent calls reuse the cached captions.

            Args:
                query: Search query.
                top_k: Max results to return.
            """
            if not toolkit._images:
                return [{"error": "No images loaded."}]

            # Lazy captioning: caption any images not yet captioned
            uncaptioned = [
                iid for iid in toolkit._images
                if iid not in toolkit._captions
            ]

            if uncaptioned and toolkit._llm_client is not None:
                # Prefer dedicated captioner (moondream) when available — cheaper
                moondream = toolkit._captioner or getattr(toolkit._llm_client, "_moondream", None)
                if moondream is not None:
                    paths = [toolkit._images[iid].path for iid in uncaptioned]
                    tokens_before = getattr(moondream, "total_input_tokens", 0)
                    out_before = getattr(moondream, "total_output_tokens", 0)
                    captions = moondream.caption_frames_batch(paths)
                    in_delta = getattr(moondream, "total_input_tokens", 0) - tokens_before
                    out_delta = getattr(moondream, "total_output_tokens", 0) - out_before
                    for iid, caption in zip(uncaptioned, captions):
                        toolkit._captions[iid] = caption
                        toolkit._queried_images.add(iid)
                    if toolkit._budget is not None:
                        from ...llm.pricing import moondream_cost

                        toolkit._budget.record(
                            input_tokens=in_delta,
                            output_tokens=out_delta,
                            cost_usd=moondream_cost(in_delta, out_delta),
                            model=getattr(moondream, "model_name", "moondream"),
                        )
                else:
                    for iid in uncaptioned:
                        info = toolkit._images[iid]
                        caption = toolkit._llm_client.vision_completion(
                            prompt=_CAPTION_PROMPT,
                            frame_paths=[info.path],
                        )
                        _record_vision_budget(toolkit._llm_client, toolkit._budget)
                        toolkit._captions[iid] = caption
                        toolkit._queried_images.add(iid)

                # Rebuild FTS index with all captions
                toolkit._fts = SQLiteFTSBackend(path=":memory:")
                toolkit._fts.index(
                    documents=[toolkit._captions[iid] for iid in toolkit._images if iid in toolkit._captions],
                    metadata=[{"image_id": iid} for iid in toolkit._images if iid in toolkit._captions],
                    collection="images",
                )
                toolkit._captions_indexed = True

            if toolkit._fts is None:
                return [{"error": "No captions indexed — vision model may not be configured."}]

            results = toolkit._fts.search(query, top_k=top_k, collection="images")
            return [
                {
                    "image_id": r["metadata"]["image_id"],
                    "caption": r["text"][:300],
                    "score": round(r["score"], 4),
                }
                for r in results
            ]

        return Tool(
            name="search_images",
            description=(
                "Search across image captions by keyword/phrase. "
                "Lazily captions all images on first call via the vision model, "
                "then uses BM25 keyword search."
            ),
            fn=_search_images,
            parameters={
                "query": ToolParam(
                    name="query",
                    type_hint="str",
                    description="Search query.",
                ),
                "top_k": ToolParam(
                    name="top_k",
                    type_hint="int",
                    default=5,
                    description="Max results to return.",
                ),
            },
            return_type="list[dict]",
        )

    def _make_compare_images_tool(self) -> Tool:
        toolkit = self

        def _compare_images(
            image_id_1: str,
            image_id_2: str,
            prompt: str | None = None,
        ) -> str:
            """Compare two images side by side using a vision model.

            Args:
                image_id_1: First image ID.
                image_id_2: Second image ID.
                prompt: Custom comparison prompt (default: general comparison).
            """
            if toolkit._llm_client is None:
                raise RuntimeError("Vision model not configured.")

            info1 = toolkit._images.get(image_id_1)
            info2 = toolkit._images.get(image_id_2)
            if info1 is None:
                raise ValueError(
                    f"Unknown image_id: {image_id_1}. "
                    f"Available: {list(toolkit._images.keys())}"
                )
            if info2 is None:
                raise ValueError(
                    f"Unknown image_id: {image_id_2}. "
                    f"Available: {list(toolkit._images.keys())}"
                )

            effective_prompt = prompt or (
                "Compare these two images carefully. Report: "
                "(1) shared elements, (2) key differences, "
                "(3) any text differences, and (4) what changed overall."
            )

            toolkit._queried_images.add(image_id_1)
            toolkit._queried_images.add(image_id_2)

            tracer = toolkit._tracer
            if tracer:
                vision_model = getattr(toolkit._llm_client, "vision_model", "unknown")
                model_label = (
                    vision_model
                    if isinstance(vision_model, str)
                    else getattr(vision_model, "model_name", "unknown")
                )
                with tracer._span(
                    "sanjaya.sub_llm_call.image_compare",
                    model=model_label,
                    image_ids=[image_id_1, image_id_2],
                ) as ctx:
                    result = toolkit._llm_client.vision_completion(
                        prompt=effective_prompt,
                        frame_paths=[info1.path, info2.path],
                    )
                    ctx.record_response(result)
                    _record_vision_budget(toolkit._llm_client, toolkit._budget)
                    return result
            else:
                result = toolkit._llm_client.vision_completion(
                    prompt=effective_prompt,
                    frame_paths=[info1.path, info2.path],
                )
                _record_vision_budget(toolkit._llm_client, toolkit._budget)
                return result

        return Tool(
            name="compare_images",
            description=(
                "Compare two images side by side using a vision model. "
                "Returns a text description of similarities and differences."
            ),
            fn=_compare_images,
            parameters={
                "image_id_1": ToolParam(
                    name="image_id_1",
                    type_hint="str",
                    description="First image ID from list_images().",
                ),
                "image_id_2": ToolParam(
                    name="image_id_2",
                    type_hint="str",
                    description="Second image ID from list_images().",
                ),
                "prompt": ToolParam(
                    name="prompt",
                    type_hint="str | None",
                    default=None,
                    description="Custom comparison prompt.",
                ),
            },
            return_type="str",
        )


def _parse_bbox_response(
    response: str,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int] | None:
    """Try to extract a bounding box from a vision model response.

    Looks for JSON with x, y, x2, y2 keys. Returns (x, y, x2, y2) or None.
    """
    # Try to find JSON in the response
    text = response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[^}]+\}", text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict):
        return None

    try:
        x = int(data["x"])
        y = int(data["y"])
        x2 = int(data["x2"])
        y2 = int(data["y2"])
    except (KeyError, ValueError, TypeError):
        return None

    # Sanity check: ensure bbox is within image bounds and valid
    if x >= x2 or y >= y2:
        return None
    if x2 <= 0 or y2 <= 0:
        return None
    if x >= img_width or y >= img_height:
        return None

    return (x, y, x2, y2)
