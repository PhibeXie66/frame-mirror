const state = {
  ready: false,
  searching: false,
  analyzing: false,
  selected: new Map(),
  latestQuery: "",
  productFile: null,
  productPreviewUrl: "",
  product: null,
  brief: null,
  generationConfig: null,
  generationJobs: new Map(),
  generationVersions: [],
  currentJobId: null,
  selectedGeneratedIndex: 0,
  agentHistory: [],
  agentReference: null,
  agentDraft: null,
  manualLayout: { scale: 62, x: 0, y: 0 },
  layoutPreviewTimer: null,
  layoutPreviewAbort: null,
  layoutPreviewUrl: "",
  motionJobId: null,
  motionPlan: null,
};

const els = {
  status: document.querySelector("#statusPill"),
  form: document.querySelector("#searchForm"),
  query: document.querySelector("#queryInput"),
  button: document.querySelector("#searchButton"),
  topK: document.querySelector("#topK"),
  yearFrom: document.querySelector("#yearFrom"),
  yearTo: document.querySelector("#yearTo"),
  gallery: document.querySelector("#gallery"),
  empty: document.querySelector("#emptyState"),
  loading: document.querySelector("#loadingState"),
  meta: document.querySelector("#resultMeta"),
  dock: document.querySelector("#selectionDock"),
  count: document.querySelector("#selectedCount"),
  thumbs: document.querySelector("#selectedThumbs"),
  export: document.querySelector("#exportButton"),
  toast: document.querySelector("#toast"),
  productFile: document.querySelector("#productFile"),
  productPreview: document.querySelector("#productPreview"),
  dropZone: document.querySelector("#dropZone"),
  dropCopy: document.querySelector("#dropCopy"),
  productName: document.querySelector("#productName"),
  productDescription: document.querySelector("#productDescription"),
  analyze: document.querySelector("#analyzeButton"),
  analysisResult: document.querySelector("#analysisResult"),
  categoryChips: document.querySelector("#categoryChips"),
  briefSection: document.querySelector("#briefSection"),
  briefProductImage: document.querySelector("#briefProductImage"),
  briefProductName: document.querySelector("#briefProductName"),
  briefProductMeta: document.querySelector("#briefProductMeta"),
  briefDirection: document.querySelector("#briefDirection"),
  briefPalette: document.querySelector("#briefPalette"),
  briefComposition: document.querySelector("#briefComposition"),
  briefPrompt: document.querySelector("#briefPrompt"),
  copyPrompt: document.querySelector("#copyPromptButton"),
  downloadBrief: document.querySelector("#downloadBriefButton"),
  generationEngineStatus: document.querySelector("#generationEngineStatus"),
  generateImages: document.querySelector("#generateImagesButton"),
  generationProgress: document.querySelector("#generationProgress"),
  generationProgressLabel: document.querySelector("#generationProgressLabel"),
  generationProgressValue: document.querySelector("#generationProgressValue"),
  generationProgressBar: document.querySelector("#generationProgressBar"),
  generationResults: document.querySelector("#generationResults"),
  generatedGrid: document.querySelector("#generatedGrid"),
  versionHistory: document.querySelector("#versionHistory"),
  revisionStudio: document.querySelector("#revisionStudio"),
  revisionPreview: document.querySelector("#revisionPreview"),
  revisionInput: document.querySelector("#revisionInput"),
  revise: document.querySelector("#reviseButton"),
  agentConversation: document.querySelector("#agentConversation"),
  agentReferenceFile: document.querySelector("#agentReferenceFile"),
  agentReferencePreview: document.querySelector("#agentReferencePreview"),
  agentReferenceImage: document.querySelector("#agentReferenceImage"),
  agentReferenceMeta: document.querySelector("#agentReferenceMeta"),
  agentPlan: document.querySelector("#agentPlan"),
  agentPlanMessage: document.querySelector("#agentPlanMessage"),
  agentPromptPreview: document.querySelector("#agentPromptPreview"),
  agentConfirm: document.querySelector("#agentConfirmButton"),
  agentGenerationProgress: document.querySelector("#agentGenerationProgress"),
  agentProgressLabel: document.querySelector("#agentProgressLabel"),
  agentProgressValue: document.querySelector("#agentProgressValue"),
  agentProgressBar: document.querySelector("#agentProgressBar"),
  agentProgressStage: document.querySelector("#agentProgressStage"),
  layoutScale: document.querySelector("#layoutScale"),
  layoutX: document.querySelector("#layoutX"),
  layoutY: document.querySelector("#layoutY"),
  layoutScaleValue: document.querySelector("#layoutScaleValue"),
  layoutXValue: document.querySelector("#layoutXValue"),
  layoutYValue: document.querySelector("#layoutYValue"),
  layoutHint: document.querySelector("#layoutHint"),
  layoutApply: document.querySelector("#layoutApplyButton"),
  layoutReset: document.querySelector("#layoutResetButton"),
  motionStyle: document.querySelector("#motionStyle"),
  motionDuration: document.querySelector("#motionDuration"),
  motionFormat: document.querySelector("#motionFormat"),
  motionPrompt: document.querySelector("#motionPrompt"),
  motionPlanButton: document.querySelector("#motionPlanButton"),
  motionConversation: document.querySelector("#motionConversation"),
  motionPlan: document.querySelector("#motionPlan"),
  motionPlanMessage: document.querySelector("#motionPlanMessage"),
  motionPlanChips: document.querySelector("#motionPlanChips"),
  motionGenerate: document.querySelector("#motionGenerateButton"),
  motionProgress: document.querySelector("#motionProgress"),
  motionProgressLabel: document.querySelector("#motionProgressLabel"),
  motionProgressValue: document.querySelector("#motionProgressValue"),
  motionProgressBar: document.querySelector("#motionProgressBar"),
  motionResult: document.querySelector("#motionResult"),
  motionVideo: document.querySelector("#motionVideo"),
  motionGif: document.querySelector("#motionGif"),
  motionResultMeta: document.querySelector("#motionResultMeta"),
  motionDownload: document.querySelector("#motionDownload"),
};

function setStatus(kind, text) {
  els.status.className = `status-pill ${kind || ""}`.trim();
  els.status.querySelector("span").textContent = text;
}

function showToast(message, duration = 3000) {
  els.toast.textContent = message;
  els.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => els.toast.classList.remove("visible"), duration);
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (data.ready) {
      state.ready = true;
      setStatus("ready", `${data.total.toLocaleString()} 张电影海报 · 视觉档案`);
      if (data.year_min) els.yearFrom.placeholder = data.year_min;
      if (data.year_max) els.yearTo.placeholder = data.year_max;
      els.button.disabled = false;
      return;
    }
    if (data.error) {
      setStatus("error", "档案载入失败");
      showToast(data.error, 6000);
      return;
    }
  } catch (error) {
    setStatus("error", "无法连接本地服务");
  }
  window.setTimeout(loadStatus, 1600);
}

async function loadGenerationStatus() {
  try {
    const response = await fetch("/api/generation/status");
    const data = await response.json();
    state.generationConfig = data;
    els.generationEngineStatus.textContent = data.message;
    updateGenerateButton();
  } catch (error) {
    state.generationConfig = { configured: false };
    els.generationEngineStatus.textContent = "无法读取图片生成引擎状态。";
    updateGenerateButton();
  }
}

function updateGenerateButton() {
  const configured = Boolean(state.generationConfig?.configured);
  const ready = configured && Boolean(state.brief);
  els.generateImages.disabled = !ready;
  if (!configured) {
    els.generateImages.textContent = "等待本地配置";
  } else if (!state.brief) {
    els.generateImages.textContent = "先完成创意方案";
  } else {
    els.generateImages.textContent = "生成 4 张广告海报 ↗";
  }
}

function selectProductFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    showToast("请选择 JPG、PNG 或 WebP 图片。");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showToast("图片不能超过 20 MB。");
    return;
  }
  if (state.productPreviewUrl) URL.revokeObjectURL(state.productPreviewUrl);
  state.productFile = file;
  state.product = null;
  state.productPreviewUrl = URL.createObjectURL(file);
  els.productPreview.src = state.productPreviewUrl;
  els.productPreview.hidden = false;
  els.dropCopy.hidden = true;
  els.analysisResult.hidden = true;
  els.analyze.disabled = false;
  els.analyze.querySelector("span").textContent = "识别商品主体";
}

async function analyzeProduct() {
  if (!state.productFile || state.analyzing) return;
  state.analyzing = true;
  els.analyze.disabled = true;
  els.analyze.querySelector("span").textContent = "正在载入 CLIP 并识别…";
  setStatus("", "正在识别商品主体");

  const body = new FormData();
  body.append("file", state.productFile);
  body.append("product_name", els.productName.value.trim());
  body.append("description", els.productDescription.value.trim());
  try {
    const response = await fetch("/api/product/analyze", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "商品识别失败");
    state.product = data;
    if (!els.productName.value.trim()) els.productName.value = data.name;
    renderCategoryChips(data.candidates, data.category);
    els.analysisResult.hidden = false;
    showToast(`已识别为“${data.category}”，你可以随时修改。`);
  } catch (error) {
    showToast(error.message, 5000);
  } finally {
    state.analyzing = false;
    els.analyze.disabled = false;
    els.analyze.querySelector("span").textContent = state.product ? "重新识别商品" : "识别商品主体";
    if (state.ready) setStatus("ready", "3,000 张电影海报 · 视觉档案");
  }
}

function renderCategoryChips(candidates, activeCategory) {
  els.categoryChips.replaceChildren();
  candidates.forEach((candidate) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = candidate.category === activeCategory ? "active" : "";
    button.textContent = `${candidate.category} · ${candidate.confidence.toFixed(2)}`;
    button.addEventListener("click", () => {
      state.product.category = candidate.category;
      els.categoryChips.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      if (!els.productName.value.trim()) els.productName.value = candidate.category;
    });
    els.categoryChips.appendChild(button);
  });
}

function currentPayload() {
  const yearFrom = Number.parseInt(els.yearFrom.value, 10);
  const yearTo = Number.parseInt(els.yearTo.value, 10);
  return {
    query: els.query.value.trim(),
    top_k: Number.parseInt(els.topK.value, 10),
    year_from: Number.isFinite(yearFrom) ? yearFrom : null,
    year_to: Number.isFinite(yearTo) ? yearTo : null,
  };
}

async function runSearch() {
  if (state.searching) return;
  const payload = currentPayload();
  if (!payload.query) {
    showToast("先写下一种你想寻找的感觉。");
    els.query.focus();
    return;
  }
  if (!state.ready) {
    showToast("档案仍在载入，请稍等片刻。");
    return;
  }
  state.searching = true;
  state.latestQuery = payload.query;
  els.button.disabled = true;
  els.loading.hidden = false;
  els.empty.hidden = true;
  els.gallery.replaceChildren();
  els.meta.textContent = "CLIP 正在寻找视觉近邻";
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "检索失败");
    renderGallery(data.items);
    els.meta.textContent = `${data.total} 张结果 · ${data.elapsed_ms} ms · ${data.query}`;
  } catch (error) {
    els.empty.hidden = false;
    els.empty.querySelector("p").textContent = error.message;
    els.meta.textContent = "检索未完成";
  } finally {
    state.searching = false;
    els.button.disabled = false;
    els.loading.hidden = true;
  }
}

function renderGallery(items) {
  els.gallery.replaceChildren();
  if (!items.length) {
    els.empty.hidden = false;
    els.empty.querySelector("p").textContent = "这个范围里没有找到海报，试试放宽年份。";
    return;
  }
  els.empty.hidden = true;
  const fragment = document.createDocumentFragment();
  items.forEach((item) => fragment.appendChild(createPosterCard(item)));
  els.gallery.appendChild(fragment);
}

function createPosterCard(item) {
  const card = document.createElement("article");
  card.className = "poster-card";
  card.dataset.id = item.vector_index;
  if (state.selected.has(item.vector_index)) card.classList.add("selected");
  const img = document.createElement("img");
  img.src = item.poster_url;
  img.alt = `${item.title} 电影海报`;
  img.loading = "lazy";
  const overlay = document.createElement("div");
  overlay.className = "poster-overlay";
  overlay.innerHTML = `<div class="poster-topline"><span class="poster-rank">${String(item.rank).padStart(2, "0")}</span><span class="poster-score">SIM ${item.similarity.toFixed(3)}</span></div><div class="poster-bottomline"><div><div class="poster-title"></div><div class="poster-year">${item.year ?? "—"} · VAR ${item.variant ?? "—"}</div></div>${item.detail_url ? `<a class="poster-link" href="${item.detail_url}" target="_blank" rel="noreferrer" aria-label="打开来源">↗</a>` : ""}</div>`;
  overlay.querySelector(".poster-title").textContent = item.title;
  const stamp = document.createElement("span");
  stamp.className = "selected-stamp";
  stamp.textContent = "SELECTED";
  card.append(img, overlay, stamp);
  card.addEventListener("click", (event) => {
    if (event.target.closest("a")) return;
    toggleSelection(item, card);
  });
  return card;
}

function toggleSelection(item, card) {
  if (state.selected.has(item.vector_index)) {
    state.selected.delete(item.vector_index);
    card.classList.remove("selected");
  } else {
    if (state.selected.size >= 4) {
      showToast("一张灵感板最多选择四张海报。");
      return;
    }
    state.selected.set(item.vector_index, item);
    card.classList.add("selected");
  }
  renderSelection();
}

function renderSelection() {
  const items = [...state.selected.values()];
  els.count.textContent = items.length;
  els.dock.classList.toggle("visible", items.length > 0);
  els.export.disabled = items.length === 0;
  els.thumbs.replaceChildren();
  items.forEach((item) => {
    const img = document.createElement("img");
    img.src = item.poster_url;
    img.alt = item.title;
    els.thumbs.appendChild(img);
  });
}

async function generateBrief() {
  if (!state.product) {
    showToast("请先上传并识别商品图片。", 4000);
    document.querySelector("#productStage").scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  const references = [...state.selected.values()].map((item) => ({
    vector_index: item.vector_index,
    title: item.title,
    year: item.year,
    similarity: item.similarity,
    detail_url: item.detail_url,
    poster_url: item.poster_url,
  }));
  if (!references.length) return;
  const payload = {
    style_query: state.latestQuery || els.query.value.trim(),
    product: {
      upload_id: state.product.upload_id,
      name: els.productName.value.trim() || state.product.name,
      category: state.product.category,
      description: els.productDescription.value.trim(),
    },
    references,
  };
  els.export.disabled = true;
  els.export.textContent = "正在生成方案…";
  try {
    const response = await fetch("/api/brief", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "方案生成失败");
    state.brief = data;
    renderBrief(data);
    els.dock.classList.remove("visible");
    els.briefSection.hidden = false;
    requestAnimationFrame(() => els.briefSection.scrollIntoView({ behavior: "smooth", block: "start" }));
  } catch (error) {
    showToast(error.message, 5000);
  } finally {
    els.export.disabled = false;
    els.export.textContent = "生成广告方案 ↗";
  }
}

function renderBrief(brief) {
  els.briefProductImage.src = state.product.image_url;
  els.briefProductName.textContent = brief.product.name;
  els.briefProductMeta.textContent = `${brief.product.category}${brief.product.description ? ` · ${brief.product.description}` : ""}`;
  els.briefDirection.textContent = brief.direction;
  els.briefPalette.replaceChildren();
  brief.palette.forEach((color) => {
    const chip = document.createElement("span");
    chip.textContent = color;
    els.briefPalette.appendChild(chip);
  });
  els.briefComposition.replaceChildren();
  brief.composition.forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    els.briefComposition.appendChild(li);
  });
  els.briefPrompt.textContent = brief.generation_prompt;
  updateGenerateButton();
}

function generationPayload() {
  return {
    prompt: state.brief.generation_prompt,
    style_query: state.brief.direction,
    product: state.brief.product,
    references: state.brief.references,
    variant_count: 4,
    width: 1024,
    height: 1365,
  };
}

function setGenerationProgress(progress, message) {
  els.generationProgress.hidden = false;
  els.generationProgressLabel.textContent = message;
  els.generationProgressValue.textContent = `${progress}%`;
  els.generationProgressBar.style.width = `${progress}%`;
}

function setAgentGenerationProgress(progress, message, status = "generating") {
  const safeProgress = Math.max(0, Math.min(100, Number(progress) || 0));
  els.agentGenerationProgress.hidden = false;
  els.agentGenerationProgress.dataset.status = status;
  els.agentProgressLabel.textContent = message;
  els.agentProgressValue.textContent = `${safeProgress}%`;
  els.agentProgressBar.style.width = `${safeProgress}%`;
  if (status === "failed") {
    els.agentProgressStage.textContent = "生成失败，请根据提示修改后重试。";
  } else if (status === "completed") {
    els.agentProgressStage.textContent = "新版本已生成，正在更新海报画廊。";
  } else if (safeProgress < 8) {
    els.agentProgressStage.textContent = "正在分析参考海报的场景、构图与创意元素。";
  } else if (safeProgress < 88) {
    els.agentProgressStage.textContent = "正在生成新的电影感背景，请保持页面开启。";
  } else {
    els.agentProgressStage.textContent = "背景已完成，正在融合原始商品与精确文字。";
  }
}

async function startGeneration() {
  if (!state.brief || !state.generationConfig?.configured) {
    showToast(state.generationConfig?.message || "图片生成引擎尚未配置。", 5000);
    return;
  }
  els.generateImages.disabled = true;
  els.generateImages.textContent = "正在创建任务…";
  setGenerationProgress(2, "正在创建生成任务");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(generationPayload()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "无法创建生成任务");
    state.currentJobId = data.job_id;
    await pollGenerationJob(data.job_id);
  } catch (error) {
    setGenerationProgress(100, "生成任务未能启动");
    showToast(error.message, 6000);
  } finally {
    updateGenerateButton();
  }
}

async function pollGenerationJob(jobId, options = {}) {
  const showAgentProgress = Boolean(options.agentProgress);
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "无法读取生成进度");
    setGenerationProgress(job.progress, job.message);
    if (showAgentProgress) setAgentGenerationProgress(job.progress, job.message, job.status);
    if (job.status === "completed") {
      state.generationJobs.set(job.job_id, job);
      if (!state.generationVersions.includes(job.job_id)) state.generationVersions.push(job.job_id);
      state.currentJobId = job.job_id;
      state.selectedGeneratedIndex = 0;
      renderGenerationJob(job);
      if (showAgentProgress) setAgentGenerationProgress(100, "新版本生成完成", "completed");
      showToast("新一轮广告海报已经生成完成。", 4200);
      return;
    }
    if (job.status === "failed") {
      if (showAgentProgress) setAgentGenerationProgress(100, job.error || job.message || "图片生成失败", "failed");
      throw new Error(job.error || job.message || "图片生成失败");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1400));
  }
}

function renderGenerationJob(job, scrollToResults = true) {
  els.generationResults.hidden = false;
  els.generatedGrid.replaceChildren();
  job.outputs.forEach((output, index) => {
    const card = document.createElement("article");
    card.className = `generated-card${index === state.selectedGeneratedIndex ? " selected" : ""}`;
    card.dataset.index = index;
    const img = document.createElement("img");
    img.src = `${output.image_url}?v=${Date.now()}`;
    img.alt = `生成的广告海报方案 ${index + 1}`;
    const meta = document.createElement("div");
    meta.className = "generated-card-meta";
    meta.innerHTML = `<span>FRAME ${String(index + 1).padStart(2, "0")}</span><strong>${job.revision_instruction ? "修改版本" : "初始提案"}</strong>`;
    const download = document.createElement("a");
    download.href = output.download_url;
    download.textContent = "下载 ↘";
    download.addEventListener("click", (event) => event.stopPropagation());
    meta.appendChild(download);
    card.append(img, meta);
    card.addEventListener("click", () => selectGeneratedFrame(job, index));
    els.generatedGrid.appendChild(card);
  });
  renderVersionHistory();
  selectGeneratedFrame(job, state.selectedGeneratedIndex);
  if (scrollToResults) {
    requestAnimationFrame(() => els.generationResults.scrollIntoView({ behavior: "smooth", block: "start" }));
  }
}

async function restoreRecentGeneration() {
  try {
    const response = await fetch("/api/generated/recent", { cache: "no-store" });
    if (!response.ok) return;
    const data = await response.json();
    const job = data.job;
    if (!job?.outputs?.length) return;
    state.generationJobs.set(job.job_id, job);
    state.generationVersions = [job.job_id];
    state.currentJobId = job.job_id;
    state.selectedGeneratedIndex = 0;
    els.briefSection.hidden = false;
    renderGenerationJob(job, false);
    setGenerationProgress(100, job.message);
  } catch (error) {
    console.warn("无法恢复最近生成结果", error);
  }
}

function selectGeneratedFrame(job, index) {
  clearLayoutPreview();
  state.currentJobId = job.job_id;
  state.selectedGeneratedIndex = index;
  els.generatedGrid.querySelectorAll(".generated-card").forEach((card, cardIndex) => {
    card.classList.toggle("selected", cardIndex === index);
  });
  els.revisionPreview.src = job.outputs[index].image_url;
  els.revisionStudio.hidden = false;
  els.motionGenerate.disabled = !state.motionPlan;
  syncLayoutControls(job, index);
}

function positionLabel(value, negativeLabel, positiveLabel) {
  const amount = Math.abs(Number(value));
  if (!amount) return "居中";
  return `${value < 0 ? negativeLabel : positiveLabel} ${amount}%`;
}

function updateLayoutLabels() {
  state.manualLayout = {
    scale: Number(els.layoutScale.value),
    x: Number(els.layoutX.value),
    y: Number(els.layoutY.value),
  };
  els.layoutScaleValue.textContent = `${state.manualLayout.scale}%`;
  els.layoutXValue.textContent = positionLabel(state.manualLayout.x, "向左", "向右");
  els.layoutYValue.textContent = positionLabel(state.manualLayout.y, "向上", "向下");
}

function clearLayoutPreview() {
  window.clearTimeout(state.layoutPreviewTimer);
  state.layoutPreviewTimer = null;
  state.layoutPreviewAbort?.abort();
  state.layoutPreviewAbort = null;
  if (state.layoutPreviewUrl) URL.revokeObjectURL(state.layoutPreviewUrl);
  state.layoutPreviewUrl = "";
  els.revisionPreview.classList.remove("is-previewing");
}

async function previewManualLayout() {
  if (!state.currentJobId || els.layoutApply.disabled) return;
  updateLayoutLabels();
  const jobId = state.currentJobId;
  const sourceIndex = state.selectedGeneratedIndex;
  state.layoutPreviewAbort?.abort();
  const controller = new AbortController();
  state.layoutPreviewAbort = controller;
  els.revisionPreview.classList.add("is-previewing");
  els.layoutHint.textContent = "正在实时预览商品大小与位置……";
  try {
    const response = await fetch(`/api/jobs/${jobId}/layout-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        source_index: sourceIndex,
        scale: state.manualLayout.scale / 100,
        offset_x: state.manualLayout.x / 100,
        offset_y: state.manualLayout.y / 100,
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "无法生成实时预览");
    }
    const blob = await response.blob();
    if (controller.signal.aborted || jobId !== state.currentJobId || sourceIndex !== state.selectedGeneratedIndex) return;
    if (state.layoutPreviewUrl) URL.revokeObjectURL(state.layoutPreviewUrl);
    state.layoutPreviewUrl = URL.createObjectURL(blob);
    els.revisionPreview.src = state.layoutPreviewUrl;
    els.layoutHint.textContent = "实时预览已更新；满意后保存为一个新版本。";
  } catch (error) {
    if (error.name !== "AbortError") els.layoutHint.textContent = error.message;
  } finally {
    if (state.layoutPreviewAbort === controller) state.layoutPreviewAbort = null;
    els.revisionPreview.classList.remove("is-previewing");
  }
}

function scheduleLayoutPreview() {
  window.clearTimeout(state.layoutPreviewTimer);
  state.layoutPreviewTimer = window.setTimeout(previewManualLayout, 180);
}

function syncLayoutControls(job, index) {
  const layout = job.layout || { scale: 0.62, offset_x: 0, offset_y: 0 };
  els.layoutScale.value = Math.round((layout.scale ?? 0.62) * 100);
  els.layoutX.value = Math.round((layout.offset_x ?? 0) * 100);
  els.layoutY.value = Math.round((layout.offset_y ?? 0) * 100);
  updateLayoutLabels();
  const editable = (job.layout_editable_indices || []).includes(index);
  els.layoutApply.disabled = !editable;
  els.layoutHint.textContent = editable
    ? "拖动滑杆即可实时预览；保存时不调用 AI，也不会在背景里再画商品。"
    : "这个旧版本没有保存纯背景层，请先用新版流程重新生成一版。";
}

function resetLayoutControls() {
  els.layoutScale.value = 62;
  els.layoutX.value = 0;
  els.layoutY.value = 0;
  updateLayoutLabels();
  scheduleLayoutPreview();
}

async function applyManualLayout() {
  if (!state.currentJobId || els.layoutApply.disabled) return;
  updateLayoutLabels();
  window.clearTimeout(state.layoutPreviewTimer);
  state.layoutPreviewAbort?.abort();
  els.layoutApply.disabled = true;
  els.layoutApply.textContent = "正在保存版本…";
  try {
    const response = await fetch(`/api/jobs/${state.currentJobId}/layout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_index: state.selectedGeneratedIndex,
        scale: state.manualLayout.scale / 100,
        offset_x: state.manualLayout.x / 100,
        offset_y: state.manualLayout.y / 100,
      }),
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "无法调整商品布局");
    state.generationJobs.set(job.job_id, job);
    if (!state.generationVersions.includes(job.job_id)) state.generationVersions.push(job.job_id);
    state.currentJobId = job.job_id;
    state.selectedGeneratedIndex = 0;
    renderGenerationJob(job);
    showToast("商品大小与位置已在本地调整完成。", 4200);
  } catch (error) {
    showToast(error.message, 6000);
  } finally {
    els.layoutApply.textContent = "保存布局版本 ↗";
    const currentJob = state.generationJobs.get(state.currentJobId);
    if (currentJob) syncLayoutControls(currentJob, state.selectedGeneratedIndex);
  }
}

function setMotionProgress(progress, message) {
  const value = Math.max(0, Math.min(100, Number(progress) || 0));
  els.motionProgress.hidden = false;
  els.motionProgressBar.style.width = `${value}%`;
  els.motionProgressValue.textContent = `${Math.round(value)}%`;
  els.motionProgressLabel.textContent = message || "正在生成动态广告";
}

function appendMotionMessage(role, text) {
  const article = document.createElement("article");
  article.className = `motion-message ${role}`;
  const label = document.createElement("span");
  label.textContent = role === "user" ? "YOU" : "MOTION DIRECTOR";
  const copy = document.createElement("p");
  copy.textContent = text;
  article.append(label, copy);
  els.motionConversation.append(article);
  els.motionConversation.scrollTop = els.motionConversation.scrollHeight;
}

async function planMotion() {
  const message = els.motionPrompt.value.trim();
  if (!message) {
    showToast("先说一句你希望视频怎样动起来。", 4200);
    els.motionPrompt.focus();
    return;
  }
  els.motionPlanButton.disabled = true;
  els.motionPlanButton.textContent = "Agent 正在设计…";
  appendMotionMessage("user", message);
  try {
    const response = await fetch("/api/motion/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const plan = await response.json();
    if (!response.ok) throw new Error(plan.detail || "无法整理动画方案");
    state.motionPlan = plan;
    els.motionStyle.value = plan.style;
    els.motionDuration.value = String(plan.duration);
    els.motionPlanMessage.textContent = plan.assistant_message;
    els.motionPlanChips.replaceChildren();
    [plan.summary, `强度：${plan.intensity === "high" ? "明显" : plan.intensity === "low" ? "克制" : "适中"}`, "本地免费渲染"].forEach((value) => {
      const chip = document.createElement("span");
      chip.textContent = value;
      els.motionPlanChips.append(chip);
    });
    els.motionPlan.hidden = false;
    appendMotionMessage("agent", plan.assistant_message);
    els.motionGenerate.disabled = !state.currentJobId;
    showToast("Motion Agent 已经整理好动画方案。", 3500);
  } catch (error) {
    appendMotionMessage("agent", `方案整理失败：${error.message}`);
    showToast(error.message, 6000);
  } finally {
    els.motionPlanButton.disabled = false;
    els.motionPlanButton.textContent = "让 Agent 设计 ↗";
  }
}

async function pollMotionJob(motionJobId) {
  while (true) {
    const response = await fetch(`/api/motion-jobs/${motionJobId}`, { cache: "no-store" });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "无法读取动态广告进度");
    setMotionProgress(job.progress, job.message);
    if (job.status === "completed") return job;
    if (job.status === "failed") throw new Error(job.error || job.message || "动态广告生成失败");
    await new Promise((resolve) => window.setTimeout(resolve, 650));
  }
}

async function startMotionGeneration() {
  if (!state.currentJobId) {
    showToast("请先选择一张已经生成的海报。", 4200);
    return;
  }
  if (!state.motionPlan) {
    showToast("请先让 Motion Agent 整理动画方案。", 4200);
    return;
  }
  els.motionGenerate.disabled = true;
  els.motionGenerate.textContent = "正在生成动态广告…";
  els.motionResult.hidden = true;
  setMotionProgress(0, "正在创建本地动态任务");
  try {
    const response = await fetch(`/api/jobs/${state.currentJobId}/motion`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_index: state.selectedGeneratedIndex,
        duration: Number(els.motionDuration.value),
        style: els.motionStyle.value,
        format: els.motionFormat.value,
        prompt: state.motionPlan.prompt || els.motionPrompt.value.trim(),
      }),
    });
    const created = await response.json();
    if (!response.ok) throw new Error(created.detail || "无法创建动态广告任务");
    state.motionJobId = created.job_id;
    const job = await pollMotionJob(created.job_id);
    const mediaUrl = `${job.file_url}?v=${Date.now()}`;
    const styleLabels = { push: "缓慢推进", pull: "逐渐拉远", drift: "水平漂移", breathe: "电影呼吸" };
    els.motionVideo.pause();
    els.motionVideo.hidden = true;
    els.motionGif.hidden = true;
    if (job.format === "mp4") {
      els.motionVideo.src = mediaUrl;
      els.motionVideo.hidden = false;
      els.motionVideo.load();
      els.motionVideo.play().catch(() => {});
    } else {
      els.motionGif.src = mediaUrl;
      els.motionGif.hidden = false;
    }
    els.motionDownload.href = job.download_url;
    els.motionDownload.download = `frame-mirror-motion-${job.job_id}.${job.format}`;
    els.motionResultMeta.textContent = `${job.plan?.summary || styleLabels[job.style] || job.style} · ${job.duration} 秒 · ${job.format.toUpperCase()}`;
    els.motionResult.hidden = false;
    showToast("动态广告已经生成完成。", 4200);
  } catch (error) {
    setMotionProgress(100, "动态广告生成失败");
    showToast(error.message, 6000);
  } finally {
    els.motionGenerate.disabled = !state.motionPlan || !state.currentJobId;
    els.motionGenerate.textContent = "按方案生成视频 ↗";
  }
}

function renderVersionHistory() {
  els.versionHistory.replaceChildren();
  state.generationVersions.forEach((jobId, index) => {
    const job = state.generationJobs.get(jobId);
    if (!job) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = jobId === state.currentJobId ? "active" : "";
    button.textContent = `V${index + 1}${job.revision_instruction ? " · 修改" : " · 初稿"}`;
    button.addEventListener("click", () => {
      state.currentJobId = jobId;
      state.selectedGeneratedIndex = 0;
      renderGenerationJob(job);
    });
    els.versionHistory.appendChild(button);
  });
}

function appendAgentMessage(role, content, questions = []) {
  const article = document.createElement("article");
  article.className = `agent-message ${role}`;
  const label = document.createElement("span");
  label.textContent = role === "assistant" ? "AGENT" : "YOU";
  const copy = document.createElement("p");
  copy.textContent = content;
  article.append(label, copy);
  if (questions.length) {
    const list = document.createElement("ul");
    questions.forEach((question) => {
      const item = document.createElement("li");
      item.textContent = question;
      list.appendChild(item);
    });
    article.appendChild(list);
  }
  els.agentConversation.appendChild(article);
  els.agentConversation.scrollTop = els.agentConversation.scrollHeight;
}

async function uploadAgentReference() {
  const file = els.agentReferenceFile.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  els.agentReferenceMeta.textContent = "正在分析参考图的色彩与光线……";
  els.agentReferencePreview.hidden = false;
  try {
    const response = await fetch("/api/agent/reference", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "参考图上传失败");
    state.agentReference = data;
    els.agentReferenceImage.src = data.image_url;
    const analysis = data.analysis;
    els.agentReferenceMeta.textContent = `${analysis.orientation} · ${analysis.lighting} · ${analysis.saturation} · ${analysis.palette.join(" / ")}`;
    appendAgentMessage("assistant", "参考图已收到。请告诉我你想借用它的色彩、光线、构图、材质，还是某个具体元素。");
  } catch (error) {
    state.agentReference = null;
    els.agentReferencePreview.hidden = true;
    showToast(error.message, 6000);
  }
}

async function reviseGeneration() {
  const message = els.revisionInput.value.trim();
  if (!message) {
    showToast("请先告诉 Agent 需要修改什么。");
    els.revisionInput.focus();
    return;
  }
  if (!state.currentJobId) return;
  els.agentGenerationProgress.hidden = true;
  els.revise.disabled = true;
  els.revise.textContent = "Agent 正在理解…";
  const priorHistory = state.agentHistory.slice(-12);
  state.agentHistory.push({ role: "user", content: message });
  appendAgentMessage("user", message);
  els.revisionInput.value = "";
  try {
    const response = await fetch("/api/agent/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history: priorHistory,
        current_prompt: state.brief?.generation_prompt || "",
        product_name: state.product?.name || state.brief?.product?.name || "商品",
        product_category: state.product?.category || state.brief?.product?.category || "",
        reference_id: state.agentReference?.reference_id || null,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Agent 暂时无法理解这条要求");
    const assistantText = data.assistant_message || "我已经理解。";
    state.agentHistory.push({ role: "assistant", content: [assistantText, ...(data.questions || [])].join("\n") });
    appendAgentMessage("assistant", assistantText, data.questions || []);
    if (data.status === "ready") {
      state.agentDraft = data;
      els.agentPlan.hidden = false;
      const referenceNote = data.reference_usage
        ? `参考图使用方式：${data.reference_usage}`
        : "商品保持原样，仅修改已确认的视觉内容。";
      const type = data.typography;
      const positionLabel = type?.position === "bottom" ? "底部" : "顶部";
      const typographyNote = type?.enabled
        ? ` 精确排版：${positionLabel} / “${type.headline || type.subtitle}” / ${type.style}。`
        : "";
      els.agentPlanMessage.textContent = `${referenceNote}${typographyNote}`;
      els.agentPromptPreview.value = data.generation_prompt || "";
      els.agentPromptPreview.focus();
    } else {
      state.agentDraft = null;
      els.agentPlan.hidden = true;
      els.revisionInput.focus();
    }
  } catch (error) {
    showToast(error.message, 6000);
  } finally {
    els.revise.disabled = false;
    els.revise.textContent = "发送给 Agent ↗";
  }
}

async function confirmAgentRevision() {
  const prompt = els.agentPromptPreview.value.trim();
  if (!prompt || !state.currentJobId) return;
  els.agentConfirm.disabled = true;
  els.agentConfirm.textContent = "正在创建修改任务…";
  setGenerationProgress(2, "已确认方案，正在创建新版本");
  setAgentGenerationProgress(2, "已确认方案，正在创建新版本", "queued");
  const scaleAction = state.agentDraft?.product_scale_action || "none";
  const instruction = prompt;
  try {
    const response = await fetch(`/api/jobs/${state.currentJobId}/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_index: state.selectedGeneratedIndex,
        instruction,
        variant_count: 2,
        typography: state.agentDraft?.typography || null,
        product_scale_action: scaleAction,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "无法创建修改任务");
    els.agentPlan.hidden = true;
    setAgentGenerationProgress(4, "任务已创建，正在分析参考海报", "analyzing");
    await pollGenerationJob(data.job_id, { agentProgress: true });
  } catch (error) {
    setAgentGenerationProgress(100, error.message || "生成任务失败", "failed");
    showToast(error.message, 6000);
  } finally {
    els.agentConfirm.disabled = false;
    els.agentConfirm.textContent = "确认方案并生成 2 张 ↗";
  }
}

async function copyPrompt() {
  if (!state.brief) return;
  await navigator.clipboard.writeText(state.brief.generation_prompt);
  showToast("生成提示词已复制。");
}

function downloadBrief() {
  if (!state.brief) return;
  const blob = new Blob([JSON.stringify(state.brief, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `frame-mirror-creative-brief-${Date.now()}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
  showToast("广告方案已下载。");
}

els.productFile.addEventListener("change", () => selectProductFile(els.productFile.files[0]));
['dragenter', 'dragover'].forEach((eventName) => els.dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  els.dropZone.classList.add("dragover");
}));
['dragleave', 'drop'].forEach((eventName) => els.dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  els.dropZone.classList.remove("dragover");
}));
els.dropZone.addEventListener("drop", (event) => selectProductFile(event.dataTransfer.files[0]));
els.analyze.addEventListener("click", analyzeProduct);
els.form.addEventListener("submit", (event) => { event.preventDefault(); runSearch(); });
document.querySelectorAll("[data-query]").forEach((button) => button.addEventListener("click", () => {
  els.query.value = button.dataset.query;
  runSearch();
}));
els.export.addEventListener("click", generateBrief);
els.copyPrompt.addEventListener("click", copyPrompt);
els.downloadBrief.addEventListener("click", downloadBrief);
els.generateImages.addEventListener("click", startGeneration);
els.revise.addEventListener("click", reviseGeneration);
els.agentConfirm.addEventListener("click", confirmAgentRevision);
els.agentReferenceFile.addEventListener("change", uploadAgentReference);
[els.layoutScale, els.layoutX, els.layoutY].forEach((input) => input.addEventListener("input", () => {
  updateLayoutLabels();
  scheduleLayoutPreview();
}));
els.layoutReset.addEventListener("click", resetLayoutControls);
els.layoutApply.addEventListener("click", applyManualLayout);
els.motionPlanButton.addEventListener("click", planMotion);
els.motionGenerate.addEventListener("click", startMotionGeneration);
document.querySelectorAll("[data-motion-prompt]").forEach((button) => button.addEventListener("click", () => {
  els.motionPrompt.value = button.dataset.motionPrompt;
  els.motionPrompt.focus();
}));
document.querySelectorAll("[data-revision]").forEach((button) => button.addEventListener("click", () => {
  els.revisionInput.value = button.dataset.revision;
  els.revisionInput.focus();
}));

els.button.disabled = true;
els.motionGenerate.disabled = true;
loadStatus();
loadGenerationStatus();
restoreRecentGeneration();
