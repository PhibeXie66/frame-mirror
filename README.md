# 帧镜影像 — 电影感商业广告工作室

一个以电影海报为视觉档案的本地商业影像工作台：上传商品图、识别主体、检索参考、生成广告海报，并通过对话继续调整画面和制作动态版本。它不会重新分析 3,000 张海报；每次搜索只编码用户输入的中文描述。

> 仓库只包含程序代码。电影海报数据集、模型权重、商品上传图、生成结果和 API 密钥均不会提交到 GitHub。

## 数据位置

默认读取：

```text
E:\简历\movie_agent\impawards_posters
```

如果数据集在其他目录，启动前设置：

```powershell
$env:POSTER_DATASET_DIR = 'D:\your\impawards_posters'
```

## 安装与启动

在 PowerShell 中进入本目录：

```powershell
python -m pip install -r requirements.txt
.\run_web.ps1
```

然后访问：

```text
http://127.0.0.1:8000
```

第一次启动会载入本地缓存的中文 CLIP 文本模型，可能需要几十秒。终端出现 `Poster search ready: 3,000 vectors` 后即可检索。

## 当前功能

- 上传商品图（JPG / PNG / WebP，最大 20 MB）
- 用本地 CLIP 自动判断商品类别，并允许用户更正名称与描述
- 中文自然语言风格检索
- 真实 CLIP 余弦相似度排序
- 年份筛选和结果数量控制
- 电影海报瀑布流
- 最多选择四张参考图
- 用本地 CLIP 提取参考海报的场景类型、构图节奏与视觉母题，而不只提取颜色
- 把商品、风格与海报参考整合成商业广告创意方案
- 复制生成提示词或下载方案 JSON
- 通过 Cloudflare Workers AI 生成最多四张广告海报
- 选择任意生成结果并用自然语言继续修改
- Creative Agent 会在要求含糊时先追问，形成可编辑 Prompt，经用户确认后才生成
- 支持上传一张视觉参考图，并明确选择借用色彩、光线、构图、材质或具体元素
- 保存每一轮生成版本并下载 JPG
- 实时调整商品大小与位置，无需重新调用图片模型
- 用自然语言设计镜头运动，并在本地导出 MP4 或 GIF

## GitHub 安全说明

- `.env`、`Cloudflare.env` 和其他密钥文件已被忽略。
- `sam_b.pt` 等模型权重不会提交；程序首次使用时会按需下载。
- 海报数据集、商品图片、透明蒙版、生成海报和视频均不会提交。
- 克隆仓库后，请复制 `.env.example` 为 `.env`，再填写自己的配置。
- MP4/GIF 导出需要系统已安装 FFmpeg，并能在命令行中运行 `ffmpeg`。

## Cloudflare 图片生成

程序会优先读取项目目录中的 `.env`，也兼容本地文件：

```text
E:\简历\movie_agent\Cloudflare.env
```

配置格式：

```env
IMAGE_PROVIDER=cloudflare
CLOUDFLARE_ACCOUNT_ID=Cloudflare 控制台中的 32 位 Account ID
CLOUDFLARE_API_TOKEN=新建的 Workers AI Token
CLOUDFLARE_IMAGE_MODEL=@cf/black-forest-labs/flux-2-klein-4b
CLOUDFLARE_AGENT_MODEL=@cf/meta/llama-3.1-8b-instruct
```

API Token 需要 Workers AI Read 和 Workers AI Edit 权限。不要把真实 Token 提交到 GitHub；仓库中的 `.env.example` 只包含占位符。

为避免免费模型拦截品牌商品或重画错误标签，当前使用“混合生成”流程：

1. 本地从所选海报提取主色、场景、构图和创意母题。
2. Cloudflare 将参考海报的视觉语言转化为超现实、图形、环境叙事或材质光影背景，并避免默认生成普通室内展厅。
3. 本地先用 SAM 定位任意品类商品，再用背景 matting 模型生成透明蒙版并合成阴影，保证商品和标签不被 AI 重画。
4. 对话式修改会重新生成背景；“商品放大/缩小”会在本地调整构图。

对话式修改采用“理解 → 必要时追问 → Prompt 预览 → 用户确认 → 生成”的流程。商品始终使用原始透明蒙版，本地只做等比例缩放；Agent 不会重绘或拉伸商品。参考图保存在本地 `agent_references` 目录，默认只提取色彩、明暗、饱和度和画面方向，并结合用户明确指定的参考元素生成新背景提示词。

第一次处理商品图会自动下载一次 `sam_b.pt`（约 358 MB）和 U2Net（约 176 MB）；以后直接使用本地模型。每张商品图的透明 PNG 蒙版会缓存到：

```text
E:\简历\movie_agent\impawards_posters\product_cutouts_sam
```

跨品类测试已覆盖手表、鞋履和手袋。上传透明背景 PNG 可以得到最干净的商品边缘；普通背景 JPG 也可自动处理。若商品与人物身体直接相连（例如脚上穿着鞋），当前版本会把相连主体一起保留，后续可增加“点击要保留的主体”或文本目标检测。

生成结果默认保存到：

```text
E:\简历\movie_agent\impawards_posters\generated_ads
```

当前任务队列与版本历史保存在运行进程内，适合本地和作品集测试。正式部署时应迁移到 PostgreSQL、Redis 队列和对象存储。

