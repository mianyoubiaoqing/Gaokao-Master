# Gaokao-Master

本项目以 Apache License 2.0 开源，详见 `LICENSE`。

Gaokao-Master 是一个本地 Python AI Agent 项目，目标是为高考备考构建可检索、可编辑、可生成个性化试卷的知识库与 Agent 系统。

当前已实现：

- Step 1: 项目结构与依赖清单
- Step 2: `KnowledgeBaseManager`，支持 `.md`、`.txt`、`.pdf`、`.docx` 资料解析、Markdown 化、逻辑切分和 ChromaDB 向量入库
- Step 3: Agent 工具层：`fuzzy_retrieve`、`workspace_editor`、`web_resource_scraper`
- Step 4: LangGraph 主 Agent 与 `Exam_Generator_Agent`
- Streamlit WebUI：资料导入、智能检索、个性化组卷、在线资源、Markdown 工作区
- OpenAI-compatible API：可接入 OpenAI 官方接口或任意兼容 OpenAI Chat Completions 协议的外部大模型

后续步骤将加入：

- 讲解生成、错因诊断和多轮学习规划

## 启动方式

推荐在 Windows 下使用项目自带脚本：

```powershell
.\install.bat
.\start_agent.bat
```

或使用 PowerShell：

```powershell
.\scripts\install.ps1
.\scripts\start_agent.ps1
```

启动成功后访问：

```text
http://127.0.0.1:8501
```

如果需要安装可选重依赖，例如 `sentence-transformers`、`pdfplumber`、完整 LangChain 组件：

```powershell
.\scripts\install.ps1 -Optional
```

安装脚本默认使用清华 PyPI 镜像，适合国内网络。如果你要切换回官方源：

```powershell
.\scripts\install.ps1 -IndexUrl https://pypi.org/simple
```

说明：核心依赖已尽量瘦身，避免默认安装时下载 PyTorch 等较大的包。默认向量化使用 `local-hash`，完全离线、无需下载模型文件。你也可以在 WebUI 左侧把 `Embedding 模型` 改成 `chromadb-default` 或 sentence-transformers 模型名，但这些模式可能触发额外模型下载。

## 接入外部大模型

方式一：在 WebUI 左侧“外部大模型”区域启用 OpenAI 兼容 API，并填写：

- `Base URL`：例如 `https://api.openai.com/v1`
- `模型名`：例如 `gpt-4o-mini`
- `API Key`：你的服务商密钥

方式二：复制 `.env.example` 为 `.env`，填写：

```bash
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
OPENAI_TIMEOUT=60
```

启用后，`Exam_Generator_Agent` 会优先调用外部模型优化试卷结构和个性化说明；未配置或调用失败时，会自动回退到本地规则生成。

## 在线资源搜索

“在线资源”页支持三种方式：

- 自动搜索：依次尝试 Tavily、DuckDuckGo/Bing、普通 HTML 搜索兜底。
- 手动链接：每行粘贴一个 PDF/DOCX/DOC 链接，最稳定。
- 本地上传：如果当前网络无法访问搜索引擎，在“资料导入”页上传本地文件。

如果你有 Tavily API Key，推荐直接在 WebUI 左侧“在线搜索”区域填写。

也可以在 `.env` 中配置作为兜底：

```bash
TAVILY_API_KEY=tvly-your-key
```

配置后在线搜索会优先使用 Tavily，通常比网页搜索更稳定。

原则：后续所有外部服务配置都会优先提供 WebUI 输入项，`.env` 只作为批量部署或默认值兜底。

## 配置持久化

WebUI 左侧的配置会自动保存到本地：

```text
.gaokao_master/settings.json
```

包括知识库路径、Embedding 模型、OpenAI-compatible API 配置、Tavily API Key 等。重启项目后会自动恢复。该文件已加入 `.gitignore`，不会被提交到仓库。

## PDF 转 Markdown 说明

PDF 转换优先使用 PyMuPDF 提取可复制文本。如果 PDF 是扫描图片版，系统会检测到文本不足。

你可以在 WebUI 左侧开启：

```text
OCR 多模态模型 -> 自动 OCR 扫描版 PDF
```

填写支持图片输入的 OpenAI-compatible 多模态模型后，系统会自动把 PDF 页面渲染成图片并调用该模型 OCR。OCR 成功后会写入 Markdown 和向量库；OCR 未配置或失败时，会生成一份“需 OCR”的诊断 Markdown，但不会把空内容写入向量库。

遇到 `needs_ocr_or_empty_pdf` 时，请优先使用：

- 可复制文字版 PDF
- DOCX 试卷
- OCR 后的 PDF/TXT/Markdown
- WebUI “资料导入”页手动上传 OCR 后文件

OCR 配置同样会自动保存到 `.gaokao_master/settings.json`。

## 工作区预览与删除

WebUI 的“工作区”包含四个面板：

- `Markdown 预览`：将 Markdown 渲染成接近 Obsidian 的阅读效果。
- `源码编辑`：保留原始 Markdown 编辑能力。
- `媒体库`：上传、预览、删除试卷题图等多媒体资源。
- `下载区`：删除 `_raw` 下载/上传的原始文件，可选择同步删除对应 RAG 向量。
- `RAG 区`：按 Markdown 删除向量记录，或清空当前 Chroma collection。

删除操作会限制在知识库目录内，并要求确认勾选或输入 collection 名。

预览支持常用 Obsidian 效果：

- `$...$`、`$$...$$`、`\(...\)`、`\[...\]` 数学公式
- 表格、代码块、引用块
- `[[双链]]`、`[[目标|别名]]`
- `==高亮==`

公式会优先使用 `latex2mathml` 渲染成 MathML；如果该依赖未安装或遇到无法转换的公式，WebUI 会使用内置的本地简易公式渲染器处理高考试卷常见语法，例如 `\frac`、`\sqrt`、上下标、向量、角度、集合符号和希腊字母，不再把 `$...$` 源码直接显示出来。

媒体库文件存放在：

```text
Gaokao_KB/assets/
```

在 Markdown 中可这样引用：

```markdown
![[figure.png]]
![函数图像](assets/figure.png)
```

WebUI 预览会自动把本地图片内嵌显示，因此题目附带图片可以直接在试卷预览里查看。
### OCR 题图提取

开启 OCR 后，WebUI 侧边栏可以勾选 `OCR 时自动提取题图到媒体库`。系统会在 OCR PDF 时尝试从页面里裁剪图片块和矢量图块，保存到：

```text
Gaokao_KB/assets/{source_file_name}/
```

OCR 模型会被提示在图形原位置输出 `[IMAGE_HERE]`，导入流程会把这些占位符替换成真实 Markdown 图片链接，例如：

```markdown
![page 1 figure 1](assets/example/page_001_figure_01.png)
```

这样立体几何、函数图像、统计图等题目图片会随 Markdown 一起在 WebUI 预览中显示。
