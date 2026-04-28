# Gaokao-Master

Gaokao-Master 是一个面向高考复习的本地知识库与学习工作台。它把试卷、笔记、解析、错题记录整理成 Obsidian 风格的 Markdown 工作区，并提供混合检索、试卷分析、错题本、OCR 入库和媒体库能力。

本项目以 Apache License 2.0 开源，详见 [LICENSE](LICENSE)。

## 当前定位

这个项目不再主打自动组卷或线上答题，而是服务于更稳定的复习闭环：

- 导入资料：把 PDF、DOCX、TXT、Markdown 转成可长期维护的 Markdown。
- 检索资料：用语义检索和关键词检索定位知识点、题目、解析。
- 分析试卷：统计题型结构、答案解析完整度、公式和图片情况，并可调用外部大模型生成复习建议。
- 记录错题：把做错的题按科目、专题、状态沉淀到本地错题本。
- 管理素材：保存 OCR 提取的题图、几何图、函数图像等媒体资源。
- 本地优先：配置、知识库、错题和分析报告默认保存在本机。

已删除的旧功能：

- 个性化组卷
- 在线做题
- 答题卡生成

## 项目结构

```text
.
├── agent_start.py
├── install.bat
├── start_agent.bat
├── requirements.txt
├── requirements-optional.txt
├── scripts/
│   ├── install.ps1
│   └── start_agent.ps1
└── src/
    └── gaokao_master/
        ├── config.py
        ├── kb/
        │   └── manager.py
        ├── llm/
        │   └── openai_compatible.py
        ├── tools/
        │   └── core.py
        └── web/
            └── app.py
```

运行后默认会创建：

```text
Gaokao_KB/
├── _raw/                 # 原始上传/下载文件
├── .vector_store/        # ChromaDB 向量库
├── assets/               # 媒体库，题图和 OCR 裁剪图片
├── 试卷分析/             # 保存的分析报告
└── {科目}/
    ├── {专题}/           # 导入后的资料 Markdown
    └── 错题本/
        └── {专题}/       # 错题 Markdown
```

## 安装与启动

Windows 推荐使用项目脚本：

```powershell
.\install.bat
.\start_agent.bat
```

或使用 PowerShell：

```powershell
.\scripts\install.ps1
.\scripts\start_agent.ps1
```

启动后访问：

```text
http://127.0.0.1:8501
```

如需安装可选重依赖，例如 `sentence-transformers` 或完整 LangChain 组件：

```powershell
.\scripts\install.ps1 -Optional
```

安装脚本默认使用清华 PyPI 镜像。切换官方源：

```powershell
.\scripts\install.ps1 -IndexUrl https://pypi.org/simple
```

## WebUI 功能

### 资料导入

支持上传：

- `.md`
- `.txt`
- `.pdf`
- `.docx`

PDF 和 DOCX 会被转换成 Markdown，再写入本地 ChromaDB 向量库。上传的原始文件保存在：

```text
Gaokao_KB/_raw/uploads/{科目}/{专题}/
```

### 智能检索

检索同时使用：

- ChromaDB 语义检索
- BM25 关键词检索

适合查知识点、找题、定位解析、回看错题。

### 错题本

错题本支持：

- 新增错题
- 记录我的错误过程
- 保存正确答案/标准解析
- 标注错因和订正要点
- 设置状态：`待复盘`、`已订正`、`已掌握`
- 按科目、状态、关键词筛选
- 可选调用 OpenAI 兼容大模型生成错因诊断

错题保存位置：

```text
Gaokao_KB/{科目}/错题本/{专题}/
```

错题会同时写入向量库，之后可以通过智能检索找回。

### 试卷分析

试卷分析会先生成本地结构统计：

- 题目数量
- 选择题特征数量
- 图片引用数量
- 公式/LaTeX 特征数量
- 答案/解析标记数量
- 栏目结构
- 高频知识点线索

如果配置了 OpenAI 兼容 API，还可以生成更完整的 Markdown 报告：

- 知识点分布
- 难度梯度
- 易错点
- 复习优先级
- 一周复盘计划

分析报告保存位置：

```text
Gaokao_KB/试卷分析/
```

### 在线资源

在线资源页支持：

- Tavily 搜索
- DuckDuckGo/Bing/Sogou HTML 兜底搜索
- 手动粘贴 PDF/DOCX/DOC 链接
- 下载后自动进入知识库管线

搜索会偏向试卷、试题、真题、模拟、答案、解析等资源，并过滤志愿、分数线、录取、招生、大学排名等高噪声内容。

下载文件保存位置：

```text
Gaokao_KB/_raw/web/{科目}/{专题}/
```

下载文件使用内容哈希命名，避免重复保存。同一内容的重复文件也可以在工作区的下载区中整理删除。

### 工作区

工作区包含：

- Markdown 预览
- 源码编辑
- 媒体库
- 下载区
- RAG 区

Markdown 预览接近 Obsidian 阅读效果，支持：

- 表格
- 代码块
- 引用块
- `[[双链]]`
- `==高亮==`
- 本地图片内嵌显示
- 常见 LaTeX 数学公式预处理和渲染

预览区内有打印按钮，可直接打印当前 Markdown 资料。

## OpenAI 兼容 API

WebUI 左侧可配置外部大模型：

- `Base URL`
- `模型名`
- `API Key`
- `Temperature`

兼容 OpenAI Chat Completions 协议的服务均可接入。也可以使用 `.env` 作为兜底配置：

```bash
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
OPENAI_TIMEOUT=60
```

当前会用到大模型的地方：

- 错题本：生成错因诊断与复习建议
- 试卷分析：生成完整分析报告

## OCR 与题图提取

当 PDF 是扫描版或文本过少时，可以在 WebUI 左侧启用 OCR 多模态模型。

需要填写：

- OCR Base URL
- OCR 模型名
- OCR API Key
- OCR DPI
- OCR 最大页数

启用 `OCR 时自动提取题图到媒体库` 后，系统会尝试从 PDF 页面中裁剪图片块和矢量图块，保存到：

```text
Gaokao_KB/assets/{source_file_name}/
```

Markdown 中会插入类似链接：

```markdown
![page 1 figure 1](assets/example/page_001_figure_01.png)
```

这样立体几何、函数图像、统计图等题图可以直接在预览里显示。

## 配置持久化

WebUI 侧边栏配置会自动保存到：

```text
.gaokao_master/settings.json
```

包括：

- 知识库路径
- Embedding 模型
- OpenAI 兼容 API 配置
- Tavily API Key
- OCR 配置

`.gaokao_master/` 已加入 `.gitignore`，不会提交到仓库。

## 向量化说明

默认 Embedding 模型为：

```text
local-hash
```

它是一个完全离线的哈希向量函数，不会下载模型文件。优点是稳定、轻量、适合零配置启动；缺点是语义能力弱于真正的 embedding 模型。

如果希望更强语义检索，可以在 WebUI 左侧把 `Embedding 模型` 改为：

```text
chromadb-default
```

或填写 sentence-transformers 模型名。注意这些模式可能触发额外模型下载。

## CLI

除了 WebUI，也可以运行简单检索 CLI：

```powershell
python agent_start.py cli
```

CLI 会读取本地知识库并返回检索命中。

## 安全提示

- 不要把真实 API Key 写入 README 或提交到 Git。
- `.env`、`.gaokao_master/`、`Gaokao_KB/`、`.venv/` 已被忽略。
- 公开仓库中只应保存代码、示例配置和文档，不应保存学生隐私数据或真实试卷版权材料。
