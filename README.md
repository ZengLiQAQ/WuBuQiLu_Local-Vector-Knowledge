[English](README_EN.md) | 中文

# 本地向量知识库

[![macOS](https://img.shields.io/badge/platform-macOS-orange)](https://www.apple.com/macos)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

基于 macOS 的本地向量知识库管理工具，支持多种文件格式导入、向量化存储、语义检索和自定义模型训练。

[快速开始](#快速开始) • [功能特性](#功能特性) • [项目结构](#项目结构) • [API 文档](#api-文档) • [配置说明](#配置说明)

---

## 快速开始

**第一步：克隆项目**

```bash
git clone https://github.com/your-repo/WuBuQiLu_Local-Vector-Knowledge.git
cd WuBuQiLu_Local-Vector-Knowledge
```

**第二步：安装依赖**

```bash
# Python 依赖
pip3 install -r requirements.txt

# 前端依赖
cd web && npm install && cd ..
```

**第三步：启动服务**

```bash
# 启动后端 (默认端口 18080)
python3 web_ui.py

# 启动前端 (开发模式，默认端口 3000)
cd web && npm run dev
```

**第四步：访问**

- 后端管理界面：http://localhost:18080
- 前端界面：http://localhost:3000

---

## 功能特性

### 📁 多格式文件支持

| 格式 | 状态 | 说明 |
|------|------|------|
| PDF | ✅ | 流式解析，减少内存占用 |
| Markdown | ✅ | 转为纯文本处理 |
| Excel | ✅ | 按工作表合并文本 |
| Word | ✅ | 段落+表格解析 |
| PPT | ✅ | 按幻灯片拆分 |
| 图片 | ✅ | OCR 文字识别 (Tesseract) |
| TXT | ✅ | 按行拆分 |

### 🔍 智能检索

- **语义检索**：基于向量相似度的语义匹配
- **混合检索**：向量 + BM25 关键词组合检索
- **重排序**：BGE-Reranker 提升结果相关性
- **关键词高亮**：检索结果中高亮匹配关键词

### 🤖 模型训练

支持自定义训练嵌入模型，适配特定领域的语义检索：

- **数据制备**：CSV/TXT 格式、钉钉文档导入
- **数据清洗**：文本清洗、长度过滤、样本平衡
- **数据增强**：噪声样本生成、同义词替换
- **模型选择**：BGE-M3、BERT、RoBERTa 等
- **训练配置**：Epochs、Batch Size、Learning Rate 可调
- **模型管理**：版本列表、部署、回滚、验证
- **Mac 加速**：Apple Silicon MPS 硬件加速

### 🏷️ 标签管理

- 文档标签添加/移除
- 标签筛选和搜索
- 批量标签操作

### 📊 数据导出

- 导出训练数据
- 知识库备份

### 🌙 界面特性

- 响应式设计
- 暗黑模式支持
- 实时任务状态
- 统计图表展示

---

## 项目结构

```
WuBuQiLu_Local-Vector-Knowledge/
├── web_ui.py                 # FastAPI 后端服务入口
├── LocalVectorKB.py          # 核心向量知识库类
├── tag_db.py                 # 标签数据库
├── bm25_search.py            # BM25 关键词检索
├── config.yaml               # 配置文件
├── requirements.txt          # Python 依赖
│
├── web/                      # Next.js 前端
│   ├── src/
│   │   ├── app/             # 页面组件
│   │   │   ├── page.tsx             # 首页/仪表盘
│   │   │   ├── search/              # 检索页面
│   │   │   ├── documents/           # 文档管理
│   │   │   ├── tags/                # 标签管理
│   │   │   ├── export/              # 数据导出
│   │   │   ├── settings/            # 设置页面
│   │   │   └── train/               # 模型训练
│   │   │       ├── data/            # 训练数据
│   │   │       ├── config/          # 训练配置
│   │   │       ├── models/          # 模型管理
│   │   │       └── run/             # 训练执行
│   │   ├── components/      # UI 组件
│   │   ├── lib/              # 工具函数
│   │   └── types/            # TypeScript 类型
│   ├── package.json
│   └── tailwind.config.ts
│
├── model_train/              # 模型训练模块
│   ├── app.py               # 训练服务
│   ├── config.py            # 训练配置
│   ├── data/                # 训练数据
│   ├── models/              # 训练模型
│   └── logs/                # 训练日志
│
├── local_vector_db/          # 向量数据库存储
├── temp_uploads/             # 临时上传文件
├── templates/                # Jinja2 模板 (旧版 Web UI)
├── docs/                    # 文档
└── README.md                # 中文文档
```

---

## 配置说明

### 配置文件 (config.yaml)

```yaml
# 知识库配置
kb:
  path: "./local_vector_db"          # 向量库存储路径
  collection_name: "multi_file_kb"   # 集合名称
  chunk_size: 500                    # 分块大小
  chunk_overlap: 50                  # 分块重叠长度
  incremental_update: true           # 开启增量更新

# 模型配置 (MacOS 优化)
model:
  embed_model: "BAAI/bge-m3"         # 嵌入模型
  device: "auto"                     # auto=自动检测 MPS/CPU
  normalize_embeddings: true         # 归一化提升检索精度
  quantization: true                 # 开启 8bit 量化
  batch_size: 64                     # 批量编码大小

# ChromaDB 配置
chroma:
  hnsw_space: "cosine"               # 余弦相似度
  hnsw_ef_construction: 200         # 构建精度
  hnsw_M: 16                        # 邻居数
  hnsw_ef: 100                      # 检索精度
  database_impl: "leveldb"          # LevelDB 后端

# Web 服务配置
web:
  host: "0.0.0.0"
  port: 18080
  api_key: "local_kb_2026_secure_key"

# 日志配置
log:
  level: "INFO"
  file: "kb_logs.log"
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `KB_API_KEY` | API 鉴权密钥 | `local_kb_2026_secure_key` |

---

## API 文档

所有 API 需要在请求头中携带 `X-API-Key`。

### 知识库 API

| 方法 | 路径 | 描述 | 限流 |
|------|------|------|------|
| GET | / | 管理界面 | - |
| POST | /api/v1/upload_async | 异步上传文件 | 60/min |
| GET | /api/v1/task/{task_id} | 查询任务状态 | 120/min |
| POST | /api/v1/search | 语义检索 | 60/min |
| POST | /api/v1/delete | 删除文件 | 30/min |
| POST | /api/v1/batch_delete | 批量删除 | 10/min |
| POST | /api/v1/clear | 清空知识库 | 5/min |
| GET | /api/v1/stats | 获取统计信息 | 30/min |
| POST | /api/v1/batch_upload | 批量上传 | 10/min |
| POST | /api/v1/batch_search | 批量检索 | 30/min |

### 标签 API

| 方法 | 路径 | 描述 | 限流 |
|------|------|------|------|
| GET | /api/v1/tags | 获取标签列表 | 30/min |
| POST | /api/v1/tags | 创建标签 | 30/min |
| PUT | /api/v1/tags/{tag_id} | 更新标签 | 30/min |
| DELETE | /api/v1/tags/{tag_id} | 删除标签 | 30/min |
| POST | /api/v1/documents/{doc_id}/tags | 添加标签 | 30/min |
| DELETE | /api/v1/documents/{doc_id}/tags/{tag_id} | 移除标签 | 30/min |

### 训练 API

| 方法 | 路径 | 描述 | 限流 |
|------|------|------|------|
| POST | /api/v1/train/upload | 上传训练数据 | 30/min |
| GET | /api/v1/train/data | 获取训练数据列表 | 30/min |
| POST | /api/v1/train/clean | 清洗训练数据 | 30/min |
| GET | /api/v1/train/models | 获取可用基础模型 | 30/min |
| GET | /api/v1/train/config | 获取默认训练配置 | 30/min |
| POST | /api/v1/train/start | 启动训练 | 10/min |
| GET | /api/v1/train/status | 获取训练状态 | 60/min |
| POST | /api/v1/train/stop | 停止训练 | 10/min |
| GET | /api/v1/train/models/list | 获取已训练模型列表 | 30/min |
| POST | /api/v1/train/deploy | 部署模型 | 10/min |
| POST | /api/v1/train/validate | 验证模型 | 30/min |

### 使用示例

```bash
# 搜索
curl -X POST http://localhost:18080/api/v1/search \
  -H "X-API-Key: local_kb_2026_secure_key" \
  -F "query=你的查询" \
  -F "top_k=5"

# 获取统计
curl http://localhost:18080/api/v1/stats \
  -H "X-API-Key: local_kb_2026_secure_key"

# 上传文件
curl -X POST http://localhost:18080/api/v1/upload_async \
  -H "X-API-Key: local_kb_2026_secure_key" \
  -F "file=@/path/to/file.pdf"
```

---

## 技术栈

### 后端

- **向量数据库**: [ChromaDB](https://www.trychroma.com/) (LevelDB 后端)
- **嵌入模型**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Web 框架**: [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- **限流**: SlowAPI
- **OCR**: Tesseract

### 前端

- **框架**: [Next.js](https://nextjs.org) 14
- **UI 库**: React 18 + TypeScript
- **样式**: [Tailwind CSS](https://tailwindcss.com) + Radix UI
- **图表**: ECharts

---

## 性能优化

| 优化项 | 说明 |
|--------|------|
| MPS 加速 | Apple Silicon 设备自动使用 GPU |
| 量化模型 | 8bit 量化降低内存占用 |
| 批量处理 | 支持批量上传和检索 |
| 增量更新 | 只处理修改过的文件 |
| LevelDB | ChromaDB 使用 LevelDB 后端 |

---

## 注意事项

1. 首次启动会下载嵌入模型，需要一定时间
2. 大文件入库时间较长，建议使用异步上传
3. 请妥善保管 API Key，避免泄露
4. 清空知识库操作不可恢复，请谨慎使用

---

## 常见问题

**Q: 启动报错 "No module named 'xxx'"？**
> A: 运行 `pip3 install -r requirements.txt` 安装所有依赖

**Q: 如何修改 API 密钥？**
> A: 编辑 `config.yaml` 中的 `web.api_key` 项

**Q: 支持 Windows/Linux 吗？**
> A: 当前版本针对 macOS 优化，其他系统需调整 Tesseract 路径

**Q: 如何查看日志？**
> A: 日志文件默认保存在 `kb_logs.log`

---

## 开源协议

MIT License

---

<div align="center">

**如果你喜欢这个项目，欢迎 ⭐ Star**

</div>
