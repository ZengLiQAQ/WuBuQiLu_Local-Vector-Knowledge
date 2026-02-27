[English](README_EN.md) | 中文

# 本地向量知识库

[![macOS](https://img.shields.io/badge/platform-macOS-orange)](https://www.apple.com/macos)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

基于 macOS 的本地向量知识库管理工具，支持多种文件格式的导入、向量化存储和语义检索。

[快速开始](#快速开始) • [功能特性](#功能特性) • [API 文档](#api-文档) • [配置说明](#配置说明)

---

## 快速开始

**第一步：安装依赖**

```bash
# 克隆项目
git clone https://github.com/your-repo/WuBuQiLu_Local-Vector-Knowledge.git
cd WuBuQiLu_Local-Vector-Knowledge

# 安装 Python 依赖
pip3 install -r requirements.txt
```

**第二步：启动服务**

```bash
python3 web_ui.py
```

**第三步：访问**

打开浏览器访问 http://localhost:8000

---

## 功能特性

### 多格式支持

支持导入多种文件格式，自动进行文本提取和向量化处理：

| 格式 | 支持 | 说明 |
|------|------|------|
| 📄 PDF | ✅ | 流式解析，减少内存占用 |
| 📝 Markdown | ✅ | 转为纯文本处理 |
| 📊 Excel | ✅ | 按工作表合并文本 |
| 🖼️ 图片 | ✅ | OCR 文字识别 |
| 📃 TXT | ✅ | 按行拆分 |
| 📄 Word | ✅ | 段落+表格解析 |
| 📽️ PPT | ✅ | 按幻灯片拆分 |

### 智能检索

- **混合检索**：向量相似度 + 关键词匹配，综合得分排序
- **关键词高亮**：检索结果中高亮匹配关键词
- **类型过滤**：支持按文件类型筛选结果

### MacOS 优化

- **MPS 加速**：Apple Silicon 设备自动使用 GPU 加速
- **量化模型**：8bit 量化减少 50% 内存占用
- **增量更新**：自动检测文件变更，只更新修改过的文件

### Web 管理界面

- 📤 文件上传入库
- 🔍 语义检索
- 📁 文件管理
- 📊 统计图表
- 🌙 暗黑模式

---

## 项目结构

```
local_vector_kb/
├── setup_macos.sh          # macOS 一键安装脚本
├── config.yaml             # 配置文件
├── LocalVectorKB.py        # 核心知识库类
├── web_ui.py               # Web 服务入口
├── templates/
│   └── index.html          # Web 前端页面
├── static/                 # 静态文件目录
├── temp_uploads/          # 临时文件目录
├── requirements.txt       # Python 依赖
├── README.md              # 中文文档
└── README_EN.md           # English Docs
```

---

## 配置说明

### 配置文件 (config.yaml)

```yaml
# 知识库配置
kb:
  path: "./local_vector_db"        # 向量库存储路径
  collection_name: "multi_file_kb" # 集合名称
  chunk_size: 500                  # 分块大小
  incremental_update: true          # 开启增量更新

# 模型配置
model:
  embed_model: "BAAI/bge-m3"       # 嵌入模型
  device: "auto"                   # auto=自动检测 MPS/CPU
  quantization: true               # 开启量化

# ChromaDB 配置
chroma:
  database_impl: "leveldb"         # LevelDB 后端

# Web 服务配置
web:
  host: "0.0.0.0"
  port: 8000
  api_key: "your-secret-key"       # API 鉴权密钥
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `KB_API_KEY` | API 鉴权密钥 | `local_kb_2026_secure_key` |

---

## API 文档

所有 API 需要在请求头中携带 `X-API-Key`。

### 接口列表

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

### 使用示例

```bash
# 搜索
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: local_kb_2026_secure_key" \
  -F "query=你的查询" \
  -F "top_k=5"

# 获取统计
curl http://localhost:8000/api/v1/stats \
  -H "X-API-Key: local_kb_2026_secure_key"

# 上传文件
curl -X POST http://localhost:8000/api/v1/upload_async \
  -H "X-API-Key: local_kb_2026_secure_key" \
  -F "file=@/path/to/file.pdf"
```

---

## 技术栈

- **向量数据库**: [ChromaDB](https://www.trychroma.com/) (LevelDB 后端)
- **嵌入模型**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Web 框架**: [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- **前端**: HTML5 + ECharts
- **OCR**: Tesseract

---

## 性能优化

| 优化项 | 说明 |
|--------|------|
| MPS 加速 | Apple Silicon 设备自动使用 GPU |
| 量化模型 | 8bit 量化降低内存占用 |
| 批量处理 | 支持批量上传和检索 |
| 增量更新 | 只处理修改过的文件 |
| LevelDB | 比 SQLite 快 10 倍 |

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
