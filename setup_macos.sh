#!/bin/bash
set -e

# 1. 检查并安装Homebrew
if ! command -v brew &> /dev/null; then
    echo "🔧 正在安装Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # 临时添加brew到PATH
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# 2. 安装系统级依赖
echo "🔧 正在安装Tesseract-OCR及中文语言包..."
brew install tesseract tesseract-lang libomp

# 3. 升级pip并安装Python依赖
echo "🔧 正在安装Python依赖..."
pip3 install --upgrade pip
pip3 install \
    chromadb[leveldb] sentence-transformers python-dotenv faiss-cpu \
    pypdf python-docx openpyxl python-multipart markdown pillow pytesseract \
    python-pptx fastapi uvicorn jinja2 python-multipart aiofiles \
    torch transformers optimum auto-gptq \
    slowapi httpx python-multipart pyyaml loguru beautifulsoup4

# 4. 验证关键依赖
echo "✅ 验证依赖安装..."
if python3 -c "import chromadb, sentence_transformers, docx, pptx, fastapi, torch, yaml, bs4" &> /dev/null; then
    echo "🎉 所有依赖安装成功！"
    echo "📌 Tesseract路径：$(which tesseract)"
    echo "📌 PyTorch MPS支持：$(python3 -c "import torch; print('可用' if torch.backends.mps.is_available() else '不可用')")"
else
    echo "❌ 部分依赖安装失败，请检查错误信息"
    exit 1
fi

# 5. 创建必要目录
echo "🔧 创建项目目录..."
mkdir -p templates static temp_uploads

echo "✅ 环境配置完成！"
echo "📚 下一步："
echo "  1. 将index.html放入templates目录"
echo "  2. 运行：python3 web_ui.py"
echo "  3. 访问：http://localhost:8000"
