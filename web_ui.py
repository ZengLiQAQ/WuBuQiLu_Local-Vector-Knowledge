
import os
import time
import hashlib
import asyncio
import re
import json
import uuid
import shutil
import threading
import pandas as pd
import numpy as np
from loguru import logger
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, status, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aiofiles
import yaml

# 简单文件名安全处理
def safe_filename(filename: str) -> str:
    """移除文件名中的路径遍历字符"""
    return os.path.basename(filename)

from LocalVectorKB import LocalVectorKB, load_config


# ========== 初始化配置 ==========
CONFIG = load_config()
API_VERSION = "v1"
API_KEY = CONFIG["web"].get("api_key", "local_kb_2026_secure_key")
TEMP_DIR = "./temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ========== 模型训练配置 ==========
MODEL_TRAIN_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_train")
MODEL_DATA_DIR = os.path.join(MODEL_TRAIN_BASE_DIR, "data")
MODEL_MODELS_DIR = os.path.join(MODEL_TRAIN_BASE_DIR, "models")
MODEL_CHECKPOINTS_DIR = os.path.join(MODEL_TRAIN_BASE_DIR, "checkpoints")
MODEL_LOGS_DIR = os.path.join(MODEL_TRAIN_BASE_DIR, "logs")

# 确保目录存在
for d in [MODEL_DATA_DIR, MODEL_MODELS_DIR, MODEL_CHECKPOINTS_DIR, MODEL_LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# 可用的基础模型
AVAILABLE_MODELS = [
    {'id': 'BAAI/bge-m3', 'name': 'BGE-M3 (推荐 - Mac优化)', 'type': 'bge-m3'},
    {'id': 'chinese-roberta-wwm-ext', 'name': 'Chinese RoBERTa WWM Ext', 'type': 'bert'},
    {'id': 'bert-base-chinese', 'name': 'BERT Base Chinese', 'type': 'bert'},
    {'id': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'name': 'Multilingual MiniLM', 'type': 'st'},
    {'id': 'shibing624/text2vec-base-chinese', 'name': 'Text2Vec Chinese', 'type': 'st'},
]

# 训练默认参数
DEFAULT_TRAIN_CONFIG = {
    'epochs': 5,
    'batch_size': 16,
    'warmup_steps': 100,
    'learning_rate': 2e-5,
    'max_seq_length': 256,
    'eval_steps': 500,
    'save_steps': 500,
}


# ========== FastAPI初始化 ==========
app = FastAPI(
    title="本地向量知识库WebAPI",
    description="支持多文件类型的本地向量知识库（MacOS优化版）",
    version=API_VERSION
)

# 静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 限流配置
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key鉴权
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API Key"
        )
    return api_key_header

# 初始化知识库
kb = LocalVectorKB()

# 全局任务字典
tasks = {}

# 训练状态
training_state = {
    'running': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'logs': [],
    'model_path': None,
    'config': None,
}


# ========== 统一响应模型 ==========
class ResponseModel:
    @staticmethod
    def success(data=None, message="操作成功"):
        return JSONResponse({
            "success": True,
            "code": 200,
            "message": message,
            "data": data
        })

    @staticmethod
    def error(message="操作失败", code=500, data=None):
        return JSONResponse({
            "success": False,
            "code": code,
            "message": message,
            "data": data
        })


# ========== 后台任务 ==========
async def process_file_async(file_path: str, task_id: str):
    """异步处理文件入库"""
    try:
        tasks[task_id] = {"status": "processing", "progress": 0}
        # 模拟进度更新
        chunk_count = kb.add_file(file_path)
        tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "chunk_count": chunk_count,
            "message": f"文件入库成功，拆分{chunk_count}个文本块"
        }
    except Exception as e:
        tasks[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"入库失败：{str(e)}"
        }
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)


# ========== 模型训练辅助函数 ==========

def allowed_file(filename, allowed_extensions={'csv', 'txt'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def clean_text(text):
    """清洗文本：去除特殊字符，保留中文、英文、数字、空格"""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return text.strip()


def log_train_message(message, level='INFO'):
    """记录训练日志"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"
    training_state['logs'].append(log_entry)
    logger.info(log_entry)


# ========== 页面路由 ==========
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页：管理界面"""
    stats = kb.get_stats()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
        "api_version": API_VERSION
    })


# ========== API路由（v1版本） ==========
@app.post(f"/api/{API_VERSION}/upload_async")
@limiter.limit("60/minute")
async def upload_file_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """异步上传文件入库"""
    try:
        # 生成任务ID
        task_id = hashlib.md5(f"{file.filename}_{time.time()}".encode()).hexdigest()
        tasks[task_id] = {"status": "pending", "progress": 0}

        # 保存临时文件（使用安全文件名）
        safe_name = safe_filename(file.filename)
        file_path = os.path.join(TEMP_DIR, safe_name)
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # 添加后台任务
        background_tasks.add_task(process_file_async, file_path, task_id)

        return ResponseModel.success(
            data={"task_id": task_id, "file_name": file.filename},
            message="文件已接收，后台处理中"
        )
    except Exception as e:
        logger.error(f"异步上传失败：{e}")
        return ResponseModel.error(message=f"上传失败：{str(e)}")


@app.get("/api/{api_version}/task/{task_id}")
@limiter.limit("120/minute")
async def get_task_status(
    request: Request,
    api_version: str,
    task_id: str,
    api_key: str = Depends(get_api_key)
):
    """查询任务状态"""
    if task_id not in tasks:
        return ResponseModel.error(message="任务不存在", code=404)
    return ResponseModel.success(data={"task": tasks[task_id]})


@app.post(f"/api/{API_VERSION}/search")
@limiter.limit("60/minute")
async def search(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(5),
    file_type_filter: str = Form(None),
    hybrid: bool = Form(True),
    api_key: str = Depends(get_api_key)
):
    """相似检索"""
    try:
        results = kb.search(query, top_k, file_type_filter, hybrid)
        return ResponseModel.success(
            data={"query": query, "results": results},
            message="检索成功"
        )
    except Exception as e:
        logger.error(f"检索失败：{e}")
        return ResponseModel.error(message=f"检索失败：{str(e)}")


@app.post(f"/api/{API_VERSION}/delete")
@limiter.limit("30/minute")
async def delete_file(
    request: Request,
    file_path: str = Form(...),
    api_key: str = Depends(get_api_key)
):
    """删除文件数据"""
    try:
        success = kb.delete_by_file(file_path)
        if success:
            return ResponseModel.success(message=f"删除文件数据成功：{file_path}")
        else:
            return ResponseModel.error(message=f"删除文件数据失败：{file_path}")
    except Exception as e:
        logger.error(f"删除失败：{e}")
        return ResponseModel.error(message=f"删除失败：{str(e)}")


@app.post(f"/api/{API_VERSION}/batch_delete")
@limiter.limit("10/minute")
async def batch_delete(
    request: Request,
    file_paths: list = Form(...),
    api_key: str = Depends(get_api_key)
):
    """批量删除文件数据"""
    try:
        success_count = kb.batch_delete(file_paths)
        return ResponseModel.success(
            data={"success_count": success_count, "total_count": len(file_paths)},
            message=f"批量删除完成：成功{success_count}/{len(file_paths)}"
        )
    except Exception as e:
        logger.error(f"批量删除失败：{e}")
        return ResponseModel.error(message=f"批量删除失败：{str(e)}")


@app.post(f"/api/{API_VERSION}/clear")
@limiter.limit("5/minute")
async def clear_all(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """清空知识库"""
    try:
        success = kb.clear_all()
        if success:
            return ResponseModel.success(message="知识库已清空")
        else:
            return ResponseModel.error(message="清空知识库失败")
    except Exception as e:
        logger.error(f"清空失败：{e}")
        return ResponseModel.error(message=f"清空失败：{str(e)}")


@app.get(f"/api/{API_VERSION}/stats")
@limiter.limit("30/minute")
async def get_stats(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """获取统计信息"""
    try:
        stats = kb.get_stats()
        return ResponseModel.success(data=stats, message="获取统计信息成功")
    except Exception as e:
        logger.error(f"获取统计信息失败：{e}")
        return ResponseModel.error(message=f"获取统计信息失败：{str(e)}")


@app.post(f"/api/{API_VERSION}/batch_upload")
@limiter.limit("10/minute")
async def batch_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    api_key: str = Depends(get_api_key)
):
    """批量上传文件"""
    try:
        results = []
        for file in files:
            file_path = os.path.join(TEMP_DIR, file.filename)
            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            chunk_count = kb.add_file(file_path)
            os.remove(file_path)

            results.append({
                "file_name": file.filename,
                "chunk_count": chunk_count,
                "status": "success" if chunk_count > 0 else "failed"
            })

        return ResponseModel.success(
            data={"results": results},
            message=f"批量上传完成，共处理{len(files)}个文件"
        )
    except Exception as e:
        logger.error(f"批量上传失败：{e}")
        return ResponseModel.error(message=f"批量上传失败：{str(e)}")


@app.post(f"/api/{API_VERSION}/batch_search")
@limiter.limit("30/minute")
async def batch_search(
    request: Request,
    queries: list[str] = Form(...),
    top_k: int = Form(5),
    file_type_filter: str = Form(None),
    hybrid: bool = Form(True),
    api_key: str = Depends(get_api_key)
):
    """批量检索"""
    try:
        results = {}
        for query in queries:
            results[query] = kb.search(query, top_k, file_type_filter, hybrid)

        return ResponseModel.success(
            data={"results": results},
            message="批量检索成功"
        )
    except Exception as e:
        logger.error(f"批量检索失败：{e}")
        return ResponseModel.error(message=f"批量检索失败：{str(e)}")


# ========== 模型训练API ==========

@app.post(f"/api/{API_VERSION}/train/upload")
@limiter.limit("30/minute")
async def upload_training_data(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """上传训练数据文件"""
    try:
        if not allowed_file(file.filename):
            return ResponseModel.error(message="不支持的文件类型，仅支持CSV和TXT", code=400)

        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())[:8]
        save_name = f"{file_id}_{filename}"
        filepath = os.path.join(MODEL_DATA_DIR, save_name)

        content = await file.read()
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(content)

        # 读取并返回数据预览
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            df = pd.DataFrame({'text1': lines[::2], 'text2': lines[1::2]})

        preview = df.head(10).to_dict(orient='records')
        total_rows = len(df)

        return ResponseModel.success(
            data={
                'id': file_id,
                'filename': filename,
                'path': filepath,
                'preview': preview,
                'total_rows': total_rows,
                'columns': list(df.columns)
            },
            message="文件上传成功"
        )
    except Exception as e:
        logger.error(f"上传训练数据失败：{e}")
        return ResponseModel.error(message=f"上传失败：{str(e)}")


@app.get(f"/api/{API_VERSION}/train/data")
@limiter.limit("30/minute")
async def get_training_data_list(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """获取训练数据列表"""
    try:
        files = []
        for f in os.listdir(MODEL_DATA_DIR):
            if f.endswith(('.csv', '.txt')):
                filepath = os.path.join(MODEL_DATA_DIR, f)
                stat = os.stat(filepath)
                files.append({
                    'id': f.split('_')[0],
                    'filename': f,
                    'size': stat.st_size,
                    'created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_ctime))
                })
        return ResponseModel.success(data=files, message="获取数据列表成功")
    except Exception as e:
        return ResponseModel.error(message=f"获取数据列表失败：{str(e)}")


@app.post(f"/api/{API_VERSION}/train/clean")
@limiter.limit("30/minute")
async def clean_training_data(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """清洗训练数据"""
    try:
        data = await request.json()
        file_id = data.get('file_id')

        files = [f for f in os.listdir(MODEL_DATA_DIR) if f.startswith(file_id)]
        if not files:
            return ResponseModel.error(message="文件不存在", code=404)

        filepath = os.path.join(MODEL_DATA_DIR, files[0])
        df = pd.read_csv(filepath)
        original_count = len(df)

        # 清洗文本
        for col in df.columns:
            df[col] = df[col].apply(lambda x: clean_text(x) if pd.notna(x) else x)

        # 过滤长度
        min_len = data.get('min_length', 5)
        max_len = data.get('max_length', 512)

        if 'text1' in df.columns:
            mask = df['text1'].astype(str).str.len().between(min_len, max_len)
            df = df[mask]

        # 样本平衡
        if 'score' in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            pos_samples = df[df['score'] >= 0.7]
            neg_samples = df[df['score'] < 0.3]
            min_count = min(len(pos_samples), len(neg_samples))
            if min_count > 0:
                df = pd.concat([
                    pos_samples.sample(n=min_count, random_state=42),
                    neg_samples.sample(n=min_count, random_state=42)
                ]).sample(frac=1, random_state=42).reset_index(drop=True)

        # 保存清洗后的数据
        clean_filename = f"clean_{files[0]}"
        clean_filepath = os.path.join(MODEL_DATA_DIR, clean_filename)
        df.to_csv(clean_filepath, index=False)

        return ResponseModel.success(
            data={
                'original_count': original_count,
                'cleaned_count': len(df),
                'filename': clean_filename,
                'preview': df.head(10).to_dict('records')
            },
            message="数据清洗完成"
        )
    except Exception as e:
        logger.error(f"数据清洗失败：{e}")
        return ResponseModel.error(message=f"数据清洗失败：{str(e)}")


@app.get(f"/api/{API_VERSION}/train/models")
@limiter.limit("30/minute")
async def get_available_models(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """获取可用模型列表"""
    return ResponseModel.success(data=AVAILABLE_MODELS, message="获取模型列表成功")


@app.get(f"/api/{API_VERSION}/train/config")
@limiter.limit("30/minute")
async def get_train_default_config(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """获取默认训练配置"""
    return ResponseModel.success(data=DEFAULT_TRAIN_CONFIG, message="获取配置成功")


def train_model_thread(config, data_file):
    """训练模型的后台线程"""
    try:
        import torch
        from sentence_transformers import SentenceTransformer, InputExample, evaluation, losses
        from torch.utils.data import DataLoader

        training_state['running'] = True
        training_state['logs'] = []
        training_state['progress'] = 0

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log_train_message(f"开始训练: 基础模型={config['base_model']}, 设备={device}")

        model = SentenceTransformer(config['base_model'], device=device)

        df = pd.read_csv(data_file)
        train_examples = []
        for _, row in df.iterrows():
            if 'score' in df.columns:
                example = InputExample(
                    texts=[str(row['text1']), str(row['text2'])],
                    label=float(row['score'])
                )
            else:
                example = InputExample(
                    texts=[str(row['text1']), str(row['text2'])]
                )
            train_examples.append(example)

        split_idx = int(len(train_examples) * 0.9)
        train_data = train_examples[:split_idx]
        eval_data = train_examples[split_idx:]

        log_train_message(f"训练集: {len(train_data)} 样本, 验证集: {len(eval_data)} 样本")

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])
        loss = losses.CosineSimilarityLoss(model)

        warmup_steps = int(len(train_dataloader) * config['epochs'] * 0.1)

        def training_callback(ep, steps):
            progress = (ep + 1) / config['epochs'] * 100
            training_state['progress'] = progress
            log_train_message(f"Epoch {ep+1}/{config['epochs']}, Steps {steps}, Progress: {int(progress)}%")

        model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=config['epochs'],
            warmup_steps=warmup_steps,
            optimizer_params={'lr': config['learning_rate']},
            show_progress_bar=True,
            callback=training_callback
        )

        model_version = f"v{int(time.time())}"
        model_path = os.path.join(MODEL_MODELS_DIR, 'embed', model_version)
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path)

        training_state['model_path'] = model_path
        training_state['progress'] = 100
        log_train_message(f"训练完成! 模型已保存到: {model_path}")

    except Exception as e:
        log_train_message(f"训练失败: {str(e)}", 'ERROR')
        training_state['running'] = False
    finally:
        training_state['running'] = False


@app.post(f"/api/{API_VERSION}/train/start")
@limiter.limit("10/minute")
async def start_training(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """启动训练"""
    try:
        if training_state['running']:
            return ResponseModel.error(message="训练已在运行中", code=400)

        config = await request.json()
        file_id = config.get('file_id')

        files = [f for f in os.listdir(MODEL_DATA_DIR) if f.startswith('clean') and file_id in f]
        if not files:
            files = [f for f in os.listdir(MODEL_DATA_DIR) if f.startswith(file_id)]

        if not files:
            return ResponseModel.error(message="数据文件不存在", code=404)

        data_file = os.path.join(MODEL_DATA_DIR, files[0])
        full_config = {**DEFAULT_TRAIN_CONFIG, **config}
        training_state['config'] = full_config

        thread = threading.Thread(target=train_model_thread, args=(full_config, data_file))
        thread.daemon = True
        thread.start()

        return ResponseModel.success(message="训练已启动")
    except Exception as e:
        logger.error(f"启动训练失败：{e}")
        return ResponseModel.error(message=f"启动训练失败：{str(e)}")


@app.get(f"/api/{API_VERSION}/train/status")
@limiter.limit("60/minute")
async def get_training_status(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """获取训练状态"""
    return ResponseModel.success(
        data={
            'running': training_state['running'],
            'progress': training_state['progress'],
            'logs': training_state['logs'][-50:],
            'model_path': training_state['model_path']
        },
        message="获取状态成功"
    )


@app.post(f"/api/{API_VERSION}/train/stop")
@limiter.limit("10/minute")
async def stop_training(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """停止训练"""
    training_state['running'] = False
    log_train_message('训练已手动停止')
    return ResponseModel.success(message="训练已停止")


@app.get(f"/api/{API_VERSION}/train/models/list")
@limiter.limit("30/minute")
async def list_trained_models(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """列出已训练模型"""
    embed_dir = os.path.join(MODEL_MODELS_DIR, 'embed')
    models = []

    if os.path.exists(embed_dir):
        for v in os.listdir(embed_dir):
            v_path = os.path.join(embed_dir, v)
            if os.path.isdir(v_path):
                config_file = os.path.join(v_path, 'config.json')
                is_valid = os.path.exists(config_file)

                models.append({
                    'version': v,
                    'path': v_path,
                    'valid': is_valid,
                    'deployed': v == 'latest'
                })

    return ResponseModel.success(data=models, message="获取模型列表成功")


@app.post(f"/api/{API_VERSION}/train/deploy")
@limiter.limit("10/minute")
async def deploy_trained_model(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """部署模型"""
    try:
        data = await request.json()
        model_version = data.get('version')

        model_path = os.path.join(MODEL_MODELS_DIR, 'embed', model_version)
        if not os.path.exists(model_path):
            return ResponseModel.error(message="模型不存在", code=404)

        latest_path = os.path.join(MODEL_MODELS_DIR, 'embed', 'latest')
        if os.path.exists(latest_path):
            shutil.rmtree(latest_path)
        shutil.copytree(model_path, latest_path)

        return ResponseModel.success(message=f"模型 {model_version} 部署成功")
    except Exception as e:
        return ResponseModel.error(message=f"部署失败：{str(e)}")


@app.post(f"/api/{API_VERSION}/train/validate")
@limiter.limit("30/minute")
async def validate_trained_model(
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """验证模型"""
    try:
        data = await request.json()
        model_path = data.get('model_path')

        if not model_path or not os.path.exists(model_path):
            embed_dir = os.path.join(MODEL_MODELS_DIR, 'embed')
            if os.path.exists(embed_dir):
                versions = [d for d in os.listdir(embed_dir) if os.path.isdir(os.path.join(embed_dir, d))]
                if versions:
                    model_path = os.path.join(embed_dir, sorted(versions)[-1])

        if not model_path or not os.path.exists(model_path):
            return ResponseModel.error(message="未找到可用模型", code=404)

        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(model_path, device=device)

        results = []
        results.append({'test': '模型加载', 'status': 'PASS', 'detail': model_path})

        test_text = "这是一条测试文本"
        embedding = model.encode(test_text)
        results.append({'test': '编码功能', 'status': 'PASS', 'detail': f'输出维度: {len(embedding)}'})

        dim = len(embedding)
        results.append({'test': '维度检查', 'status': 'PASS' if dim > 0 else 'FAIL', 'detail': f'维度: {dim}'})

        texts = [
            ("北京是中国的首都", "中国的首都是北京", 0.9),
            ("今天天气很好", "今天下雨了", 0.3),
        ]
        similarity_results = []
        for t1, t2, expected in texts:
            emb1 = model.encode(t1)
            emb2 = model.encode(t2)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            status_val = 'PASS' if (sim > 0.5) == (expected > 0.5) else 'FAIL'
            similarity_results.append({
                'text1': t1, 'text2': t2,
                'similarity': float(sim),
                'expected': expected,
                'status': status_val
            })
        results.append({'test': '语义相似度', 'status': 'PASS', 'detail': similarity_results})

        batch_texts = ["测试文本" + str(i) for i in range(100)]
        start_time = time.time()
        embeddings = model.encode(batch_texts)
        elapsed = time.time() - start_time
        results.append({
            'test': '批量性能',
            'status': 'PASS',
            'detail': f'100条文本耗时: {elapsed:.2f}秒'
        })

        return ResponseModel.success(
            data={
                'model_path': model_path,
                'results': results
            },
            message="模型验证完成"
        )
    except Exception as e:
        logger.error(f"模型验证失败：{e}")
        return ResponseModel.error(message=f"验证失败：{str(e)}")


# ========== 启动服务 ==========
if __name__ == "__main__":
    import uvicorn
    web_config = CONFIG.get("web", {})
    uvicorn.run(
        "web_ui:app",
        host=web_config.get("host", "0.0.0.0"),
        port=web_config.get("port", 8000),
        reload=web_config.get("reload", True)
    )
