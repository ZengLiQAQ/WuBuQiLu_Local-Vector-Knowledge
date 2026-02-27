
import os
import time
import hashlib
import asyncio
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

# 导入核心知识库类
from LocalVectorKB import LocalVectorKB, load_config


# ========== 初始化配置 ==========
CONFIG = load_config()
API_VERSION = "v1"
API_KEY = CONFIG["web"].get("api_key", "local_kb_2026_secure_key")
TEMP_DIR = "./temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


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

        # 保存临时文件
        file_path = os.path.join(TEMP_DIR, file.filename)
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
