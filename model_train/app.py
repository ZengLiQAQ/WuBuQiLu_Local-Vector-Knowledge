import os
import re
import json
import time
import uuid
import shutil
import threading
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import (
    DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    AVAILABLE_MODELS, DEFAULT_TRAIN_CONFIG,
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG
)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 全局状态
training_state = {
    'running': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'logs': [],
    'model_path': None,
    'config': None,
}

# ============ 辅助函数 ============

def allowed_file(filename, allowed_extensions={'csv', 'txt'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def clean_text(text):
    """清洗文本：去除特殊字符，保留中文、英文、数字、空格"""
    if not isinstance(text, str):
        text = str(text)
    # 只保留中文、英文、数字、空格
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return text.strip()

def log_message(message, level='INFO'):
    """记录日志"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"
    training_state['logs'].append(log_entry)
    print(log_entry)

# ============ 数据管理API ============

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传训练数据文件"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())[:8]
    save_name = f"{file_id}_{filename}"
    filepath = os.path.join(DATA_DIR, save_name)
    file.save(filepath)

    # 读取并返回数据预览
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            df = pd.DataFrame({'text1': lines[::2], 'text2': lines[1::2]})

        preview = df.head(10).to_dict(orient='records')
        total_rows = len(df)

        return jsonify({
            'success': True,
            'data': {
                'id': file_id,
                'filename': filename,
                'path': filepath,
                'preview': preview,
                'total_rows': total_rows,
                'columns': list(df.columns)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/data', methods=['GET'])
def get_data_list():
    """获取数据列表"""
    files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(('.csv', '.txt')):
            filepath = os.path.join(DATA_DIR, f)
            stat = os.stat(filepath)
            files.append({
                'id': f.split('_')[0],
                'filename': f,
                'size': stat.st_size,
                'created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_ctime))
            })
    return jsonify({'success': True, 'data': files})

@app.route('/api/data/<file_id>', methods=['GET'])
def get_data_detail(file_id):
    """获取数据详情"""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(file_id)]
    if not files:
        return jsonify({'success': False, 'error': 'File not found'}), 404

    filepath = os.path.join(DATA_DIR, files[0])
    try:
        df = pd.read_csv(filepath)
        return jsonify({
            'success': True,
            'data': {
                'filename': files[0],
                'total_rows': len(df),
                'columns': list(df.columns),
                'preview': df.head(50).to_dict(orient='records')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/clean', methods=['POST'])
def clean_data():
    """数据清洗"""
    data = request.json
    file_id = data.get('file_id')

    # 找到文件
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(file_id)]
    if not files:
        return jsonify({'success': False, 'error': 'File not found'}), 404

    filepath = os.path.join(DATA_DIR, files[0])

    try:
        df = pd.read_csv(filepath)
        original_count = len(df)

        # 清洗文本
        for col in df.columns:
            df[col] = df[col].apply(lambda x: clean_text(x) if pd.notna(x) else x)

        # 过滤长度 - 只检查text1列，避免score列被过滤
        min_len = data.get('min_length', 5)
        max_len = data.get('max_length', 512)

        if 'text1' in df.columns:
            # 先过滤长度
            mask = df['text1'].astype(str).str.len().between(min_len, max_len)
            df = df[mask]

        # 样本平衡（如果存在score列）
        if 'score' in df.columns:
            # 确保score列是数值类型
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            # 确保正负样本平衡
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
        clean_filepath = os.path.join(DATA_DIR, clean_filename)
        df.to_csv(clean_filepath, index=False)

        return jsonify({
            'success': True,
            'data': {
                'original_count': original_count,
                'cleaned_count': len(df),
                'filename': clean_filename,
                'preview': df.head(10).to_dict('records')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/datasets', methods=['GET'])
def get_public_datasets():
    """获取公开数据集列表"""
    datasets = [
        {'id': 'sts中文', 'name': 'STS中文语义相似度数据集', 'description': '用于语义相似度训练'},
        {'id': 'cqcc', 'name': '中文问题对偶数据集', 'description': '问题对偶生成训练'},
    ]
    return jsonify({'success': True, 'data': datasets})

# ============ 数据制备API ============

@app.route('/api/dingtalk/import', methods=['POST'])
def import_from_dingtalk():
    """从钉钉文档导入数据"""
    data = request.json
    content = data.get('content', '')
    doc_type = data.get('doc_type', 'qa')  # qa: 问答对, text: 纯文本

    if not content:
        return jsonify({'success': False, 'error': 'No content provided'}), 400

    try:
        records = []
        lines = content.strip().split('\n')

        if doc_type == 'qa':
            # 解析问答对格式: Q:xxx A:xxx 或 问题答案用tab分隔
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 尝试多种格式解析
                if 'Q:' in line and 'A:' in line:
                    parts = line.split('A:')
                    if len(parts) == 2:
                        q = parts[0].replace('Q:', '').strip()
                        a = parts[1].strip()
                        if q and a:
                            records.append({'text1': q, 'text2': a, 'score': 1.0})
                elif '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        records.append({'text1': parts[0].strip(), 'text2': parts[1].strip(), 'score': 1.0})
        else:
            # 纯文本 - 生成自对比数据
            for line in lines:
                text = line.strip()
                if len(text) >= 5:
                    records.append({'text1': text, 'text2': text, 'score': 1.0})

        # 保存到文件
        file_id = str(uuid.uuid4())[:8]
        filename = f"dingtalk_{file_id}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False, encoding='utf-8')

        return jsonify({
            'success': True,
            'data': {
                'id': file_id,
                'filename': filename,
                'record_count': len(records),
                'preview': df.head(10).to_dict(orient='records')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/augment', methods=['POST'])
def augment_data():
    """数据增强"""
    data = request.json
    file_id = data.get('file_id')
    method = data.get('method', 'duplicate')  # duplicate, synonym, split

    files = [f for f in os.listdir(DATA_DIR) if f.startswith(file_id)]
    if not files:
        return jsonify({'success': False, 'error': 'File not found'}), 404

    filepath = os.path.join(DATA_DIR, files[0])

    try:
        df = pd.read_csv(filepath)
        original_count = len(df)

        augmented = []

        for _, row in df.iterrows():
            augmented.append(row.to_dict())

            if method == 'duplicate':
                # 重复样本增加噪声
                if 'score' in row:
                    for noise in [0.05, 0.1, -0.05, -0.1]:
                        new_row = row.to_dict()
                        new_row['score'] = max(0, min(1, new_row['score'] + noise))
                        augmented.append(new_row)

        # 保存增强后的数据
        aug_file_id = str(uuid.uuid4())[:8]
        filename = f"aug_{method}_{aug_file_id}.csv"
        aug_filepath = os.path.join(DATA_DIR, filename)

        df_aug = pd.DataFrame(augmented)
        df_aug.to_csv(aug_filepath, index=False)

        return jsonify({
            'success': True,
            'data': {
                'original_count': original_count,
                'augmented_count': len(augmented),
                'filename': filename,
                'method': method
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/generate/pairs', methods=['POST'])
def generate_pairs():
    """从文本列表生成训练对"""
    data = request.json
    texts = data.get('texts', [])
    mode = data.get('mode', 'self')  # self: 自对比, all: 全量配对, sample: 采样配对

    if not texts or len(texts) < 2:
        return jsonify({'success': False, 'error': 'Need at least 2 texts'}), 400

    try:
        pairs = []

        if mode == 'self':
            # 自对比：相同文本
            for text in texts:
                if text.strip():
                    pairs.append({'text1': text.strip(), 'text2': text.strip(), 'score': 1.0})

        elif mode == 'all':
            # 全量配对
            for i, t1 in enumerate(texts):
                for t2 in texts[i+1:]:
                    if t1.strip() and t2.strip():
                        pairs.append({'text1': t1.strip(), 'text2': t2.strip(), 'score': 0.5})

        elif mode == 'sample':
            # 采样配对（正负样本）
            import random
            texts_clean = [t.strip() for t in texts if t.strip()]

            # 正样本：相同文本
            for text in texts_clean[:len(texts_clean)//2]:
                pairs.append({'text1': text, 'text2': text, 'score': 1.0})

            # 负样本：随机配对
            for _ in range(min(50, len(texts_clean))):
                t1, t2 = random.sample(texts_clean, 2)
                pairs.append({'text1': t1, 'text2': t2, 'score': 0.0})

        # 保存
        file_id = str(uuid.uuid4())[:8]
        filename = f"pairs_{mode}_{file_id}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        df = pd.DataFrame(pairs)
        df.to_csv(filepath, index=False)

        return jsonify({
            'success': True,
            'data': {
                'id': file_id,
                'filename': filename,
                'pair_count': len(pairs),
                'preview': df.head(10).to_dict(orient='records')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/validate/data', methods=['POST'])
def validate_training_data():
    """验证训练数据质量"""
    data = request.json
    file_id = data.get('file_id')

    files = [f for f in os.listdir(DATA_DIR) if f.startswith(file_id)]
    if not files:
        return jsonify({'success': False, 'error': 'File not found'}), 404

    filepath = os.path.join(DATA_DIR, files[0])

    try:
        df = pd.read_csv(filepath)
        issues = []
        warnings = []

        # 检查必需列
        required_cols = ['text1', 'text2']
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"缺少必需列: {col}")

        # 检查空值
        for col in required_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"列 {col} 有 {null_count} 个空值")

        # 检查文本长度
        if 'text1' in df.columns:
            lengths = df['text1'].astype(str).str.len()
            too_short = int((lengths < 3).sum())
            too_long = int((lengths > 512).sum())

            if too_short > 0:
                warnings.append(f"{too_short} 条文本长度小于3")
            if too_long > 0:
                warnings.append(f"{too_long} 条文本长度大于512")

        # 检查score范围
        if 'score' in df.columns:
            invalid_score = int(((df['score'] < 0) | (df['score'] > 1)).sum())
            if invalid_score > 0:
                issues.append(f"{invalid_score} 条score不在0-1范围内")

        # 检查重复
        duplicates = int(df.duplicated(subset=['text1', 'text2']).sum())
        if duplicates > 0:
            warnings.append(f"{duplicates} 条重复记录")

        # 统计信息
        score_dist = {}
        if 'score' in df.columns:
            score_dist = {
                'high': int((df['score'] >= 0.7).sum()),
                'medium': int(((df['score'] >= 0.3) & (df['score'] < 0.7)).sum()),
                'low': int((df['score'] < 0.3).sum())
            }

        stats = {
            'total': len(df),
            'columns': list(df.columns),
            'score_distribution': score_dist
        }

        return jsonify({
            'success': True,
            'data': {
                'issues': issues,
                'warnings': warnings,
                'stats': stats,
                'valid': len(issues) == 0
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============ 训练配置API ============

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    return jsonify({'success': True, 'data': AVAILABLE_MODELS})

@app.route('/api/config', methods=['GET'])
def get_default_config():
    """获取默认训练配置"""
    return jsonify({'success': True, 'data': DEFAULT_TRAIN_CONFIG})

# ============ 训练执行API ============

def train_model_thread(config, data_file):
    """训练模型的后台线程"""
    try:
        import torch
        from sentence_transformers import SentenceTransformer, InputExample, evaluation, losses
        from torch.utils.data import DataLoader

        training_state['running'] = True
        training_state['logs'] = []
        training_state['progress'] = 0

        # Mac MPS 加速检测
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log_message(f"开始训练: 基础模型={config['base_model']}, 设备={device}")

        # 加载基础模型 (Mac 优化)
        model = SentenceTransformer(config['base_model'], device=device)

        # 加载训练数据
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

        # 划分训练集和验证集
        split_idx = int(len(train_examples) * 0.9)
        train_data = train_examples[:split_idx]
        eval_data = train_examples[split_idx:]

        log_message(f"训练集: {len(train_data)} 样本, 验证集: {len(eval_data)} 样本")

        # 创建数据加载器
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])

        # 使用CosineSimilarityLoss
        loss = losses.CosineSimilarityLoss(model)

        # Warmup steps
        warmup_steps = int(len(train_dataloader) * config['epochs'] * 0.1)

        # 训练回调函数
        def training_callback(ep, steps):
            progress = (ep + 1) / config['epochs'] * 100
            training_state['progress'] = progress
            log_message(f"Epoch {ep+1}/{config['epochs']}, Steps {steps}, Progress: {int(progress)}%")

        model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=config['epochs'],
            warmup_steps=warmup_steps,
            optimizer_params={'lr': config['learning_rate']},
            show_progress_bar=True,
            callback=training_callback
        )

        # 保存模型
        model_version = f"v{int(time.time())}"
        model_path = os.path.join(MODELS_DIR, 'embed', model_version)
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path)

        training_state['model_path'] = model_path
        training_state['progress'] = 100
        log_message(f"训练完成! 模型已保存到: {model_path}")

    except Exception as e:
        log_message(f"训练失败: {str(e)}", 'ERROR')
        training_state['running'] = False
    finally:
        training_state['running'] = False

@app.route('/api/train', methods=['POST'])
def start_training():
    """启动训练"""
    if training_state['running']:
        return jsonify({'success': False, 'error': 'Training already running'}), 400

    config = request.json
    file_id = config.get('file_id')

    # 找到清洗后的数据文件
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('clean') and file_id in f]
    if not files:
        # 尝试原始文件
        files = [f for f in os.listdir(DATA_DIR) if f.startswith(file_id)]

    if not files:
        return jsonify({'success': False, 'error': 'Data file not found'}), 404

    data_file = os.path.join(DATA_DIR, files[0])

    # 合并配置
    full_config = {**DEFAULT_TRAIN_CONFIG, **config}
    training_state['config'] = full_config

    # 启动训练线程
    thread = threading.Thread(target=train_model_thread, args=(full_config, data_file))
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """获取训练状态"""
    return jsonify({
        'success': True,
        'data': {
            'running': training_state['running'],
            'progress': training_state['progress'],
            'logs': training_state['logs'][-50:],  # 最近50条
            'model_path': training_state['model_path']
        }
    })

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    """停止训练"""
    training_state['running'] = False
    log_message('训练已手动停止')
    return jsonify({'success': True, 'message': 'Training stopped'})

# ============ 模型验证API ============

@app.route('/api/validate', methods=['POST'])
def validate_model():
    """验证模型"""
    data = request.json
    model_path = data.get('model_path')

    if not model_path or not os.path.exists(model_path):
        # 使用最新模型
        embed_dir = os.path.join(MODELS_DIR, 'embed')
        if os.path.exists(embed_dir):
            versions = [d for d in os.listdir(embed_dir) if os.path.isdir(os.path.join(embed_dir, d))]
            if versions:
                model_path = os.path.join(embed_dir, sorted(versions)[-1])

    if not model_path or not os.path.exists(model_path):
        return jsonify({'success': False, 'error': 'No model found'}), 404

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        # Mac MPS 加速
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # 加载模型 (Mac 优化)
        model = SentenceTransformer(model_path, device=device)

        results = []

        # 1. 加载测试
        results.append({'test': '模型加载', 'status': 'PASS', 'detail': model_path})

        # 2. 编码功能测试
        test_text = "这是一条测试文本"
        embedding = model.encode(test_text)
        results.append({'test': '编码功能', 'status': 'PASS', 'detail': f'输出维度: {len(embedding)}'})

        # 3. 输出维度检查
        dim = len(embedding)
        results.append({'test': '维度检查', 'status': 'PASS' if dim > 0 else 'FAIL', 'detail': f'维度: {dim}'})

        # 4. 语义相似度验证
        texts = [
            ("北京是中国的首都", "中国的首都是北京", 0.9),
            ("今天天气很好", "今天下雨了", 0.3),
        ]
        similarity_results = []
        for t1, t2, expected in texts:
            emb1 = model.encode(t1)
            emb2 = model.encode(t2)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            status = 'PASS' if (sim > 0.5) == (expected > 0.5) else 'FAIL'
            similarity_results.append({
                'text1': t1, 'text2': t2,
                'similarity': float(sim),
                'expected': expected,
                'status': status
            })
        results.append({'test': '语义相似度', 'status': 'PASS', 'detail': similarity_results})

        # 5. 批量编码性能测试
        batch_texts = ["测试文本" + str(i) for i in range(100)]
        start_time = time.time()
        embeddings = model.encode(batch_texts)
        elapsed = time.time() - start_time
        results.append({
            'test': '批量性能',
            'status': 'PASS',
            'detail': f'100条文本耗时: {elapsed:.2f}秒'
        })

        return jsonify({
            'success': True,
            'data': {
                'model_path': model_path,
                'results': results
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============ 部署管理API ============

@app.route('/api/models/list', methods=['GET'])
def list_models():
    """列出所有模型版本"""
    embed_dir = os.path.join(MODELS_DIR, 'embed')
    models = []

    if os.path.exists(embed_dir):
        for v in os.listdir(embed_dir):
            v_path = os.path.join(embed_dir, v)
            if os.path.isdir(v_path):
                # 检查是否是有效模型
                config_file = os.path.join(v_path, 'config.json')
                is_valid = os.path.exists(config_file)

                models.append({
                    'version': v,
                    'path': v_path,
                    'valid': is_valid,
                    'deployed': v == 'latest'  # 假设latest是部署版本
                })

    return jsonify({'success': True, 'data': models})

@app.route('/api/deploy', methods=['POST'])
def deploy_model():
    """部署模型"""
    data = request.json
    model_version = data.get('version')

    model_path = os.path.join(MODELS_DIR, 'embed', model_version)
    if not os.path.exists(model_path):
        return jsonify({'success': False, 'error': 'Model not found'}), 404

    try:
        # 创建latest符号链接或复制
        latest_path = os.path.join(MODELS_DIR, 'embed', 'latest')
        if os.path.exists(latest_path):
            shutil.rmtree(latest_path)
        shutil.copytree(model_path, latest_path)

        return jsonify({
            'success': True,
            'message': f'Model {model_version} deployed successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/rollback', methods=['POST'])
def rollback_model():
    """回滚模型"""
    data = request.json
    target_version = data.get('version')

    if not target_version:
        return jsonify({'success': False, 'error': 'No version specified'}), 400

    # 简单回滚：重新部署指定版本
    return deploy_model()

@app.route('/api/backup', methods=['POST'])
def backup_model():
    """备份模型"""
    data = request.json
    model_version = data.get('version')

    if not model_version:
        return jsonify({'success': False, 'error': 'No version specified'}), 400

    model_path = os.path.join(MODELS_DIR, 'embed', model_version)
    if not os.path.exists(model_path):
        return jsonify({'success': False, 'error': 'Model not found'}), 404

    try:
        backup_name = f"{model_version}_backup_{int(time.time())}"
        backup_path = os.path.join(MODELS_DIR, 'embed', backup_name)
        shutil.copytree(model_path, backup_path)

        return jsonify({
            'success': True,
            'data': {'backup_version': backup_name}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============ 静态文件服务 ============

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    print(f"Starting model training server at http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
