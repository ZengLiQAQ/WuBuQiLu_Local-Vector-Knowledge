import os

# 项目根目录 (向上找两层到主项目)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'model_train', 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'model_train', 'models')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'model_train', 'checkpoints')
LOGS_DIR = os.path.join(BASE_DIR, 'model_train', 'logs')

# 确保目录存在
for d in [DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
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

# Flask配置 - 使用不同的端口避免冲突
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5002
FLASK_DEBUG = True
