'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { getTrainingDataList, getAvailableModels, getDefaultConfig, startTraining } from '@/lib/train-api';
import { useRouter } from 'next/navigation';

interface ModelOption {
  id: string;
  name: string;
}

interface DataOption {
  id: string;
  filename: string;
}

export default function TrainConfigPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [dataFiles, setDataFiles] = useState<DataOption[]>([]);
  const [config, setConfig] = useState({
    file_id: '',
    base_model: '',
    epochs: 5,
    batch_size: 16,
    learning_rate: 2e-5,
  });

  useEffect(() => {
    loadOptions();
  }, []);

  const loadOptions = async () => {
    try {
      setLoading(true);
      const [modelsData, dataData, defaultConfig] = await Promise.all([
        getAvailableModels(),
        getTrainingDataList(),
        getDefaultConfig(),
      ]);
      setModels(modelsData || []);
      setDataFiles(dataData || []);
      if (defaultConfig) {
        setConfig(prev => ({
          ...prev,
          base_model: defaultConfig.base_model || 'BAAI/bge-m3',
          epochs: defaultConfig.epochs || 5,
          batch_size: defaultConfig.batch_size || 16,
          learning_rate: defaultConfig.learning_rate || 2e-5,
        }));
      }
    } catch (error) {
      console.error('加载配置失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = async () => {
    if (!config.file_id) {
      alert('请选择训练数据');
      return;
    }
    if (!config.base_model) {
      alert('请选择基础模型');
      return;
    }

    try {
      setStarting(true);
      const result = await startTraining(config);
      if (result.success) {
        alert('训练已启动');
        router.push('/train/run');
      } else {
        alert('启动失败: ' + result.message);
      }
    } catch (error) {
      console.error('启动训练失败:', error);
      alert('启动训练失败');
    } finally {
      setStarting(false);
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto py-8">
        <div className="text-center text-muted-foreground">加载中...</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>训练配置</CardTitle>
          <CardDescription>配置训练参数并启动模型训练</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* 训练数据选择 */}
          <div className="space-y-2">
            <Label>训练数据</Label>
            <Select
              value={config.file_id}
              onValueChange={(v) => setConfig({ ...config, file_id: v })}
            >
              <SelectTrigger>
                <SelectValue placeholder="选择训练数据文件" />
              </SelectTrigger>
              <SelectContent>
                {dataFiles.map((f) => (
                  <SelectItem key={f.id} value={f.id}>{f.filename}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            {dataFiles.length === 0 && (
              <p className="text-sm text-muted-foreground">请先在"数据管理"中上传训练数据</p>
            )}
          </div>

          {/* 基础模型选择 */}
          <div className="space-y-2">
            <Label>基础模型</Label>
            <Select
              value={config.base_model}
              onValueChange={(v) => setConfig({ ...config, base_model: v })}
            >
              <SelectTrigger>
                <SelectValue placeholder="选择基础模型" />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m.id} value={m.id}>{m.name || m.id}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 参数配置 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label>Epochs (训练轮数)</Label>
              <Input
                type="number"
                value={config.epochs}
                onChange={(e) => setConfig({ ...config, epochs: Number(e.target.value) })}
                min={1}
                max={100}
              />
            </div>
            <div className="space-y-2">
              <Label>Batch Size (批大小)</Label>
              <Input
                type="number"
                value={config.batch_size}
                onChange={(e) => setConfig({ ...config, batch_size: Number(e.target.value) })}
                min={1}
                max={128}
              />
            </div>
            <div className="space-y-2">
              <Label>Learning Rate (学习率)</Label>
              <Input
                type="number"
                step="0.00001"
                value={config.learning_rate}
                onChange={(e) => setConfig({ ...config, learning_rate: Number(e.target.value) })}
              />
            </div>
          </div>

          {/* 启动按钮 */}
          <div className="flex gap-4 pt-4">
            <Button
              onClick={handleStartTraining}
              disabled={starting || !config.file_id || !config.base_model}
              size="lg"
            >
              {starting ? '启动中...' : '开始训练'}
            </Button>
            <Button
              variant="outline"
              onClick={() => router.push('/train/run')}
            >
              查看训练状态
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
