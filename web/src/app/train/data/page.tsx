'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, FileText, Trash2, RefreshCw, Sparkles } from 'lucide-react';
import { uploadTrainingData, getTrainingDataList, cleanTrainingData, augmentTrainingData } from '@/lib/train-api';
import { KbExporter } from '@/components/train/KbExporter';

interface TrainingDataItem {
  id: string;
  filename: string;
  size: number;
  created: string;
}

export default function TrainDataPage() {
  const [dataList, setDataList] = useState<TrainingDataItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [cleaningId, setCleaningId] = useState<string | null>(null);
  const [augmentingId, setAugmentingId] = useState<string | null>(null);
  const [cleanConfig, setCleanConfig] = useState({
    min_length: 5,
    max_length: 512,
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const data = await getTrainingDataList();
      setDataList(data);
    } catch (error) {
      console.error('加载失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      const result = await uploadTrainingData(file);
      if (result.success) {
        await loadData();
      } else {
        alert('上传失败: ' + result.message);
      }
    } catch (error) {
      console.error('上传失败:', error);
      alert('上传失败');
    } finally {
      setUploading(false);
    }
  };

  const handleClean = async (fileId: string) => {
    try {
      setCleaningId(fileId);
      const result = await cleanTrainingData(fileId, cleanConfig);
      if (result.success) {
        alert(`清洗完成: ${result.data.original_count} → ${result.data.cleaned_count}`);
        await loadData();
      } else {
        alert('清洗失败: ' + result.message);
      }
    } catch (error) {
      console.error('清洗失败:', error);
    } finally {
      setCleaningId(null);
    }
  };

  const handleAugment = async (fileId: string) => {
    try {
      setAugmentingId(fileId);
      const result = await augmentTrainingData(fileId, 'duplicate');
      if (result.success) {
        alert(`增强完成: ${result.data.original_count} → ${result.data.augmented_count}`);
        await loadData();
      } else {
        alert('增强失败: ' + result.message);
      }
    } catch (error) {
      console.error('增强失败:', error);
    } finally {
      setAugmentingId(null);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="container mx-auto py-8">
      {/* 上传区域 */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>上传训练数据</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Input
              type="file"
              accept=".csv,.txt"
              onChange={handleUpload}
              className="max-w-xs"
              disabled={uploading}
            />
            <Button disabled={uploading}>
              <Upload className="mr-2 h-4 w-4" />
              {uploading ? '上传中...' : '上传'}
            </Button>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            支持 CSV 或 TXT 格式，CSV需包含 text1, text2, score 列
          </p>
        </CardContent>
      </Card>

      {/* 知识库导出 */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>从知识库导出</CardTitle>
        </CardHeader>
        <CardContent>
          <KbExporter />
        </CardContent>
      </Card>

      {/* 数据列表 */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>训练数据列表</CardTitle>
          <Button variant="outline" size="sm" onClick={loadData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            刷新
          </Button>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-8 text-muted-foreground">加载中...</div>
          ) : dataList.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">暂无训练数据</div>
          ) : (
            <div className="space-y-4">
              {dataList.map((item) => (
                <div key={item.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <FileText className="h-8 w-8 text-muted-foreground" />
                    <div>
                      <p className="font-medium">{item.filename}</p>
                      <p className="text-sm text-muted-foreground">
                        {formatSize(item.size)} · {item.created}
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleClean(item.id)}
                      disabled={cleaningId === item.id}
                    >
                      <Sparkles className="mr-2 h-4 w-4" />
                      {cleaningId === item.id ? '清洗中...' : '清洗'}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleAugment(item.id)}
                      disabled={augmentingId === item.id}
                    >
                      <RefreshCw className="mr-2 h-4 w-4" />
                      {augmentingId === item.id ? '增强中...' : '增强'}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 清洗配置 */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>清洗配置</CardTitle>
        </CardHeader>
        <CardContent className="flex gap-4 items-end">
          <div className="space-y-2">
            <Label>最小长度</Label>
            <Input
              type="number"
              value={cleanConfig.min_length}
              onChange={(e) => setCleanConfig({ ...cleanConfig, min_length: Number(e.target.value) })}
              className="w-32"
            />
          </div>
          <div className="space-y-2">
            <Label>最大长度</Label>
            <Input
              type="number"
              value={cleanConfig.max_length}
              onChange={(e) => setCleanConfig({ ...cleanConfig, max_length: Number(e.target.value) })}
              className="w-32"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
