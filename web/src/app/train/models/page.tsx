'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, Upload, Check, AlertCircle, Trash2 } from 'lucide-react';
import { getTrainedModels, deployModel } from '@/lib/train-api';

interface TrainedModel {
  version: string;
  path: string;
  valid: boolean;
  deployed: boolean;
}

export default function TrainModelsPage() {
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [deploying, setDeploying] = useState<string | null>(null);

  const loadModels = async () => {
    try {
      setLoading(true);
      const data = await getTrainedModels();
      setModels(data);
    } catch (error) {
      console.error('加载模型失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleDeploy = async (version: string) => {
    try {
      setDeploying(version);
      const result = await deployModel(version);
      if (result.success) {
        alert(`模型 ${version} 部署成功`);
        await loadModels();
      } else {
        alert('部署失败: ' + result.message);
      }
    } catch (error) {
      console.error('部署失败:', error);
      alert('部署失败');
    } finally {
      setDeploying(null);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>已训练模型</CardTitle>
            <CardDescription>管理和部署训练好的模型</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={loadModels} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            刷新
          </Button>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-8 text-muted-foreground">加载中...</div>
          ) : models.length === 0 ? (
            <div className="text-center py-12">
              <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">暂无训练模型</p>
              <p className="text-sm text-muted-foreground mt-2">
                请先在"训练配置"页面启动训练
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {models.map((model) => (
                <div key={model.version} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                      <span className="text-lg font-bold text-primary">
                        {model.version.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium flex items-center gap-2">
                        {model.version}
                        {model.deployed && (
                          <Badge variant="secondary" className="text-xs">
                            <Check className="w-3 h-3 mr-1" />
                            已部署
                          </Badge>
                        )}
                      </p>
                      <p className="text-sm text-muted-foreground font-mono truncate max-w-md">
                        {model.path}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {model.valid ? (
                      <Badge variant="outline" className="text-xs">有效</Badge>
                    ) : (
                      <Badge variant="destructive" className="text-xs">无效</Badge>
                    )}
                    <Button
                      size="sm"
                      onClick={() => handleDeploy(model.version)}
                      disabled={deploying === model.version || model.deployed}
                    >
                      <Upload className="mr-2 h-4 w-4" />
                      {deploying === model.version ? '部署中...' : model.deployed ? '已部署' : '部署'}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
