'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { AlertCircle, Play, Square, RefreshCw, CheckCircle } from 'lucide-react';
import { getTrainingStatus, stopTraining } from '@/lib/train-api';

interface TrainingStatus {
  running: boolean;
  progress: number;
  logs: string[];
  model_path: string | null;
}

export default function TrainRunPage() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [stopping, setStopping] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchStatus = async () => {
    try {
      const data = await getTrainingStatus();
      setStatus(data);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('获取状态失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleStop = async () => {
    try {
      setStopping(true);
      await stopTraining();
      await fetchStatus();
    } catch (error) {
      console.error('停止失败:', error);
    } finally {
      setStopping(false);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            训练状态
            {loading ? (
              <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
            ) : status?.running ? (
              <span className="flex items-center gap-1 text-sm text-green-600">
                <span className="w-2 h-2 bg-green-600 rounded-full animate-pulse" />
                运行中
              </span>
            ) : status?.progress === 100 ? (
              <span className="flex items-center gap-1 text-sm text-green-600">
                <CheckCircle className="w-4 h-4" />
                已完成
              </span>
            ) : (
              <span className="text-sm text-muted-foreground">空闲</span>
            )}
          </CardTitle>
          {lastUpdate && (
            <p className="text-xs text-muted-foreground">
              最后更新: {lastUpdate.toLocaleTimeString()}
            </p>
          )}
        </CardHeader>
        <CardContent className="space-y-6">
          {loading ? (
            <div className="text-center py-8 text-muted-foreground">加载中...</div>
          ) : status?.running || status?.progress === 100 ? (
            <>
              {/* 进度条 */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>训练进度</span>
                  <span>{Math.round(status?.progress || 0)}%</span>
                </div>
                <Progress value={status?.progress || 0} className="h-3" />
              </div>

              {/* 模型路径 */}
              {status?.model_path && (
                <div className="p-3 bg-muted rounded-lg">
                  <p className="text-sm font-medium">模型保存路径</p>
                  <p className="text-sm text-muted-foreground font-mono">{status.model_path}</p>
                </div>
              )}

              {/* 日志 */}
              <div className="space-y-2">
                <p className="text-sm font-medium">训练日志</p>
                <div className="bg-black text-green-400 font-mono text-xs p-4 rounded-lg max-h-80 overflow-auto">
                  {status?.logs && status.logs.length > 0 ? (
                    status.logs.map((log, i) => (
                      <div key={i}>{log}</div>
                    ))
                  ) : (
                    <div className="text-gray-500">暂无日志</div>
                  )}
                </div>
              </div>

              {/* 停止按钮 */}
              {status?.running && (
                <Button
                  variant="destructive"
                  onClick={handleStop}
                  disabled={stopping}
                  className="w-full"
                >
                  <Square className="mr-2 h-4 w-4" />
                  {stopping ? '停止中...' : '停止训练'}
                </Button>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">当前没有训练任务运行</p>
              <p className="text-sm text-muted-foreground mt-2">
                请在"训练配置"页面启动训练
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
