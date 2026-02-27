'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Key, Eye, EyeOff, Trash2, AlertTriangle } from 'lucide-react';
import { setApiKey, getStoredApiKey, clearApiKey, clearKnowledgeBase } from '@/lib/api';

export default function SettingsPage() {
  const [apiKey, setApiKeyInput] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [saved, setSaved] = useState(false);
  const [clearing, setClearing] = useState(false);

  // 加载已保存的 API Key
  useState(() => {
    const savedKey = getStoredApiKey();
    if (savedKey) {
      setApiKey(savedKey);
    }
  });

  const handleSaveApiKey = () => {
    if (apiKey.trim()) {
      setApiKey(apiKey.trim());
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    }
  };

  const handleClearApiKey = () => {
    clearApiKey();
    setApiKey('');
  };

  const handleClearKnowledgeBase = async () => {
    if (!confirm('警告：此操作将清空整个知识库，包括所有文档和标签！此操作不可恢复！')) {
      return;
    }
    if (!confirm('确定要继续吗？')) {
      return;
    }

    try {
      setClearing(true);
      const result = await clearKnowledgeBase();
      if (result.success) {
        alert('知识库已清空');
      } else {
        alert('清空失败: ' + result.message);
      }
    } catch (error) {
      console.error('清空失败:', error);
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">设置</h1>
        <p className="text-muted-foreground">配置知识库管理设置</p>
      </div>

      <div className="space-y-6 max-w-2xl">
        {/* API Key 设置 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Key className="h-5 w-5" />
              API Key 设置
            </CardTitle>
            <CardDescription>
              配置 API Key 用于访问后端服务
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="apiKey">API Key</Label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    id="apiKey"
                    type={showKey ? 'text' : 'password'}
                    value={apiKey}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    placeholder="输入 API Key"
                  />
                </div>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setShowKey(!showKey)}
                >
                  {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
            </div>
            <div className="flex gap-2">
              <Button onClick={handleSaveApiKey} disabled={!apiKey.trim()}>
                {saved ? '已保存' : '保存'}
              </Button>
              <Button variant="outline" onClick={handleClearApiKey}>
                清除
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 知识库管理 */}
        <Card>
          <CardHeader>
            <CardTitle>知识库管理</CardTitle>
            <CardDescription>
              管理知识库中的数据
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div>
                <h3 className="font-medium flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-destructive" />
                  清空知识库
                </h3>
                <p className="text-sm text-muted-foreground">
                  删除所有文档和标签，此操作不可恢复
                </p>
              </div>
              <Button
                variant="destructive"
                onClick={handleClearKnowledgeBase}
                disabled={clearing}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                {clearing ? '清空中...' : '清空'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 关于 */}
        <Card>
          <CardHeader>
            <CardTitle>关于</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>本地向量知识库 v1.0</p>
              <p>基于 Next.js + FastAPI 构建</p>
              <p>支持向量检索 + BM25 全文搜索</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
