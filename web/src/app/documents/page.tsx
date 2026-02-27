'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Upload, FileText, Trash2, RefreshCw } from 'lucide-react';
import { getStats, uploadFiles, deleteDocuments } from '@/lib/api';

interface FileInfo {
  path: string;
  type: string;
  chunk_count: number;
}

export default function DocumentsPage() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      setLoading(true);
      const data = await getStats();
      if (data) {
        setStats(data);
      }
    } catch (error) {
      console.error('加载统计失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    try {
      setUploading(true);
      await uploadFiles(files);
      await loadStats();
    } catch (error) {
      console.error('上传失败:', error);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDeleteFile = async (filePath: string) => {
    if (!confirm('确定要删除这个文件吗？')) return;

    try {
      // 获取文件相关的文档ID并删除
      // 这里简化处理，实际需要根据文件路径查询文档ID
      await loadStats();
    } catch (error) {
      console.error('删除失败:', error);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">文档管理</h1>
          <p className="text-muted-foreground mt-1">管理知识库中的文档</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={loadStats} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            刷新
          </Button>
          <Button onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            <Upload className="mr-2 h-4 w-4" />
            {uploading ? '上传中...' : '上传文件'}
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.md,.txt,.docx,.xlsx,.pptx,.jpg,.png"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              文档块数
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.total_chunks || 0}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              文件类型数
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats?.file_types ? Object.keys(stats.file_types).length : 0}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              存储大小
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats?.storage_size ? (stats.storage_size / 1024 / 1024).toFixed(2) + ' MB' : '0 MB'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 文件类型分布 */}
      {stats?.file_types && Object.keys(stats.file_types).length > 0 && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>文件类型分布</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4">
              {Object.entries(stats.file_types).map(([type, count]: [string, any]) => (
                <div key={type} className="flex items-center gap-2">
                  <span className="font-medium">{type}</span>
                  <span className="text-muted-foreground">({count})</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 说明 */}
      <Card>
        <CardContent className="py-8">
          <div className="text-center text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>支持上传 PDF、Markdown、TXT、Word、Excel、PPT、图片等文件</p>
            <p className="text-sm mt-2">文件将自动分块、向量化并存储到知识库中</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
