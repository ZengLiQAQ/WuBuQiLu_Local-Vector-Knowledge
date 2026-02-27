'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Download, Upload, FileJson, FileText, File } from 'lucide-react';
import { exportData, importData } from '@/lib/api';

export default function ExportPage() {
  const [exporting, setExporting] = useState(false);
  const [importing, setImporting] = useState(false);

  const handleExport = async (format: 'json' | 'csv' | 'md') => {
    try {
      setExporting(true);
      const blob = await exportData(format);

      // 下载文件
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `kb_export_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('导出失败:', error);
    } finally {
      setExporting(false);
    }
  };

  const handleImport = async (format: 'json' | 'csv') => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = format === 'json' ? '.json' : '.csv';

    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      try {
        setImporting(true);
        const result = await importData(file, format);
        if (result.success) {
          alert(`导入成功！共导入 ${result.imported_count || 0} 条记录`);
        } else {
          alert('导入失败: ' + result.message);
        }
      } catch (error) {
        console.error('导入失败:', error);
      } finally {
        setImporting(false);
      }
    };

    input.click();
  };

  return (
    <div className="container mx-auto py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">导出导入</h1>
        <p className="text-muted-foreground">导出和导入知识库数据</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* 导出 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Download className="h-5 w-5" />
              导出数据
            </CardTitle>
            <CardDescription>
              将知识库中的文档和标签导出为不同格式
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button
              className="w-full"
              variant="outline"
              onClick={() => handleExport('json')}
              disabled={exporting}
            >
              <FileJson className="mr-2 h-4 w-4" />
              导出为 JSON（完整数据）
            </Button>
            <Button
              className="w-full"
              variant="outline"
              onClick={() => handleExport('csv')}
              disabled={exporting}
            >
              <FileText className="mr-2 h-4 w-4" />
              导出为 CSV（文档列表）
            </Button>
            <Button
              className="w-full"
              variant="outline"
              onClick={() => handleExport('md')}
              disabled={exporting}
            >
              <File className="mr-2 h-4 w-4" />
              导出为 Markdown
            </Button>
          </CardContent>
        </Card>

        {/* 导入 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              导入数据
            </CardTitle>
            <CardDescription>
              从 JSON 或 CSV 文件导入文档到知识库
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button
              className="w-full"
              variant="outline"
              onClick={() => handleImport('json')}
              disabled={importing}
            >
              <FileJson className="mr-2 h-4 w-4" />
              导入 JSON 文件
            </Button>
            <Button
              className="w-full"
              variant="outline"
              onClick={() => handleImport('csv')}
              disabled={importing}
            >
              <FileText className="mr-2 h-4 w-4" />
              导入 CSV 文件
            </Button>
            <div className="text-sm text-muted-foreground mt-4">
              <p className="font-medium mb-2">导入说明：</p>
              <ul className="list-disc list-inside space-y-1">
                <li>JSON 格式：完整恢复（文档+标签）</li>
                <li>CSV 格式：批量导入文档（自动向量化）</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
