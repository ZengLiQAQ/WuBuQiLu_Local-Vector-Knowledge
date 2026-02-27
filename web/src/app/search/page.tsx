'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Search, FileText, Filter, Zap } from 'lucide-react';
import { search } from '@/lib/api';
import type { SearchResult } from '@/types';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [hybrid, setHybrid] = useState(true);
  const [useBm25, setUseBm25] = useState(true);
  const [topK, setTopK] = useState(5);

  const handleSearch = async () => {
    if (!query.trim()) return;

    try {
      setLoading(true);
      setSearched(true);
      const response = await search({
        query: query,
        top_k: topK,
        hybrid: hybrid,
        use_bm25: useBm25,
      });

      if (response.success) {
        setResults(response.results || []);
      }
    } catch (error) {
      console.error('搜索失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="container mx-auto py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">搜索</h1>
        <p className="text-muted-foreground">在知识库中搜索相关内容</p>
      </div>

      {/* 搜索框 */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                className="pl-10"
                placeholder="输入搜索内容..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
              />
            </div>
            <Button onClick={handleSearch} disabled={loading || !query.trim()}>
              {loading ? '搜索中...' : '搜索'}
            </Button>
          </div>

          {/* 搜索选项 */}
          <div className="flex flex-wrap gap-4 mt-4 pt-4 border-t">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="hybrid"
                checked={hybrid}
                onChange={(e) => setHybrid(e.target.checked)}
                className="rounded"
              />
              <label htmlFor="hybrid" className="text-sm flex items-center gap-1">
                <Zap className="h-4 w-4" />
                混合检索
              </label>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="useBm25"
                checked={useBm25}
                onChange={(e) => setUseBm25(e.target.checked)}
                disabled={!hybrid}
                className="rounded"
              />
              <label htmlFor="useBm25" className="text-sm flex items-center gap-1">
                <Filter className="h-4 w-4" />
                BM25 检索
              </label>
            </div>
            <div className="flex items-center gap-2">
              <label htmlFor="topK" className="text-sm">返回条数:</label>
              <select
                id="topK"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="rounded border border-input bg-background px-2 py-1 text-sm"
              >
                <option value={3}>3</option>
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 搜索结果 */}
      {searched && (
        <div>
          <h2 className="text-xl font-semibold mb-4">
            搜索结果 {results.length > 0 && `(${results.length} 条)`}
          </h2>

          {results.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Search className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">未找到相关结果</p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {results.map((result, index) => (
                <Card key={result.id || index}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <FileText className="h-5 w-5 text-muted-foreground" />
                        <CardTitle className="text-base">
                          {String(result.metadata?.file_path || result.id)}
                        </CardTitle>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        相似度: {((result as any).total_score || (result as any).score || 0).toFixed(4)}
                      </div>
                    </div>
                    <CardDescription>
                      类型: {String(result.file_type || result.metadata?.file_type || 'unknown')}
                      {(result as any).vector_score !== undefined && (
                        <span className="ml-2">
                          | 向量: {((result as any).vector_score).toFixed(4)}
                        </span>
                      )}
                      {(result as any).bm25_score !== undefined && ((result as any).bm25_score > 0) && (
                        <span className="ml-2">
                          | BM25: {((result as any).bm25_score).toFixed(4)}
                        </span>
                      )}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-muted p-3 rounded-lg text-sm">
                      {result.text?.length > 500
                        ? result.text.substring(0, 500) + '...'
                        : result.text}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
