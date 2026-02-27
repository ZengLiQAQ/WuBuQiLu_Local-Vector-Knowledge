'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Plus, Edit2, Trash2, Tag as TagIcon } from 'lucide-react';
import { getTags, createTag, updateTag, deleteTag } from '@/lib/api';
import type { Tag } from '@/types';

// 预设颜色
const TAG_COLORS = [
  '#FF6B6B', // 红色
  '#4ECDC4', // 青色
  '#45B7D1', // 蓝色
  '#96CEB4', // 绿色
  '#FFEAA7', // 黄色
  '#DDA0DD', // 紫色
  '#FF9F43', // 橙色
  '#54A0FF', // 天蓝
  '#5F27CD', // 深紫
  '#00D2D3', // 湖蓝
  '#FF6B9D', // 粉红
  '#C8D6E5', // 灰色
];

export default function TagsPage() {
  const [tags, setTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingTag, setEditingTag] = useState<Tag | null>(null);
  const [tagName, setTagName] = useState('');
  const [tagColor, setTagColor] = useState(TAG_COLORS[0]);
  const [deletingTag, setDeletingTag] = useState<Tag | null>(null);

  useEffect(() => {
    loadTags();
  }, []);

  const loadTags = async () => {
    try {
      setLoading(true);
      const response = await getTags();
      if (response.success) {
        setTags(response.tags || []);
      }
    } catch (error) {
      console.error('加载标签失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateOrUpdate = async () => {
    if (!tagName.trim()) return;

    try {
      if (editingTag) {
        await updateTag(editingTag.id, { name: tagName, color: tagColor });
      } else {
        await createTag({ name: tagName, color: tagColor });
      }
      setDialogOpen(false);
      resetForm();
      loadTags();
    } catch (error) {
      console.error('保存标签失败:', error);
    }
  };

  const handleDelete = async () => {
    if (!deletingTag) return;

    try {
      await deleteTag(deletingTag.id);
      setDeletingTag(null);
      loadTags();
    } catch (error) {
      console.error('删除标签失败:', error);
    }
  };

  const openCreateDialog = () => {
    setEditingTag(null);
    setTagName('');
    setTagColor(TAG_COLORS[Math.floor(Math.random() * TAG_COLORS.length)]);
    setDialogOpen(true);
  };

  const openEditDialog = (tag: Tag) => {
    setEditingTag(tag);
    setTagName(tag.name);
    setTagColor(tag.color);
    setDialogOpen(true);
  };

  const resetForm = () => {
    setEditingTag(null);
    setTagName('');
    setTagColor(TAG_COLORS[0]);
  };

  return (
    <div className="container mx-auto py-8">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">标签管理</h1>
          <p className="text-muted-foreground mt-1">管理知识库的标签</p>
        </div>
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button onClick={openCreateDialog}>
              <Plus className="mr-2 h-4 w-4" />
              新建标签
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{editingTag ? '编辑标签' : '新建标签'}</DialogTitle>
              <DialogDescription>
                {editingTag ? '修改标签名称和颜色' : '创建一个新的标签'}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">标签名称</Label>
                <Input
                  id="name"
                  value={tagName}
                  onChange={(e) => setTagName(e.target.value)}
                  placeholder="输入标签名称"
                />
              </div>
              <div className="space-y-2">
                <Label>选择颜色</Label>
                <div className="flex flex-wrap gap-2">
                  {TAG_COLORS.map((color) => (
                    <button
                      key={color}
                      type="button"
                      className={`w-8 h-8 rounded-full border-2 ${
                        tagColor === color ? 'border-primary border-4' : 'border-transparent'
                      }`}
                      style={{ backgroundColor: color }}
                      onClick={() => setTagColor(color)}
                    />
                  ))}
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDialogOpen(false)}>
                取消
              </Button>
              <Button onClick={handleCreateOrUpdate} disabled={!tagName.trim()}>
                {editingTag ? '保存' : '创建'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {loading ? (
        <div className="text-center py-8 text-muted-foreground">加载中...</div>
      ) : tags.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <TagIcon className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">暂无标签</p>
            <Button className="mt-4" variant="outline" onClick={openCreateDialog}>
              创建第一个标签
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {tags.map((tag) => (
            <Card key={tag.id}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: tag.color }}
                    />
                    <CardTitle className="text-lg">{tag.name}</CardTitle>
                  </div>
                  <div className="flex gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => openEditDialog(tag)}
                    >
                      <Edit2 className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setDeletingTag(tag)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                <CardDescription>
                  {(tag as any).document_count || 0} 个文档
                </CardDescription>
              </CardHeader>
            </Card>
          ))}
        </div>
      )}

      {/* 删除确认对话框 */}
      <Dialog open={!!deletingTag} onOpenChange={() => setDeletingTag(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除标签</DialogTitle>
            <DialogDescription>
              确定要删除标签 "{deletingTag?.name}" 吗？此操作不会删除关联的文档。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeletingTag(null)}>
              取消
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
