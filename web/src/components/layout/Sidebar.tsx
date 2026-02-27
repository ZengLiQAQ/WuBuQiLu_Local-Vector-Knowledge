'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FileText, Search, Tag, Download, Settings, TrainTrack } from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/documents', label: '文档管理', icon: FileText },
  { href: '/search', label: '搜索', icon: Search },
  { href: '/tags', label: '标签', icon: Tag },
  { href: '/export', label: '导出导入', icon: Download },
  { href: '/train', label: '模型训练', icon: TrainTrack },
  { href: '/settings', label: '设置', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r bg-card h-screen sticky top-0">
      <div className="p-4">
        <h1 className="text-xl font-bold px-2">知识库</h1>
      </div>
      <nav className="px-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || pathname.startsWith(item.href + '/');

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
            >
              <Icon className="w-5 h-5" />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
