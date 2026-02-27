import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional
import uuid


class TagDatabase:
    """标签数据库管理"""

    def __init__(self, db_path: str = "./tags.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建标签表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                color TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # 创建文档-标签关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_tags (
                document_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (document_id, tag_id),
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        conn.close()

    def create_tag(self, name: str, color: str) -> dict:
        """创建标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        tag_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        try:
            cursor.execute(
                "INSERT INTO tags (id, name, color, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (tag_id, name, color, now, now)
            )
            conn.commit()
            result = {"id": tag_id, "name": name, "color": color, "created_at": now}
        except sqlite3.IntegrityError:
            raise ValueError(f"标签名称 '{name}' 已存在")
        finally:
            conn.close()

        return result

    def get_all_tags(self) -> List[dict]:
        """获取所有标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT t.id, t.name, t.color, t.created_at,
                   COUNT(dt.document_id) as document_count
            FROM tags t
            LEFT JOIN document_tags dt ON t.id = dt.tag_id
            GROUP BY t.id
            ORDER BY t.created_at DESC
        """)

        tags = []
        for row in cursor.fetchall():
            tags.append({
                "id": row[0],
                "name": row[1],
                "color": row[2],
                "created_at": row[3],
                "document_count": row[4]
            })

        conn.close()
        return tags

    def update_tag(self, tag_id: str, name: str, color: str) -> dict:
        """更新标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute(
            "UPDATE tags SET name = ?, color = ?, updated_at = ? WHERE id = ?",
            (name, color, now, tag_id)
        )

        if cursor.rowcount == 0:
            conn.close()
            raise ValueError(f"标签不存在: {tag_id}")

        conn.commit()
        conn.close()

        return {"id": tag_id, "name": name, "color": color, "updated_at": now}

    def delete_tag(self, tag_id: str) -> bool:
        """删除标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM document_tags WHERE tag_id = ?", (tag_id,))
        cursor.execute("DELETE FROM tags WHERE id = ?", (tag_id,))

        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return deleted

    def add_tag_to_document(self, document_id: str, tag_id: str) -> bool:
        """给文档添加标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        try:
            cursor.execute(
                "INSERT OR IGNORE INTO document_tags (document_id, tag_id, created_at) VALUES (?, ?, ?)",
                (document_id, tag_id, now)
            )
            conn.commit()
            result = True
        except Exception:
            result = False
        finally:
            conn.close()

        return result

    def remove_tag_from_document(self, document_id: str, tag_id: str) -> bool:
        """从文档移除标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM document_tags WHERE document_id = ? AND tag_id = ?",
            (document_id, tag_id)
        )

        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return deleted

    def get_document_tags(self, document_id: str) -> List[dict]:
        """获取文档的所有标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT t.id, t.name, t.color
            FROM tags t
            INNER JOIN document_tags dt ON t.id = dt.tag_id
            WHERE dt.document_id = ?
        """, (document_id,))

        tags = []
        for row in cursor.fetchall():
            tags.append({"id": row[0], "name": row[1], "color": row[2]})

        conn.close()
        return tags

    def get_tag_documents(self, tag_id: str) -> List[str]:
        """获取标签下的所有文档ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT document_id FROM document_tags WHERE tag_id = ?",
            (tag_id,)
        )

        doc_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        return doc_ids

    def get_all_tags_map(self) -> dict:
        """获取所有标签映射 (document_id -> [tags])"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT dt.document_id, t.id, t.name, t.color
            FROM document_tags dt
            INNER JOIN tags t ON dt.tag_id = t.id
        """)

        tags_map = {}
        for row in cursor.fetchall():
            doc_id = row[0]
            tag = {"id": row[1], "name": row[2], "color": row[3]}
            if doc_id not in tags_map:
                tags_map[doc_id] = []
            tags_map[doc_id].append(tag)

        conn.close()
        return tags_map
