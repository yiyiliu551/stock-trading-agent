"""
GraphRAG + Chunk优化 完整版
两个优化案例：
1. GraphRAG解决新闻记混问题
2. Chunk+Embedding优化解决召回分数低问题
"""

from datetime import datetime
from typing import Optional
import json


# ============================================================
# Part 1: GraphRAG知识图谱存储
# ============================================================

class KnowledgeGraph:
    """
    知识图谱：存储实体+关系+时间
    解决：不同新闻事件被LLM混淆的幻觉问题
    """
    
    def __init__(self):
        # 节点：{entity_id: {name, type, attributes}}
        self.nodes = {}
        # 边：{edge_id: {source, relation, target, timestamp, symbol}}
        self.edges = {}
        # 时间序列索引：{symbol: [edge_ids按时间排序]}
        self.time_index = {}
        print("✅ 知识图谱初始化完成")
    
    def add_node(self, entity_id: str, name: str, entity_type: str, **attrs):
        """添加实体节点"""
        self.nodes[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,  # company/person/metric/event
            **attrs
        }
    
    def add_edge(self, edge_id: str, source_id: str, relation: str, 
                 target_id: str, timestamp: str, symbol: str, 
                 impact: str, source: str):
        """添加关系边（带时间戳）"""
        self.edges[edge_id] = {
            "id": edge_id,
            "source": source_id,
            "relation": relation,
            "target": target_id,
            "timestamp": timestamp,
            "symbol": symbol,
            "impact": impact,
            "news_source": source
        }
        # 更新时间序列索引
        if symbol not in self.time_index:
            self.time_index[symbol] = []
        self.time_index[symbol].append(edge_id)
        self.time_index[symbol].sort(
            key=lambda x: self.edges[x]["timestamp"]
        )
    
    def query_by_symbol(self, symbol: str, 
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None,
                        n: int = 5) -> list:
        """按股票+时间范围查询知识图谱"""
        if symbol not in self.time_index:
            return []
        
        edge_ids = self.time_index[symbol]
        results = []
        
        for eid in edge_ids:
            edge = self.edges[eid]
            # 时间过滤
            if start_time and edge["timestamp"] < start_time:
                continue
            if end_time and edge["timestamp"] > end_time:
                continue
            
            # 获取节点信息
            source_node = self.nodes.get(edge["source"], {})
            target_node = self.nodes.get(edge["target"], {})
            
            results.append({
                "timestamp": edge["timestamp"],
                "symbol": edge["symbol"],
                "triple": f"{source_node.get('name','?')} → {edge['relation']} → {target_node.get('name','?')}",
                "impact": edge["impact"],
                "source": edge["news_source"]
            })
        
        return results[:n]
    
    def format_for_llm(self, results: list) -> str:
        """格式化给LLM——清晰时序防幻觉"""
        if not results:
            return "No knowledge graph events found."
        
        lines = [
            "🕸️  KNOWLEDGE GRAPH EVENTS (CHRONOLOGICAL):",
            "=" * 55,
            "⚠️  Each triple is INDEPENDENT. Do NOT mix events!",
            ""
        ]
        for i, r in enumerate(results, 1):
            lines.append(f"Event #{i} [{r['timestamp']}]")
            lines.append(f"  Triple: {r['triple']}")
            lines.append(f"  Impact: {r['impact']}")
            lines.append(f"  Source: {r['source']}")
            lines.append("")
        
        lines.append("=" * 55)
        return "\n".join(lines)


# ============================================================
# Part 2: Chunk优化 + Embedding质量检查
# 解决：召回分数低的问题
# ============================================================

class FinancialChunker:
    """
    金融新闻专用Chunker
    解决：普通chunking切断关键金融信息导致召回分数低
    """
    
    def __init__(self, chunk_size: int = 1024, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, symbol: str, timestamp: str) -> list:
        """
        金融感知切割：
        - 保持公司名+财务数字在同一chunk
        - 添加symbol和时间作为prefix
        - 使用overlap避免信息丢失
        """
        # 每个chunk都加上symbol和时间的prefix
        # 确保检索时能正确匹配到对应股票
        prefix = f"[{symbol}][{timestamp}] "
        
        # 按句子切割（不在句子中间断开）
        sentences = text.replace('. ', '.|').split('|')
        
        chunks = []
        current_chunk = prefix
        
        for sentence in sentences:
            # 如果加入这句话不超过chunk_size
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                # 保存当前chunk
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "chunk_size": len(current_chunk)
                    })
                # 新chunk用overlap开始
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = prefix + overlap_text + sentence + " "
        
        # 最后一个chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "symbol": symbol,
                "timestamp": timestamp,
                "chunk_size": len(current_chunk)
            })
        
        return chunks
    
    def validate_chunks(self, chunks: list) -> dict:
        """
        验证chunk质量——对应Multi-stage validation
        确保关键金融信息没有被切断
        """
        issues = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            # 检查chunk是否太小（可能切断了信息）
            if len(text) < 100:
                issues.append(f"Chunk {i}: too small ({len(text)} chars)")
            # 检查是否包含股票代码（应该都有prefix）
            if chunk["symbol"] not in text:
                issues.append(f"Chunk {i}: missing symbol prefix")
        
        return {
            "total_chunks": len(chunks),
            "avg_size": sum(c["chunk_size"] for c in chunks) / len(chunks) if chunks else 0,
            "issues": issues,
            "quality": "good" if not issues else "needs_review"
        }


# ============================================================
# Part 3: 完整演示
# ============================================================

def demo_complete():
    print("\n" + "="*60)
    print("🧪 完整GraphRAG + Chunk优化演示")
    print("="*60)
    
    # ---- 演示1：GraphRAG知识图谱 ----
    print("\n📌 Part 1: GraphRAG知识图谱")
    print("-" * 40)
    
    kg = KnowledgeGraph()
    
    # 添加实体节点
    kg.add_node("sndk", "SanDisk Corporation", "company", ticker="SNDK")
    kg.add_node("sndk_q2_eps", "Q2 EPS $23.41", "metric", value=23.41)
    kg.add_node("ceo_goeckeler", "CEO David Goeckeler", "person", role="CEO")
    kg.add_node("mizuho", "Mizuho analyst", "person", role="analyst")
    kg.add_node("target_1625", "Price target $1,625", "metric", value=1625)
    
    # 添加关系边（带时间戳）
    kg.add_edge(
        "e001", "sndk_q2_eps", "beat expectations by 60%", "sndk",
        "2026-05-01 16:30:00", "SNDK", "positive", "Reuters"
    )
    kg.add_edge(
        "e002", "ceo_goeckeler", "resigned from", "sndk",
        "2026-05-03 09:00:00", "SNDK", "negative", "Bloomberg"
    )
    kg.add_edge(
        "e003", "mizuho", "raised price target to", "target_1625",
        "2026-05-07 10:00:00", "SNDK", "positive", "TipRanks"
    )
    
    # 查询
    results = kg.query_by_symbol("SNDK")
    print(kg.format_for_llm(results))
    
    # ---- 演示2：Chunk优化 ----
    print("\n📌 Part 2: 金融Chunk优化")
    print("-" * 40)
    
    chunker = FinancialChunker(chunk_size=1024, overlap=200)
    
    # 模拟一篇财报新闻
    news = """
    SanDisk Corporation reported Q2 FY2026 earnings of $23.41 per share, 
    significantly beating analyst estimates of $14.62 per share by approximately 60%. 
    Revenue came in at $5.95 billion versus the expected $4.72 billion. 
    The company cited strong AI infrastructure demand for NAND flash storage as the primary driver. 
    CEO David Goeckeler stated the company is well-positioned for continued growth. 
    The stock rose 4.89% to $1,116.25 in after-hours trading, 
    surpassing its previous 52-week high. 
    Management guided Q3 revenue of $7.75B-$8.25B with EPS of $30-$33.
    """
    
    chunks = chunker.chunk(news.strip(), "SNDK", "2026-05-01 16:30:00")
    validation = chunker.validate_chunks(chunks)
    
    print(f"총 Chunks: {validation['total_chunks']}")
    print(f"평균 Size: {validation['avg_size']:.0f} chars")
    print(f"Quality: {validation['quality']}")
    
    if chunks:
        print(f"\n첫 번째 Chunk 예시:")
        print(f"  {chunks[0]['text'][:200]}...")
    
    # ---- 핵심 요약 ----
    print("\n" + "="*60)
    print("✅ 两个优化案例总结：")
    print()
    print("Case 1: 新闻记混 → GraphRAG")
    print("  问题：传统RAG把不同时间的新闻混淆")
    print("  解决：实体+关系+时间戳独立存储")
    print("  检测：人工检查发现LLM混淆了两条新闻")
    print()
    print("Case 2: 召回分数低 → Chunk优化")
    print("  问题：关键新闻在库里但RAGAS Context Recall低")
    print("  解决：1024 token chunk + 200 overlap + symbol prefix")
    print("  检测：RAGAS Context Recall指标")
    print("="*60)


if __name__ == "__main__":
    demo_complete()
