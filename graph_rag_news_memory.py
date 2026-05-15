"""
GraphRAG News Memory System for AI Trading Agent
解决幻觉问题：新闻按实体+关系+时间序列存储
"""

from typing import Optional


class NewsEvent:
    def __init__(self, event_id, timestamp, symbol, event_type,
                 subject, predicate, object_, impact, source, raw_content):
        self.event_id = event_id
        self.timestamp = timestamp
        self.symbol = symbol
        self.event_type = event_type
        self.subject = subject
        self.predicate = predicate
        self.object_ = object_
        self.impact = impact
        self.source = source
        self.raw_content = raw_content


class GraphRAGNewsMemory:
    def __init__(self):
        self.memory = {}
        print("✅ GraphRAG Memory初始化完成")
    
    def store_event(self, event: NewsEvent):
        if event.symbol not in self.memory:
            self.memory[event.symbol] = []
        self.memory[event.symbol].append(event)
        print(f"📌 [{event.timestamp}] {event.symbol} | {event.subject} {event.predicate}")
    
    def store_batch(self, events):
        for event in events:
            self.store_event(event)
        print(f"\n✅ 批量存储完成: {len(events)}条\n")
    
    def query(self, symbol, start_time=None, end_time=None, event_type=None, n_results=5):
        if symbol not in self.memory:
            return []
        events = self.memory[symbol].copy()
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        events.sort(key=lambda x: x.timestamp)
        return events[:n_results]
    
    def format_for_llm(self, events):
        if not events:
            return "No events found."
        lines = ["📰 NEWS EVENTS (CHRONOLOGICAL - DO NOT MIX EVENTS):", "="*60]
        for i, e in enumerate(events, 1):
            lines.append(f"\nEvent #{i}:")
            lines.append(f"  ⏰ Time:   {e.timestamp}")
            lines.append(f"  📊 Stock:  {e.symbol}")
            lines.append(f"  📌 Type:   {e.event_type}")
            lines.append(f"  🔗 Graph:  {e.subject} → {e.predicate} → {e.object_}")
            lines.append(f"  💹 Impact: {e.impact}")
        lines.append("\n" + "="*60)
        lines.append("⚠️  Each event is INDEPENDENT. Do NOT combine events from different times!")
        return "\n".join(lines)


def demo():
    print("\n" + "="*60)
    print("🧪 GraphRAG防幻觉演示")
    print("="*60)
    print("\n❌ 问题：")
    print("   新闻A：SNDK财报超预期（正面）")
    print("   新闻B：SNDK CEO离职（负面）")
    print("   传统RAG可能混淆成：'CEO离职导致财报超预期'\n")
    
    memory = GraphRAGNewsMemory()
    
    events = [
        NewsEvent("SNDK_001", "2026-05-01 16:30:00", "SNDK", "earnings",
                  "SNDK Q2 Earnings", "beat expectations",
                  "EPS $23.41 vs $14.62 (+60%)", "positive", "Reuters",
                  "SanDisk Q2 EPS beat by 60%"),
        NewsEvent("SNDK_002", "2026-05-03 09:00:00", "SNDK", "management",
                  "CEO David Goeckeler", "resigned",
                  "effective immediately", "negative", "Bloomberg",
                  "SanDisk CEO resigned effective immediately"),
        NewsEvent("SNDK_003", "2026-05-07 10:00:00", "SNDK", "analyst",
                  "Mizuho analyst", "raised price target",
                  "$1,220 → $1,625 (+33%)", "positive", "TipRanks",
                  "Mizuho raised SNDK target to $1,625"),
        NewsEvent("SNDK_004", "2026-05-12 09:30:00", "SNDK", "market",
                  "SNDK stock", "dropped",
                  "-8.74% intraday", "negative", "CNBC",
                  "SNDK dropped 8.74% intraday"),
    ]
    
    memory.store_batch(events)
    
    print("📊 所有事件（时间序列）：")
    results = memory.query("SNDK", n_results=10)
    print(memory.format_for_llm(results))
    
    print("\n📊 只看5月1-5日：")
    results2 = memory.query("SNDK", start_time="2026-05-01", end_time="2026-05-05 23:59:59")
    print(memory.format_for_llm(results2))
    
    print("\n✅ GraphRAG效果：")
    print("   每个事件有独立时间戳+实体关系")
    print("   LLM按时间序列理解，不会混淆A和B！")


if __name__ == "__main__":
    demo()
