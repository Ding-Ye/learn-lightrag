// The locked curriculum from the plan. SessionNav and the landing page both
// read from this single source of truth. Slugs match docs/{zh,en}/<slug>.md.
//
// "available: false" means the chapter exists in the curriculum but its
// docs aren't written yet — the link will render but go to a placeholder.

export type ChapterMeta = {
  slug: string;
  num: string; // "s01", "s02", "s_full"
  title: { zh: string; en: string };
  available: boolean;
};

export const CURRICULUM: ChapterMeta[] = [
  {
    slug: "multi-model",
    num: "M",
    title: {
      zh: "多模型接入指南（OpenAI / Anthropic / Bedrock / Ollama …）",
      en: "Multi-model guide (OpenAI / Anthropic / Bedrock / Ollama …)",
    },
    available: false,
  },
  {
    slug: "s01-minimum-loop",
    num: "s01",
    title: { zh: "最小 RAG 闭环", en: "Minimum RAG loop" },
    available: true,
  },
  {
    slug: "s02-provider",
    num: "s02",
    title: {
      zh: "提供方接口 (OpenAI 聊天补全)",
      en: "Provider interface (OpenAI chat completion)",
    },
    available: false,
  },
  {
    slug: "s03-doc-status",
    num: "s03",
    title: { zh: "文档状态机", en: "Document status state machine" },
    available: false,
  },
  {
    slug: "s04-chunking",
    num: "s04",
    title: {
      zh: "基于 token 的切分",
      en: "Token-based chunking with overlap",
    },
    available: false,
  },
  {
    slug: "s05-kv-store",
    num: "s05",
    title: { zh: "键值存储与过滤", en: "KV store with filter_keys" },
    available: false,
  },
  {
    slug: "s06-embeddings",
    num: "s06",
    title: {
      zh: "嵌入提供方与批处理",
      en: "Embedding provider with batching",
    },
    available: false,
  },
  {
    slug: "s07-vector-store",
    num: "s07",
    title: {
      zh: "余弦相似向量库",
      en: "Cosine-similarity vector store",
    },
    available: false,
  },
  {
    slug: "s08-graph-store",
    num: "s08",
    title: {
      zh: "邻接图存储与子图",
      en: "Adjacency graph store and subgraph BFS",
    },
    available: false,
  },
  {
    slug: "s09-extraction",
    num: "s09",
    title: {
      zh: "实体关系抽取与 gleaning",
      en: "Entity/relation extraction with gleaning",
    },
    available: false,
  },
  {
    slug: "s10-summarization",
    num: "s10",
    title: {
      zh: "描述归并 (map-reduce)",
      en: "Map-reduce description summarization",
    },
    available: false,
  },
  {
    slug: "s11-query-modes",
    num: "s11",
    title: {
      zh: "双层检索四种模式",
      en: "Dual-level retrieval, four modes",
    },
    available: false,
  },
  {
    slug: "s_full-integration",
    num: "s_full",
    title: { zh: "端到端集成", en: "End-to-end integration" },
    available: false,
  },
  {
    slug: "appendix-a-prompt-secrets",
    num: "A",
    title: {
      zh: "附录 A · 提示工程的秘密",
      en: "Appendix A · Prompt-engineering secret sauce",
    },
    available: false,
  },
  {
    slug: "appendix-b-upstream-map",
    num: "B",
    title: {
      zh: "附录 B · 上游源码导读地图",
      en: "Appendix B · Upstream source-reading map",
    },
    available: false,
  },
];

export type Locale = "zh" | "en";

export function chapterTitle(c: ChapterMeta, locale: Locale): string {
  return c.title[locale];
}
