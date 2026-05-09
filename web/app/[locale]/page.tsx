import Link from "next/link";
import { notFound } from "next/navigation";
import { CURRICULUM, chapterTitle, type Locale } from "@/lib/curriculum";

export default async function Landing({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  if (locale !== "zh" && locale !== "en") notFound();
  const l = locale as Locale;

  const intro = l === "zh" ? INTRO_ZH : INTRO_EN;
  const ctaLabel = l === "zh" ? "从 s01 开始 →" : "Start at s01 →";

  return (
    <article className="prose-doc">
      <h1>learn-lightrag</h1>
      <p className="text-[var(--fg-muted)]">
        {l === "zh"
          ? "用 Go 从零渐进构建一个 LightRAG（GraphRAG 的开源实现），每节末尾对照上游 Python 源码。"
          : "Build LightRAG (an open-source GraphRAG implementation) from scratch in Go, session by session — each chapter ends with the upstream Python source."}
      </p>

      {intro.map((p, i) => (
        <p key={i}>{p}</p>
      ))}

      <p>
        <Link
          href={`/${l}/s/s01-minimum-loop`}
          className="inline-block mt-2 px-4 py-2 rounded border border-[var(--accent-soft)] hover:border-[var(--accent)]"
        >
          {ctaLabel}
        </Link>
      </p>

      <h2>{l === "zh" ? "课程" : "Curriculum"}</h2>
      <ul>
        {CURRICULUM.map((c) => (
          <li key={c.slug}>
            <span className="font-mono text-[var(--fg-muted)] mr-2">
              {c.num}
            </span>
            {c.available ? (
              <Link href={`/${l}/s/${c.slug}`}>{chapterTitle(c, l)}</Link>
            ) : (
              <span className="text-[var(--fg-muted)]">
                {chapterTitle(c, l)}{" "}
                <span className="text-xs">
                  ({l === "zh" ? "未发布" : "not yet"})
                </span>
              </span>
            )}
          </li>
        ))}
      </ul>
    </article>
  );
}

const INTRO_ZH = [
  "这个仓库的目标不是教你「用」 LightRAG，是教你「它怎么从零长出来」。",
  "每一节加一个机制——文档状态机、token 切分、KV 存储、向量检索、图存储、实体抽取、归并总结、双层检索——用 Go 写一份精简实现。看完十一节，你会觉得 LightRAG 不再是一团黑魔法。",
  "Go 实现是教学骨架，LightRAG 上游是 Python 实现 (HKUDS, EMNLP'25)。每节末尾的「上游源码阅读」把这两边对照起来，你能从 mini 版顺着指针读到生产代码。",
];

const INTRO_EN = [
  "The goal of this repo is not to teach you to *use* LightRAG — it is to teach you how it grows from scratch.",
  "Each chapter adds one mechanism — doc status, token chunking, KV store, vector retrieval, graph store, entity extraction, summarization, dual-level retrieval — implemented as a small Go file. After eleven chapters, LightRAG stops being black magic.",
  "Go is the teaching skeleton; the upstream Python (HKUDS, EMNLP'25) is the production implementation. The 'Upstream Source Reading' section at the end of every chapter bridges them — you can follow the pointers from the mini version straight into the real code.",
];
