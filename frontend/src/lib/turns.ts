import type { AnswerStatus, ChatEvent, Citation, Source } from "./api";

export type Retrieval = Extract<ChatEvent, { event: "retrieval" }>;

export type Turn = {
  id: string;
  question: string;
  standalone: string | null;
  retrievals: Retrieval[];
  blocks: string[];
  citations: Citation[];
  sources: Source[];
  status: AnswerStatus | "running";
  error: string | null;
};

export function newTurn(question: string): Turn {
  return {
    id: crypto.randomUUID(),
    question,
    standalone: null,
    retrievals: [],
    blocks: [],
    citations: [],
    sources: [],
    status: "running",
    error: null,
  };
}

export function applyEvent(turn: Turn, e: ChatEvent): Turn {
  switch (e.event) {
    case "plan":
      return { ...turn, standalone: e.question };
    case "retrieval":
      return { ...turn, retrievals: [...turn.retrievals, e], sources: e.passages };
    case "delta": {
      const blocks = [...turn.blocks];
      while (blocks.length <= e.block) blocks.push("");
      blocks[e.block] += e.text;
      return { ...turn, blocks };
    }
    case "citation":
      return { ...turn, citations: [...turn.citations, { block: e.block, passage: e.passage, text: e.text }] };
    case "answer":
      return {
        ...turn,
        status: e.status,
        blocks: e.blocks.length ? e.blocks : [e.content],
        citations: e.citations,
        sources: e.sources,
      };
    case "error":
      return { ...turn, status: "failed", error: e.message };
    default:
      return turn;
  }
}

/** Sources in the order they are first cited; uncited sources are dropped once any are cited. */
export function citedSources(turn: Turn): { source: Source; n: number }[] {
  const order: number[] = [];
  for (const c of turn.citations) if (!order.includes(c.passage)) order.push(c.passage);
  const byId = new Map(turn.sources.map((s) => [s.id, s]));
  const ids = order.length ? order : turn.sources.map((s) => s.id);
  return ids.flatMap((id, i) => {
    const source = byId.get(id);
    return source ? [{ source, n: i + 1 }] : [];
  });
}
