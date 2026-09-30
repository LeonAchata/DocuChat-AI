export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type DocStatus = "processing" | "ready" | "failed";

export type DocumentInfo = {
  id: string;
  title: string;
  filename: string;
  pages: number | null;
  chunk_count: number;
  status: DocStatus;
  error: string | null;
  created_at: string;
};

export type Source = {
  id: number;
  document_id: string;
  title: string;
  location: string;
  text: string;
  relevance: number | null;
};

export type Citation = { block: number; passage: number; text: string };

export type AnswerStatus = "answered" | "no_evidence" | "chitchat" | "failed";

export type ChatEvent =
  | { event: "thread"; thread_id: string }
  | { event: "plan"; question: string; queries: string[]; hypothetical: string | null }
  | {
      event: "retrieval";
      round: number;
      queries: string[];
      candidates: number;
      confidence: number | null;
      timings_ms: Record<string, number>;
      passages: Source[];
    }
  | { event: "reformulate"; queries: string[] }
  | { event: "delta"; block: number; text: string }
  | ({ event: "citation" } & Citation)
  | {
      event: "answer";
      status: AnswerStatus;
      content: string;
      blocks: string[];
      sources: Source[];
      citations: Citation[];
    }
  | { event: "error"; message: string }
  | { event: "done" };

async function check(res: Response): Promise<Response> {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {}
    throw new Error(detail);
  }
  return res;
}

export async function listDocuments(signal?: AbortSignal): Promise<DocumentInfo[]> {
  return (await check(await fetch(`${API_URL}/api/documents`, { signal }))).json();
}

export async function uploadDocuments(files: File[]): Promise<DocumentInfo[]> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${API_URL}/api/documents`, { method: "POST", body: form });
  return (await check(res)).json();
}

export async function deleteDocument(id: string): Promise<void> {
  await check(await fetch(`${API_URL}/api/documents/${id}`, { method: "DELETE" }));
}

/** POST a chat message and yield Server-Sent Events as they arrive. */
export async function* streamChat(
  message: string,
  threadId: string | null,
  documentIds: string[] | null,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await check(
    await fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId, document_ids: documentIds }),
      signal,
    }),
  );
  if (!res.body) throw new Error("Empty response");
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseEvent(block);
      if (parsed) yield parsed;
    }
  }
}

function parseEvent(block: string): ChatEvent | null {
  let name = "";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!name) return null;
  try {
    return { ...JSON.parse(data.join("\n") || "{}"), event: name } as ChatEvent;
  } catch {
    return null;
  }
}
