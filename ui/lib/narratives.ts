export const NARRATIVES: Record<number, { title: string; text: string; improvement: string }> = {
  3: {
    title: "Enhanced Critic",
    text: "The stricter critic now rejects vague corrections like 'the presenter continues' and demands specific descriptions of what changed.",
    improvement: "forced@20 iters \u2192 85/100@11 iters",
  },
  5: {
    title: "Quote Validation",
    text: "The critic now rejects mid-sentence transcript fragments as quotes. The agent was forced to find complete, standalone utterances.",
    improvement: "garbage passed@1 iter \u2192 complete quotes@20 iters",
  },
  7: {
    title: "Vision-First Mode",
    text: "Question classified as vision_primary. Dense frame sampling replaced transcript keyword search as primary evidence. Found on-screen code, schemas, and terminal output.",
    improvement: "empty answer \u2192 90/100@6 iters",
  },
  11: {
    title: "Subtitle Fix + Vision-First",
    text: "Fixed subtitle sidecar lookup (was defaulting to wrong directory). Combined with vision-first mode, extracted 14 specific product features with timestamps and pricing.",
    improvement: "75min frame captions \u2192 4min real analysis",
  },
  12: {
    title: "Stricter Critic + Evidence Grounding",
    text: "Enhanced critic demanded specific player names and match-clock timestamps. Agent correctly identified Alvarez's goal, Holding's yellow card, and Stones substitution.",
    improvement: "80/100@16 iters \u2192 88/100@17 iters",
  },
};
