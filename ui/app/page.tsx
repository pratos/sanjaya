"use client";

import { QueryInput } from "@/components/hud/query-input";
import { RunStatusHeader } from "@/components/hud/run-status";

import { WindowsPanel } from "@/components/hud/windows-panel";
import { CodeExecutionPanel } from "@/components/hud/code-execution-panel";
import { ClipPanel } from "@/components/hud/clip-panel";
import { SubLLMPanel } from "@/components/hud/sub-llm-panel";
import { VisionPanel } from "@/components/hud/vision-panel";
import { TraceTimeline } from "@/components/hud/trace-timeline";
import { FinalAnswerPanel } from "@/components/hud/final-answer-panel";
import { RunHistory } from "@/components/hud/run-history";
import { useRun } from "@/lib/use-run";

export default function Dashboard() {
  const {
    state,
    startRun,

    codeExecutions,
    subLLMCalls,
    clips,
    visionQueries,
  } = useRun();

  const isRunning = state.status === "running";
  const isFinished = state.status === "complete" || state.status === "error";

  // Extract question from run_start event
  const question =
    state.events.find((e) => e.kind === "run_start")?.payload
      ?.question_preview as string | undefined ?? null;

  return (
    <div className="flex min-h-screen flex-col">
      {/* Status Bar */}
      <RunStatusHeader
        status={state.status}
        runId={state.runId}
        iteration={state.currentIteration}
        maxIterations={state.maxIterations}
        startTime={state.startTime}
      />

      {/* Final Answer Banner (shown when complete/error) */}
      {isFinished && (
        <FinalAnswerPanel
          status={state.status}
          finalAnswer={state.finalAnswer}
          finalStatus={state.finalStatus}
          question={question}
          events={state.events}
        />
      )}

      {/* Main Grid
        ┌──────────────────────────────────────────────────────────┐
        │  STATUS BAR (above)                                      │
        ├─────────────────────────┬────────────────────────────────┤
        │  QUERY INPUT            │  PROGRESS                      │
        ├─────────────────────────┼────────────────────────────────┤
        │  CANDIDATE WINDOWS      │  CODE EXECUTION LOG            │
        ├─────────────────────────┼────────────────────────────────┤
        │  CLIPS & FRAMES         │  SUB-LLM QUERIES               │
        ├─────────────────────────┼────────────────────────────────┤
        │  VISION QUERIES         │  TRACE TIMELINE                │
        ├─────────────────────────┴────────────────────────────────┤
        │  RUN HISTORY                                             │
        └──────────────────────────────────────────────────────────┘
      */}
      <div className="flex-1 grid grid-cols-2 grid-rows-[auto_minmax(200px,350px)_minmax(200px,350px)_minmax(200px,350px)_minmax(150px,1fr)] gap-px bg-hud-border">
        {/* Row 1: Query Input (full width) */}
        <div className="col-span-2">
          <QueryInput onSubmit={startRun} disabled={isRunning} />
        </div>

        {/* Row 2: Candidate Windows + Code Execution */}
        <WindowsPanel events={state.events} />
        <CodeExecutionPanel
          executions={codeExecutions}
          isRunning={isRunning}
        />

        {/* Row 3: Clips & Frames + Sub-LLM Queries */}
        <ClipPanel clips={clips} />
        <SubLLMPanel calls={subLLMCalls} />

        {/* Row 4: Vision Queries + Trace Timeline */}
        <VisionPanel queries={visionQueries} />
        <TraceTimeline events={state.events} startTime={state.startTime} />

        {/* Row 5: Run History (full width) */}
        <div className="col-span-2">
          <RunHistory runStatus={state.status} />
        </div>
      </div>
    </div>
  );
}
