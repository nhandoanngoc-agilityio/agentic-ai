import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangGraphAgent } from "@copilotkit/runtime/langgraph";
import { NextRequest } from "next/server";

const GRAPH_ID = "market_research_team";

const runtime = new CopilotRuntime({
  agents: {
    anthropicAgent: new LangGraphAgent({
      deploymentUrl: process.env.ANTHROPIC_DEPLOYMENT_URL || "http://localhost:2024",
      graphId: GRAPH_ID,
    }),
    openaiAgent: new LangGraphAgent({
      deploymentUrl: process.env.OPENAI_DEPLOYMENT_URL || "http://localhost:2025",
      graphId: GRAPH_ID,
    }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    endpoint: "/api/copilotkit",
    serviceAdapter: new ExperimentalEmptyAdapter(),
    runtime,
  });

  return handleRequest(req);
};
