import { describe, expect, it } from "vitest";
import { POST } from "./route";

describe("copilotkit runtime route", () => {
  it("exports a POST handler", () => {
    expect(typeof POST).toBe("function");
  });
});
