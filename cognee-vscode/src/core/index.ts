/**
 * Editor-agnostic core for the Cognee memory integration.
 *
 * Nothing in this folder imports `vscode`, so it can be unit-tested in plain
 * Node against a mocked backend and reused by other editor front-ends.
 */
export * from "./types";
export * from "./errors";
export * from "./config";
export * from "./scope";
export * from "./git";
export * from "./evidence";
export * from "./client";
export * from "./httpClient";
