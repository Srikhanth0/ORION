# ORION Test Run — Break-Point Report
Date: 2026-04-14
Commit: v1.0.0

## Environment
- Python: 3.13.12
- GROQ reachable: YES
- OpenRouter reachable: NO (model `mistralai/mistral-7b-instruct` returned 404 — OpenRouter modified endpoint or model availability changed)
- Vision API reachable: YES (latency: ~200ms ping via ngrok)
- Qdrant running: NO (requires podman compose up locally, test integration spins qdrant temporarily)
- MCP tools loaded: 0 (No active tools loaded initially due to background setup / uvx missing on bare metal Windows powershell vs WSL)

## Task Results
| Task | Status | Break Points Found |
|------|--------|--------------------|
| Task 1 (Shell) | failed (timeout) | Baremetal execution without specific permissions mapped correctly caused HITL lock or timeout |
| Task 2 (Reasoning + Memory) | failed (timeout) | Missing OpenRouter fallback context and Qdrant db not exposed outside container |
| Task 3 (GUI Vision) | failed (timeout) | Missing Ngrok context mapping locally in host |

## Critical Breaks (P0 — blocks core functionality)
- [x] **Agent MCP Bindings via npx / uvx**: On Windows without WSL, the raw `uvx` and `npx` commands map differently causing the Tool Registry to return `0` tools loaded inside the FastAPI environment.
- [x] **OpenRouter Model 404**: The default mistral model in `router.yaml`/`.env` is offline. Needs updating to a new active free route.

## High Severity (P1 — degrades reliability)
- [x] **Memory Vector Database**: The integration tests spawn `qdrant` as a service. However, running the API server locally without `.env` mapping to a live `qdrant` instance causes long-term tasks (Task 2) to loop or crash the verifier.
- [x] **Windows Encoding**: The Python print outputs defaulted to `cp1252` causing unicode crashes (`━━━`) in the test harness wrapper. 

## Medium (P2 — edge cases / observability gaps)
- [x] **Subtask ID Missing**: Subtask polling relies entirely on the DAG, but timeout polling didn't capture intermediate granular steps effectively without listening to the `/v1/tasks/{id}/stream` endpoint via SSE.

## LLM Router Observed Path
Task 1: vLLM -> Groq -> OpenRouter (Fail)
Task 2: vLLM -> Groq
Task 3: vLLM -> Groq

## Recommendations
1. Patch the `uvx` and `npx` dependency resolutions to support vanilla Windows `.cmd` paths or enforce the stack runs entirely inside WSL2 / Windows Subsystem.
2. Update the OpenRouter model fallbacks inside `.env` to `google/gemma-2-9b-it:free` or similar active endpoints. 
3. Recommend users explicitly run `podman-compose up` specifically for the `qdrant` database dependency *before* running `make dev` locally, otherwise Task 2 loops.
