import argparse
import sys
import os
import subprocess
import yaml
import requests
import time
import json
from dotenv import load_dotenv

load_dotenv()


def audit_dependencies():
    print("--- Phase 1.1: Dependency Audit ---")
    print("Python:", sys.version.split(" ")[0])
    subprocess.run([sys.executable, "-m", "uv", "--version"])
    res = subprocess.run(
        [sys.executable, "-m", "uv", "pip", "list"], capture_output=True, text=True
    )
    for line in res.stdout.split("\n"):
        line_lower = line.lower()
        if any(
            p in line_lower
            for p in ["fastapi", "agentscope", "qdrant", "pyautogui", "uvicorn", "langsmith"]
        ):
            print(line.strip())

    npx_res = subprocess.run(["where", "npx"], capture_output=True)
    print("npx missing" if npx_res.returncode else "npx found")

    uvx_res = subprocess.run(["where", "uvx"], capture_output=True)
    print("uvx missing" if uvx_res.returncode else "uvx found")


def audit_configs():
    print("\n--- Phase 1.2: Config Check ---")
    for f in ["configs/llm/router.yaml", "configs/safety/permissions.yaml", ".env"]:
        print(f"OK: {f}" if os.path.exists(f) else f"BREAK: Missing {f}")

    try:
        yaml.safe_load(open("configs/llm/router.yaml"))
        print("router.yaml: valid")
    except Exception as e:
        print("BREAK: router.yaml invalid YAML", e)

    try:
        yaml.safe_load(open("configs/safety/permissions.yaml"))
        print("permissions.yaml: valid")
    except Exception as e:
        print("BREAK: permissions.yaml invalid YAML", e)

    for k in ["GROQ_API_KEY", "VISION_API_URL", "MAX_CONCURRENT_TASKS"]:
        v = os.getenv(k)
        print(f"OK: {k}={v[:8]}..." if v else f"WARN: {k} is unset")


def audit_llm_reachability():
    print("\n--- Phase 1.3: LLM Reachability ---")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
                timeout=10,
            )
            if r.ok:
                print("Groq OK:", r.status_code)
            else:
                print("BREAK: Groq error", r.status_code, r.text[:200])
        except Exception as e:
            print("Groq error:", e)
    else:
        print("WARN: Groq key not set")

    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {or_key}"},
                json={
                    "model": "mistralai/mistral-7b-instruct",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
                timeout=10,
            )
            if r.ok:
                print("OpenRouter OK:", r.status_code)
            else:
                print("BREAK: OpenRouter error", r.status_code, r.text[:200])
        except Exception as e:
            print("OpenRouter error:", e)
    else:
        print("WARN: OpenRouter key not set")

    url = os.environ.get("VISION_API_URL", "")
    if not url:
        print("BREAK: VISION_API_URL not set")
    else:
        try:
            r = requests.get(f"{url}/health", timeout=10)
            print("Vision API OK:", r.status_code)
        except Exception as e:
            print("BREAK: Vision API unreachable —", e)


def wait_for_server(base_url="http://localhost:8080"):
    print("Waiting for server to start...")
    for _ in range(30):
        try:
            r = requests.get(f"{base_url}/ready")
            if r.status_code == 200:
                print("Server is READY!")
                return True
        except:
            pass
        time.sleep(1)
    print("Server failed to start")
    return False


def run_task(instruction, timeout_sec, base_url="http://localhost:8080"):
    print(f"\n=== SUBMITTING: {instruction}")
    r = requests.post(
        f"{base_url}/v1/tasks", json={"instruction": instruction, "timeout_seconds": timeout_sec}
    )
    if not r.ok:
        print("Submit error:", r.text)
        return None

    task_id = r.json().get("task_id")
    print(f"Task ID: {task_id}")

    for i in range(timeout_sec // 2):
        time.sleep(2)
        status_r = requests.get(f"{base_url}/v1/tasks/{task_id}")
        if not status_r.ok:
            continue
        data = status_r.json()
        status = data.get("status")
        print(f"  [{i}] status={status}")
        if status in ["completed", "failed", "error"]:
            break

    final_res = requests.get(f"{base_url}/v1/tasks/{task_id}").json()
    print("=== FINAL RESULT:")
    print(json.dumps(final_res, indent=2))
    return final_res


def run_integration_tests(base_url="http://localhost:8080"):
    print("\n=== Phase 1.4: Tool Discovery ===")
    tools_r = requests.get(f"{base_url}/v1/tools")
    if tools_r.ok:
        tools_data = tools_r.json()
        count = (
            len(tools_data) if isinstance(tools_data, list) else len(tools_data.get("tools", []))
        )
        print(f"Tools found: {count}")
    else:
        print("Tools endpoints failed")

    print("\n=== Task 1 --- Shell Execution ===")
    task1 = run_task(
        "Create a temporary directory at c:\\temp\\orion_test, write the current date and hostname into a file called info.txt inside it, then read and print the file contents.",
        90,
        base_url,
    )

    print("\n=== Phase 3 --- Rollback Task 1 ===")
    if task1 and task1.get("id"):
        rb = requests.post(f"{base_url}/v1/tasks/{task1['id']}/rollback")
        print("Rollback Task 1:", rb.status_code, rb.text)

    print("\n=== Task 2 --- Multi-Step Reasoning ---")
    task2 = run_task(
        "Find the top 3 Python packages for async HTTP requests ranked by GitHub stars as of this year, compare their pros and cons in a structured table, then save a summary to long-term memory tagged 'async_http_libs'.",
        150,
        base_url,
    )

    print("\n=== Task 3 --- GUI Vision Action ---")
    task3 = run_task(
        "Take a screenshot of the current desktop. Identify the taskbar or dock visible in the screenshot, return its bounding-box coordinates in JSON format, and describe what applications are pinned to it.",
        150,
        base_url,
    )

    print("\n=== Phase 4 --- Metric Check ---")
    try:
        metrics_r = requests.get("http://localhost:9091/metrics")
        if metrics_r.ok:
            print("Prometheus /metrics exposed OK!")
    except:
        print("Metrics endpoint not available")


def main():
    parser = argparse.ArgumentParser(description="ORION Audit Runner")
    parser.add_argument(
        "--mode",
        choices=["deps", "config", "llm", "integration", "all"],
        default="all",
        help="Audit mode",
    )
    parser.add_argument("--base-url", default="http://localhost:8080", help="Base URL for tests")
    args = parser.parse_args()

    if args.mode in ("deps", "all"):
        audit_dependencies()
    if args.mode in ("config", "all"):
        audit_configs()
    if args.mode in ("llm", "all"):
        audit_llm_reachability()
    if args.mode == "integration":
        if wait_for_server(args.base_url):
            run_integration_tests(args.base_url)


if __name__ == "__main__":
    main()
