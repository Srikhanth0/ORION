import asyncio
import argparse
import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()


async def test_groq_model():
    from orion.agentscope_config import build_model

    model = build_model("groq")
    print("Groq model initialized.")
    try:
        resp = await model([{"role": "user", "content": "hi"}])
        print("RESP:", getattr(resp, "content", resp))
    except Exception as e:
        print("Error:", e)


async def test_openai_client():
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            timeout=15.0,
        )
        print("RESP:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)


async def test_sync_model():
    os.environ["OPENAI_API_KEY"] = os.environ.get("GROQ_API_KEY")
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    from agentscope.model import OpenAIChatModel

    model = OpenAIChatModel(model_name="llama-3.3-70b-versatile")
    try:
        resp = model(messages=[{"role": "user", "content": "hi"}], stream=False)
        print(f"Resp type: {type(resp)}")
        print(f"Resp text: {getattr(resp, 'text', 'No text')}")
    except Exception as e:
        print("ERROR:", e)


async def test_final_model():
    os.environ["OPENAI_API_KEY"] = os.environ.get("GROQ_API_KEY")
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    from agentscope.model import OpenAIChatModel

    model = OpenAIChatModel(model_name="llama-3.3-70b-versatile")
    resp = await model(messages=[{"role": "user", "content": "count to 3"}])
    import inspect

    if inspect.isasyncgen(resp):
        async for chunk in resp:
            content = chunk.get("content")
            print(f"Chunk content: {content}")
    else:
        print(f"Resp: {resp}")


async def test_planner():
    from orion.agents.planner import PlannerAgent
    from agentscope.message import Msg

    planner = PlannerAgent()
    msg = Msg(name="user", role="user", content="list files in current directory")
    result = await planner.reply(msg)
    print(result.content)


async def test_full_pipeline():
    from orion.agents.planner import PlannerAgent
    from orion.agents.executor import ExecutorAgent
    from orion.tools.registry import ToolRegistry
    from agentscope.message import Msg

    registry = ToolRegistry.get_instance()
    registry.load_from_config()
    registry.register_os_tools()
    registry.register_vision_tools()
    print("Native tools:", list(registry._native_tools.keys()))
    planner = PlannerAgent()
    executor = ExecutorAgent(tool_registry=registry)
    msg = Msg(name="user", role="user", content="list files in current directory")
    plan_msg = await planner.reply(msg)
    plan = json.loads(plan_msg.content)
    print("\n=== Plan ===")
    print(json.dumps(plan, indent=2))
    exec_msg = Msg(name="planner", role="assistant", content=json.dumps(plan))
    result = await executor.reply(exec_msg)
    print("\n=== Result ===")
    print(result.content)


async def test_executor():
    from orion.agents.planner import PlannerAgent
    from orion.agents.executor import ExecutorAgent
    from orion.tools.registry import ToolRegistry
    from agentscope.message import Msg

    registry = ToolRegistry.get_instance()
    registry.load_from_config()
    registry.register_vision_tools()
    planner = PlannerAgent()
    executor = ExecutorAgent(tool_registry=registry)
    msg = Msg(name="user", role="user", content="list files in .")
    plan_msg = await planner.reply(msg)
    plan = json.loads(plan_msg.content)
    print("=== Plan ===")
    print(json.dumps(plan, indent=2))
    subtask = plan["subtasks"][0]
    tool_name = subtask.get("tool")
    print(f"\nLooking for tool: {tool_name}")
    print(f"Registry tools: {registry.list_tools()[:5]}...")
    try:
        tool = registry.get(tool_name)
        print(f"Found: {tool}")
    except Exception as e:
        print(f"Not found: {e}")
    print(f"Native tools: {list(registry._native_tools.keys())}")
    exec_msg = Msg(name="planner", role="assistant", content=json.dumps(plan))
    result = await executor.reply(exec_msg)
    print("\n=== Result ===")
    print(result.content)


async def test_native_tools():
    from orion.tools.registry import ToolRegistry

    registry = ToolRegistry.get_instance()
    registry.register_os_tools()
    print(f"Native tools: {list(registry._native_tools.keys())}")
    result = await registry.call_native("list_directory", {"path": "."})
    print(f"Result: {result}")


async def test_os_tools():
    from orion.tools.registry import ToolRegistry

    registry = ToolRegistry.get_instance()
    registry.register_os_tools()
    registry.register_vision_tools()
    print(f"Total tools: {len(registry.list_tools())}")
    print("\n=== Testing list_directory ===")
    result = await registry.call_native("list_directory", {"path": "."})
    print(result)
    print("\n=== Testing read_text_file ===")
    result = await registry.call_native("read_text_file", {"path": "orion_cli.py"})
    print(f"Read {len(str(result))} chars")


async def test_tool_registry():
    from orion.tools.registry import ToolRegistry

    registry = ToolRegistry.get_instance()
    registry.register_vision_tools()
    print("Testing native tool...")
    print(f"Native tools: {list(registry._native_tools.keys())}")
    print(f"\nMCP tools discovered: {len(registry.list_tools())}")
    print(f"Tool names: {[t.name for t in registry.list_tools()]}")


async def test_tool_mapping():
    from orion.tools.executor import ExecutorAgent
    from orion.tools.registry import ToolRegistry

    registry = ToolRegistry.get_instance()
    registry.load_from_config()
    registry.register_os_tools()
    registry.register_vision_tools()
    executor = ExecutorAgent(tool_registry=registry)
    native_tools = getattr(registry, "_native_tools", {})
    print(f"Registry native tools: {list(native_tools.keys())}")
    tool_name = "execute_command"
    tool_name_mappings = {
        "execute_command": "list_directory",
        "execute_shell": "list_directory",
        "bash": "list_directory",
    }
    print(f"Mapping exists: {tool_name in tool_name_mappings}")
    if tool_name in tool_name_mappings:
        mapped = tool_name_mappings[tool_name]
        print(f"Mapped to: {mapped}")
        print(f"Is in native_tools: {mapped in native_tools}")


async def test_api():
    import httpx

    print("1. Submitting Task...")
    resp = httpx.post(
        "http://localhost:8080/v1/tasks", json={"instruction": "List all files in /tmp directory"}
    )
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("task_id")
    print(f"Task ID: {task_id}")
    print("\n2. Streaming Events...")
    url = f"http://localhost:8080/v1/tasks/{task_id}/stream"
    with httpx.stream("GET", url) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                print(line)


async def run_all_tests():
    print("=" * 50)
    print("Running all quick tests...")
    print("=" * 50)
    await test_groq_model()
    print()
    await test_os_tools()
    print()
    await test_native_tools()
    print("=" * 50)
    print("All tests completed")
    print("=" * 50)


TESTS = {
    "groq": test_groq_model,
    "openai": test_openai_client,
    "sync": test_sync_model,
    "final": test_final_model,
    "planner": test_planner,
    "full": test_full_pipeline,
    "executor": test_executor,
    "native": test_native_tools,
    "os": test_os_tools,
    "tool": test_tool_registry,
    "mapping": test_tool_mapping,
    "api": test_api,
    "all": run_all_tests,
}


def main():
    parser = argparse.ArgumentParser(description="ORION Test Runner")
    parser.add_argument("test", choices=list(TESTS.keys()), help="Test to run")
    args = parser.parse_args()
    asyncio.run(TESTS[args.test]())


if __name__ == "__main__":
    main()
