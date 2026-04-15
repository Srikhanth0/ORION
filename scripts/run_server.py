import argparse
import os
import sys

import uvicorn

sys.path.insert(0, "C:/Users/srikh/Downloads/ORION")
os.chdir("C:/Users/srikh/Downloads/ORION")


def run_server(port: int = 8080, host: str = "0.0.0.0", reload: bool = False):
    uvicorn.run(
        "orion.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(description="ORION Server Runner")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host, reload=args.reload)


if __name__ == "__main__":
    main()
