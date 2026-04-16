import sys

import uvicorn

sys.path.insert(0, "C:/Users/srikh/Downloads/ORION")

# Change to project root
import os

os.chdir("C:/Users/srikh/Downloads/ORION")

# Start server
if __name__ == "__main__":
    uvicorn.run(
        "orion.api.server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
