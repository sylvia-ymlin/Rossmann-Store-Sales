import uvicorn
import os
import sys

# Ensure the root directory is in the path for internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi_app.main import app

if __name__ == "__main__":
    # Hugging Face Spaces expects the app to run on port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
