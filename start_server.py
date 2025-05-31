"""
Script to start the server without auto-reload to avoid dependency issues.
"""
import uvicorn

if __name__ == "__main__":
    print("Starting server WITHOUT auto-reload to avoid dependency issues...")
    print("This prevents hanging during reload with heavy ML libraries.")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
