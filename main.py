import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        import os
        import uvicorn
        uvicorn.run("jan_agent.api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
    else:
        from jan_agent.core import main
        raise SystemExit(main())
