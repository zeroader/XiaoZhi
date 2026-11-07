"""Console entry point for the portable Windows executable."""

import multiprocessing
import sys
import traceback


if __name__ == "__main__":
    multiprocessing.freeze_support()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        from main import main
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        if getattr(sys, "frozen", False) and sys.stdin is not None and sys.stdin.isatty():
            try:
                input("\nStartup failed. Press Enter to close...")
            except (EOFError, OSError):
                pass
        sys.exit(1)
