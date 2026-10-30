"""Reproduce the new guarded serial failure with dedicated fatal traceback output."""
from native_bootstrap import main


if __name__ == "__main__":
    main(null_face_guard=True, focused_guard_failure=True)
