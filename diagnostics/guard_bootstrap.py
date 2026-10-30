"""Test a null-face guard on frozen train-only candidates and original sequences."""
from native_bootstrap import main


if __name__ == "__main__":
    main(null_face_guard=True)
