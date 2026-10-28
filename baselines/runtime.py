"""Host ABI requirements of the pinned additional-baseline wheels."""
import platform


def check_runtime(libc=None):
    """Fail before dependency downloads if upstream Linux wheels cannot load."""
    name, version = platform.libc_ver() if libc is None else libc
    try:
        parsed = tuple(int(part) for part in version.split("."))
    except ValueError:
        parsed = ()
    if name != "glibc" or len(parsed) < 2 or parsed < (2, 35):
        raise RuntimeError(
            f"Additional baselines require glibc >= 2.35; found {name!r} {version!r}. "
            "Use an Ubuntu 22.04 or newer compatible Linux container. "
            "CADFit's pinned PyMeshLab 2025.7 wheel requires manylinux_2_35; "
            "DualBrep's torch_cluster wheel also exceeds Ubuntu 20.04's glibc. "
            "Changing the isolated Python version does not upgrade host glibc."
        )
    return {"libc": name, "version": version, "minimum_glibc": "2.35"}
