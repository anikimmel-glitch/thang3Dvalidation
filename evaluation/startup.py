"""Standard-library checks for the disposable Sagey container."""
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import stat
import subprocess
import sys
import time


def temp_access(directory, sandbox_user="_apt"):
    """Restore standard shared-temp permissions and test the APT user's access."""
    directory = Path(directory)
    before = directory.stat()
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"Temporary path is not a directory: {directory}")
    mode = stat.S_IMODE(before.st_mode)
    if mode != 0o1777:
        if os.geteuid() != 0 or before.st_uid != 0:
            raise RuntimeError(f"Cannot repair temporary-directory mode {oct(mode)}")
        directory.chmod(0o1777)
    disk = shutil.disk_usage(directory)
    inodes = os.statvfs(directory)
    print(json.dumps(dict(stage="temporary_directory", path=str(directory),
          mode_before=oct(mode), mode_after=oct(stat.S_IMODE(directory.stat().st_mode)),
          uid=before.st_uid, gid=before.st_gid, free_bytes=disk.free,
          free_inodes=inodes.f_favail)), flush=True)
    # A root-only write test would miss the APT sandbox user's failure.
    account = pwd.getpwnam(sandbox_user)
    subprocess.run([sys.executable, "-c",
        "import os,sys,tempfile; "
        "fd,p=tempfile.mkstemp(prefix='apt-access-probe-',dir=sys.argv[1]); "
        "os.write(fd,b'ok'); os.close(fd); os.unlink(p); "
        "print('APT sandbox temporary-file probe passed',flush=True)",
        str(directory)], user=account.pw_uid, group=account.pw_gid, extra_groups=[],
        env={"PATH": "/usr/bin:/bin"}, check=True, timeout=30)


def wait_display(process, display, timeout=15):
    """Wait for Xvfb's socket, rather than racing its startup."""
    endpoint = f"/tmp/.X11-unix/X{display.lstrip(':').split('.')[0]}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Xvfb exited during startup: {process.returncode}")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(endpoint)
            return
        except OSError:
            time.sleep(.1)
    raise RuntimeError("Xvfb did not become ready; inspect xvfb.log")
