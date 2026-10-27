"""Sagey entrypoint. Installs dependencies on the remote worker only."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
from urllib.parse import urlparse

root=Path(__file__).resolve().parent
arguments=sys.argv[1:]
if "--manifest-s3" in arguments:
    import boto3
    i=arguments.index("--manifest-s3")
    uri=urlparse(arguments[i+1])
    if uri.scheme!="s3" or not uri.netloc or not uri.path:
        raise ValueError("Expected a manifest S3 URI")
    manifest=Path("/tmp/irl-cad-run-manifest.json")
    boto3.client("s3").download_file(uri.netloc,uri.path.lstrip("/"),str(manifest))
    arguments[i:i+2]=["--manifest",str(manifest)]
conda=shutil.which("conda")
if not conda:
    raise RuntimeError("Sagey image needs conda")
env=Path("/tmp/irl-cad-eval-env")
python=env/"bin/python"
if not python.exists():
    subprocess.run([conda,"create","-y","-p",str(env),"--override-channels",
        "-c","conda-forge","python=3.11","pip"],check=True,timeout=1200)
subprocess.run([str(python),"-m","pip","install","--no-cache-dir","-r",
                str(root/"requirements.txt")],check=True,timeout=2400)
if not shutil.which("Xvfb"):
    subprocess.run(["apt-get","update"],check=True,timeout=300)
    subprocess.run(["apt-get","install","-y","xvfb","libgl1","libglu1-mesa",
                    "libxrender1","libosmesa6"],check=True,timeout=600)
display=":97"
xvfb=subprocess.Popen(["Xvfb",display,"-screen","0","1024x1024x24","-nolisten","tcp"])
run_env=dict(os.environ,DISPLAY=display,PYTHONUNBUFFERED="1")
out=Path("/opt/ml/checkpoints/irl-cad-pilot")
out.mkdir(parents=True,exist_ok=True)
try:
    with (out/"environment.txt").open("w") as stream:
        subprocess.run([str(python),"-m","pip","freeze"],stdout=stream,check=True)
    subprocess.run([str(python),"-m","unittest","discover","-s","tests","-v"],
                   cwd=root,env=run_env,check=True,timeout=600)
    subprocess.run([str(python),"-m","evaluation.run",*arguments],
                   cwd=root,env=run_env,check=True)
finally:
    xvfb.terminate()
