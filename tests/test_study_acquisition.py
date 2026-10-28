import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from studies.acquire import acquire, sdk_runtime


class StudyAcquisition(unittest.TestCase):
    def row(self):
        raw=b"ISO-10303-21;\nfixture"
        return raw, dict(pilot_id="x",file="part.step",bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest(),
                         fetch=dict(kind="s3",uri="s3://fixture/part.step"))

    def test_external_process_receives_frozen_rows_and_output_is_verified(self):
        raw,row=self.row()
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            def run(command,**kwargs):
                self.assertEqual(command[:3],["/image/python","-m","studies.acquire"])
                self.assertEqual(json.loads(Path(command[4]).read_text()),[row])
                self.assertTrue(kwargs["check"])
                (out/"inputs/part.step").write_bytes(raw)
            with patch("studies.acquire.subprocess.run",side_effect=run):
                self.assertEqual(acquire([row],out,"/image/python"),out/"inputs")

    def test_successful_process_cannot_hide_missing_or_corrupt_geometry(self):
        raw,row=self.row()
        for contents in (None,raw[:-1],b"x"*len(raw)):
            with tempfile.TemporaryDirectory() as temp:
                out=Path(temp)
                def run(*args,**kwargs):
                    if contents is not None:(out/"inputs/part.step").write_bytes(contents)
                with patch("studies.acquire.subprocess.run",side_effect=run):
                    with self.assertRaisesRegex(RuntimeError,"frozen geometry hashes"):
                        acquire([row],out,"/image/python")

    def test_acquisition_failure_propagates(self):
        _,row=self.row()
        with tempfile.TemporaryDirectory() as temp:
            with patch("studies.acquire.subprocess.run",
                       side_effect=subprocess.CalledProcessError(1,["acquire"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    acquire([row],Path(temp),"/image/python")

    def test_sdk_canary_checks_s3_without_creating_a_client(self):
        class Session:
            def get_available_services(self):return ["s3"]
        boto=types.SimpleNamespace(Session=Session,__version__="fixture")
        with patch.dict("sys.modules",boto3=boto,
                        botocore=types.SimpleNamespace(__version__="fixture")):
            self.assertTrue(sdk_runtime()["s3_service_available"])
        class NoS3:
            def get_available_services(self):return []
        with patch.dict("sys.modules",boto3=types.SimpleNamespace(Session=NoS3),
                        botocore=types.SimpleNamespace(__version__="fixture")):
            with self.assertRaisesRegex(RuntimeError,"S3 service"):sdk_runtime()
