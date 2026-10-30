from pathlib import Path
import tempfile
import unittest

from baselines.rejected_candidates import all_candidates_rejected


class RejectedCandidatesTests(unittest.TestCase):
    def check(self, skipped=None, header=None, has_candidate=False, execution=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/"output"
            output.mkdir()
            if has_candidate:
                (output/"post.npz").write_bytes(b"candidate")
            skipped = [(f"f00112", i) for i in range(24)] if skipped is None else skipped
            header = header or "1 shape(s) x 24 rotation(s), latent optimisation 200 steps, weights {}."
            log = root/"stage_1.log"
            log.write_text(header+"\n"+"\n".join(
                f"[{part} rot{i:02d}] <2 faces, skipping" for part,i in skipped)+"\n")
            return all_candidates_rejected(output,log,"f00112",execution or dict(
                returncode=0,signal=None,timeout=False))

    def test_complete_successfully_processed_empty_cohort_is_model_failure(self):
        result = self.check()
        self.assertEqual(result["rejected_count"],24)
        self.assertFalse(result["source_geometry_invalidity_proven"])

    def test_incomplete_duplicate_and_wrong_subject_logs_remain_unexplained(self):
        for skipped in ([],[("f00112",i) for i in range(23)],
                        [("f00112",0)]*24,[("f00113",i) for i in range(24)]):
            self.assertIsNone(self.check(skipped=skipped))

    def test_existing_candidate_or_changed_cohort_is_not_empty_rejection(self):
        self.assertIsNone(self.check(has_candidate=True))
        self.assertIsNone(self.check(header="2 shape(s) x 24 rotation(s), latent optimisation 200 steps"))

    def test_failed_crashed_or_timed_out_prior_stage_is_not_model_failure(self):
        for execution in (dict(returncode=1,signal=None,timeout=False),
                          dict(returncode=-11,signal="SIGSEGV",timeout=False),
                          dict(returncode=-9,signal="SIGKILL",timeout=True)):
            self.assertIsNone(self.check(execution=execution))
