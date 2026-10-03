import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("calibrate", Path(__file__).parents[1] / "scripts/calibrate_yolo.py")
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


class DetectionGateTest(unittest.TestCase):
    def box(self, x=0, cls=2, score=.8):
        return dict(x1=x, y1=1, x2=x+5, y2=6, class_id=cls, score=score)

    def test_missing_false_positive_class_and_score_reject(self):
        original = [self.box()]
        for changed in ([], original*2, [self.box(cls=3)], [self.box(score=.83)], [self.box(x=1.01)]):
            self.assertFalse(calibrate.detections_match(original, changed))

    def test_permutation_and_tolerances(self):
        self.assertTrue(calibrate.detections_match([self.box(), self.box(20)], [self.box(20.5), self.box(.5, score=.81)]))

    def test_duplicate_cannot_match_same_candidate(self):
        self.assertFalse(calibrate.detections_match([self.box(), self.box(.5)], [self.box(), self.box(20)]))

    def test_non_finite_reject(self):
        self.assertFalse(calibrate.detections_match([self.box()], [self.box(score=float("nan"))]))

    def test_graph_every_round_must_improve(self):
        def report(values, enabled=True):
            return dict(runtime=dict(cuda_graph=enabled), rounds=[dict(lanes=[dict(p95_ms=n)]) for n in values])
        self.assertTrue(calibrate.graph_qualified([report([20,20,20])], [report([18,17,17])]))
        self.assertFalse(calibrate.graph_qualified([report([20,20,20])], [report([18,19,17])]))
        self.assertFalse(calibrate.graph_qualified([report([20])], [report([15], False)]))

    def test_selection_tail_target_and_resource_tie(self):
        def entry(p50, p95, device=2, threads=1):
            return dict(p50=p50, p95=p95, device=device, threads=threads, sessions=5,
                        reports=[dict(rounds=[dict(throughput_per_s=250)])])
        cpu, gpu = entry(19, 28), entry(18.5, 27.5, 3)
        self.assertIs(calibrate.choose_candidate([gpu, cpu]), cpu)
        bad_tail, stable = entry(15, 31), entry(20, 29)
        self.assertIs(calibrate.choose_candidate([bad_tail, stable]), stable)


if __name__ == "__main__":
    unittest.main()
