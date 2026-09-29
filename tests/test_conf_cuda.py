"""GPU regression checks for the compiled, per-Gaussian CUDA Conf window.

Run with the freshly built extension first on PYTHONPATH, for example through
``scripts/run_conf_cuda_validation.sh kernel``.  The reference below uses
Python deques and float64 arithmetic; it does not call production Conf code.
"""

from collections import deque
import math
import unittest

import torch


def _state(n, window, device="cuda"):
    return {
        "history": torch.zeros((n, window, 4), device=device, dtype=torch.float32),
        "camera_ids": torch.full((n, window), -1, device=device, dtype=torch.int64),
        "view_count": torch.zeros((n, 1), device=device, dtype=torch.int32),
        "world_sum": torch.zeros((n, 3), device=device, dtype=torch.float32),
        "norm_sum": torch.zeros((n, 1), device=device, dtype=torch.float32),
        "conf": torch.zeros((n, 1), device=device, dtype=torch.float32),
    }


def _reference_push(rows, camera_id, samples, window):
    """Update independent ordered histories; invalid w sentinels do nothing."""
    for i, sample in enumerate(samples):
        x, y, z, valid = (float(v) for v in sample)
        if not valid >= 0 or not all(math.isfinite(v) for v in (x, y, z, valid)):
            continue
        vector = (x, y, z)
        magnitude = math.hypot(math.hypot(x, y), z)
        if not math.isfinite(magnitude):
            continue
        row = rows[i]
        for entry in tuple(row):
            if entry[0] == camera_id:
                row.remove(entry)
                break
        if len(row) == window:
            row.popleft()
        row.append((camera_id, vector, magnitude))


def _assert_state_matches(test, state, rows):
    torch.cuda.synchronize()
    history = state["history"].cpu().double()
    ids = state["camera_ids"].cpu()
    counts = state["view_count"].cpu().flatten()
    sums = state["world_sum"].cpu().double()
    norms = state["norm_sum"].cpu().double().flatten()
    scores = state["conf"].cpu().double().flatten()
    for i, row in enumerate(rows):
        entries = list(row)
        test.assertEqual(int(counts[i]), len(entries), f"row {i} count")
        expected_ids = [entry[0] for entry in entries]
        test.assertEqual(ids[i, :len(entries)].tolist(), expected_ids, f"row {i} order")
        test.assertEqual(ids[i, len(entries):].tolist(), [-1] * (ids.shape[1] - len(entries)))
        expected_sum = [sum(entry[1][axis] for entry in entries) for axis in range(3)]
        expected_norm = sum(entry[2] for entry in entries)
        for axis in range(3):
            test.assertAlmostEqual(float(sums[i, axis]), expected_sum[axis], delta=2e-5)
        test.assertAlmostEqual(float(norms[i]), expected_norm, delta=2e-5)
        # Active xyz and magnitudes are also observable, not just aggregate scores.
        for j, (_, vector, magnitude) in enumerate(entries):
            test.assertEqual(int(ids[i, j]), expected_ids[j])
            test.assertTrue(torch.allclose(history[i, j, :3], torch.tensor(vector, dtype=torch.float64), atol=2e-6, rtol=2e-6))
            test.assertAlmostEqual(float(history[i, j, 3]), magnitude, delta=2e-6)
        score = 0.0
        if len(entries) >= 2 and expected_norm > 0:
            length = math.hypot(math.hypot(expected_sum[0], expected_sum[1]), expected_sum[2])
            score = min(1.0, max(0.0, 1.0 - length / expected_norm))
        test.assertTrue(math.isfinite(float(scores[i])))
        test.assertGreaterEqual(float(scores[i]), -1e-7)
        test.assertLessEqual(float(scores[i]), 1.0 + 1e-7)
        test.assertAlmostEqual(float(scores[i]), score, delta=2e-5)


class TestConfCudaKernel(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device unavailable")

    def test_ordered_window_matches_float64_deque_at_capacities(self):
        # Three rows have different visibility patterns.  The stream includes
        # eviction, duplicate refresh, long repeats, zero gradients, NaNs, and
        # invalid markers; every operation is checked against a Python deque.
        generator = torch.Generator(device="cpu").manual_seed(8122026)
        from diff_gaussian_rasterization import accumulate_conf
        for window in (2, 3, 5):
            state = _state(3, window)
            refs = [deque(), deque(), deque()]
            sequence = [8, 3, 5, 8, 7, 12, 3, 2, 2, 5, 19, 8]
            sequence += [int(torch.randint(0, 7, (), generator=generator)) for _ in range(257)]
            for step, camera_id in enumerate(sequence):
                samples = torch.randn((3, 4), generator=generator, dtype=torch.float32)
                samples[:, 3] = torch.linalg.vector_norm(samples[:, :3], dim=1)
                # Independent late visibility for rows 1 and 2.
                if step % 4 == 0:
                    samples[1] = torch.tensor([float("nan"), 1.0, 2.0, 0.0])
                if step % 5 in (0, 1):
                    samples[2] = torch.tensor([1.0, 2.0, 3.0, -1.0])
                # Exercise a valid visible zero and ensure it consumes a slot.
                if step in (2, 8, 22, 110):
                    samples[step % 3] = 0
                _reference_push(refs, camera_id, samples.tolist(), window)
                samples = samples.to("cuda").contiguous()
                accumulate_conf(samples, camera_id, state["history"], state["camera_ids"],
                                state["view_count"], state["world_sum"], state["norm_sum"],
                                state["conf"])
                _assert_state_matches(self, state, refs)

            # Each window must end with only its most recent distinct camera IDs.
            self.assertTrue((state["view_count"] <= window).all().item())

    def test_empty_singleton_invalid_arguments_and_nondefault_stream(self):
        from diff_gaussian_rasterization import accumulate_conf
        # P=0 is a valid no-op and P=1 preserves the (1,*) shapes.
        empty = _state(0, 3)
        out = accumulate_conf(torch.empty((0, 4), device="cuda"), 0,
                              empty["history"], empty["camera_ids"], empty["view_count"],
                              empty["world_sum"], empty["norm_sum"], empty["conf"])
        self.assertEqual(tuple(out.shape), (0, 1))

        one = _state(1, 2)
        samples = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device="cuda")
        accumulate_conf(samples, 11, one["history"], one["camera_ids"], one["view_count"],
                        one["world_sum"], one["norm_sum"], one["conf"])
        self.assertEqual(one["camera_ids"].tolist(), [[11, -1]])
        self.assertEqual(one["conf"].item(), 0.0)

        before_ids = one["camera_ids"].clone()
        before_count = one["view_count"].clone()
        bad_samples = torch.tensor([[float("inf"), 0.0, 0.0, 0.0]], device="cuda")
        accumulate_conf(bad_samples, 12, one["history"], one["camera_ids"], one["view_count"],
                        one["world_sum"], one["norm_sum"], one["conf"])
        self.assertTrue(torch.equal(before_ids, one["camera_ids"]))
        self.assertTrue(torch.equal(before_count, one["view_count"]))

        with self.assertRaisesRegex(RuntimeError, "nonnegative"):
            accumulate_conf(samples, -1, one["history"], one["camera_ids"], one["view_count"],
                            one["world_sum"], one["norm_sum"], one["conf"])
        bad_window = _state(1, 3)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            accumulate_conf(samples, 1, one["history"], bad_window["camera_ids"], one["view_count"],
                            one["world_sum"], one["norm_sum"], one["conf"])

        # Producer and consumer are queued on a nondefault stream, and the
        # kernel must observe the producer without a host synchronization.
        stream = torch.cuda.Stream()
        stream_state = _state(1, 3)
        with torch.cuda.stream(stream):
            produced = torch.empty((1, 4), device="cuda")
            produced.fill_(0)
            produced[0, 0] = 2.0
            produced[0, 3] = 2.0
            accumulate_conf(produced, 20, stream_state["history"], stream_state["camera_ids"],
                            stream_state["view_count"], stream_state["world_sum"],
                            stream_state["norm_sum"], stream_state["conf"])
            produced2 = torch.tensor([[-2.0, 0.0, 0.0, 2.0]], device="cuda")
            accumulate_conf(produced2, 21, stream_state["history"], stream_state["camera_ids"],
                            stream_state["view_count"], stream_state["world_sum"],
                            stream_state["norm_sum"], stream_state["conf"])
        stream.synchronize()
        self.assertEqual(stream_state["view_count"].item(), 2)
        self.assertAlmostEqual(stream_state["conf"].item(), 1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
