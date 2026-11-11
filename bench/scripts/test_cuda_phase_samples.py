import itertools
import json
import unittest

import cuda_phase_samples as samples

COMPLETE = "test result: ok. 1 passed; 0 failed; 0 ignored; 86 filtered out; finished in 1s"


def rows(values=(3.0, 1.0, 2.0)):
    return [
        f"phase_samples {scalar_set} {group} n={n} c=13 phase={phase} ms={json.dumps(values)}"
        for scalar_set, group, n, phase in itertools.product(
            samples.SETS, samples.GROUPS, samples.SIZES, samples.PHASES
        )
    ]


def log(lines):
    return "\n".join(["shapes ignored", *lines, COMPLETE])


class PhaseSamples(unittest.TestCase):
    def test_complete_raw_samples_and_clocks(self):
        result = samples.parse_samples(log(rows()), 3)
        self.assertEqual(len(result["rows"]), 88)
        self.assertEqual(result["repetitions"], 3)
        for row in result["rows"]:
            self.assertEqual(row["samples_ms"], [3, 1, 2])
            self.assertEqual((row["min_ms"], row["median_ms"], row["max_ms"]), (1, 2, 3))
            expected = "wall" if row["phase"] in ("combine", "batch_wall") else "device"
            self.assertEqual(row["clock"], expected)

    def test_even_median_matches_rust_mean_of_middle_pair(self):
        result = samples.parse_samples(log(rows((1.0, 3.0))), 2)
        self.assertEqual(result["rows"][0]["median_ms"], 2)

    def test_nonfinite_derived_median_is_refused(self):
        with self.assertRaises(ValueError):
            samples.parse_samples(log(rows((1.7e308, 1.7e308))), 2)

    def test_zero_is_valid_at_driver_clock_resolution(self):
        samples.parse_samples(log(rows((0.0,))), 1)

    def test_missing_duplicate_and_malformed_rows_are_refused(self):
        valid = rows()
        for lines in (valid[1:], [*valid, valid[0]], [*valid, "phase_samples bad"]):
            with self.subTest(lines=len(lines)), self.assertRaises(ValueError):
                samples.parse_samples(log(lines), 3)

    def test_unexpected_configuration_is_refused(self):
        for old, new in (("c=13", "c=12"), ("n=65536", "n=1"),
                         ("mixed", "ones"), ("phase=count", "phase=unknown"),
                         ("g1", "g3")):
            valid = rows()
            valid[0] = valid[0].replace(old, new)
            with self.subTest(new=new), self.assertRaises(ValueError):
                samples.parse_samples(log(valid), 3)

    def test_nonfinite_negative_boolean_nested_and_wrong_count_are_refused(self):
        for values in ([float("nan")] * 3, [float("inf")] * 3, [-1] * 3,
                       [True] * 3, [[1]] * 3, [10 ** 400] * 3, [1, 2]):
            with self.subTest(values=str(values)[:30]), self.assertRaises(ValueError):
                samples.parse_samples(log(rows(values)), 3)

    def test_failure_missing_or_multiple_completion_is_refused(self):
        valid = log(rows())
        for text in (valid.replace(COMPLETE, ""), valid.replace("ok. 1 passed", "FAILED. 0 passed"),
                     valid + "\n" + COMPLETE):
            with self.subTest(ending=text[-100:]), self.assertRaises(ValueError):
                samples.parse_samples(text, 3)

    def test_invalid_repetitions_are_refused(self):
        for count in (0, -1, 2):
            with self.subTest(count=count), self.assertRaises(ValueError):
                samples.parse_samples(log(rows()), count)


if __name__ == "__main__":
    unittest.main()
