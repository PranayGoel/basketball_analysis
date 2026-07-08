import unittest

from pipeline.progress import ProgressReporter, STAGE_NAMES


class TestProgressReporter(unittest.TestCase):
    def test_stage_emits_running_then_done(self):
        events = []
        reporter = ProgressReporter(callback=lambda *args: events.append(args))

        with reporter.stage(STAGE_NAMES[0]):
            pass

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][0], STAGE_NAMES[0])
        self.assertEqual(events[0][3], "running")
        self.assertEqual(events[1][0], STAGE_NAMES[0])
        self.assertEqual(events[1][3], "done")

    def test_stage_reports_correct_index_and_total(self):
        events = []
        reporter = ProgressReporter(callback=lambda *args: events.append(args))
        target_index = 2

        with reporter.stage(STAGE_NAMES[target_index]):
            pass

        self.assertEqual(events[0][1], target_index)
        self.assertEqual(events[0][2], len(STAGE_NAMES))

    def test_emits_done_even_if_stage_body_raises(self):
        events = []
        reporter = ProgressReporter(callback=lambda *args: events.append(args))

        with self.assertRaises(ValueError):
            with reporter.stage(STAGE_NAMES[0]):
                raise ValueError("boom")

        statuses = [e[3] for e in events]
        self.assertEqual(statuses, ["running", "done"])

    def test_none_callback_is_a_no_op(self):
        reporter = ProgressReporter(callback=None)
        with reporter.stage(STAGE_NAMES[0]):
            pass  # must not raise

    def test_unknown_stage_name_reports_index_minus_one(self):
        events = []
        reporter = ProgressReporter(callback=lambda *args: events.append(args))

        with reporter.stage("not_a_real_stage"):
            pass

        self.assertEqual(events[0][1], -1)


if __name__ == "__main__":
    unittest.main()
