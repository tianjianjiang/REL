import collections
import time
import typing
import os
import contextlib

import numpy as np


def profile(func):
    def wrapper_do_twice(self, *args, **kwargs):
        with self.profiler.profile(func.__name__):
            return func(self, *args, **kwargs)

    return wrapper_do_twice


class Profiler:
    """
    Heaviliy inspired by and large portion of code copied from PyTorch lightning profiler code.

    As the number of calls may be extremely large

    """

    def __init__(self):
        self.current_actions: typing.Dict[str, float] = {}

        self.recorded_durations_sum = collections.defaultdict(float)
        self.recorded_instance_counts = collections.defaultdict(int)

        self.start_time = time.monotonic()

    def set_global_start_time(self):
        self.start_time = time.monotonic()

    def start(self, action_name: str):
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str):
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations_sum[action_name] += duration
        self.recorded_instance_counts[action_name] += 1

    @contextlib.contextmanager
    def profile(self, action_name: str):
        """
        Yields a context manager to encapsulate the scope of a profiled action.
        Example::
            with self.profile('load training data'):
                # load training data code
        The profiler will start once you've entered the context and will automatically
        stop once you exit the code block.
        """
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)

    def _make_report(self):
        total_duration = time.monotonic() - self.start_time

        # Action, sum, instance count, percentage total
        report = [
            [
                action,
                r_sum,
                self.recorded_instance_counts[action],
                100.0 * r_sum / total_duration,
            ]
            for action, r_sum in self.recorded_durations_sum.items()
        ]
        report.sort(key=lambda x: x[3], reverse=True)
        return report, total_duration

    def summary(self) -> str:
        sep = os.linesep
        output_string = ""
        output_string += f"Profiler Report{sep}"

        if len(self.recorded_durations_sum) > 0:
            max_key = np.max([len(k) for k in self.recorded_durations_sum.keys()])

            def log_row(action, mean, num_calls, total, per):
                row = f"{sep}{action:<{max_key}s}\t|  {mean:<15}\t|"
                row += f"{num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                return row

            output_string += log_row(
                "Action",
                "Mean duration (s)",
                "Num calls",
                "Total time (s)",
                "Percentage %",
            )
            output_string_len = len(output_string)
            output_string += f"{sep}{'-' * output_string_len}"
            report, total_duration = self._make_report()
            output_string += log_row("Total", "-", "_", f"{total_duration:.5}", "100 %")
            output_string += f"{sep}{'-' * output_string_len}"

            # Store report for unit testing
            self._report = {x[0]: x[1] / x[2] for x in report}

            # Action, sum, instance count, percentage total
            for action, duration_sum, instance_count, percentage in report:
                output_string += log_row(
                    action,
                    f"{duration_sum/instance_count:.5}",
                    f"{instance_count:}",
                    f"{duration_sum:.5}",
                    f"{percentage:.3} %",
                )

        output_string += sep
        return output_string
