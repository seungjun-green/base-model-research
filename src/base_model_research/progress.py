"""Forward training counters from the training subprocess to the notebook kernel."""

import json
import signal
import subprocess

PREFIX = "BMR_PROGRESS "


def emit_progress(label, unit, completed, total, detail=""):
    payload = {"label": label, "unit": unit, "completed": completed, "total": total, "detail": detail}
    print(PREFIX + json.dumps(payload), flush=True)


def run_with_progress(command, *, cwd):
    from tqdm.auto import tqdm

    bars = {}
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                if not line.startswith(PREFIX):
                    tqdm.write(line.rstrip("\n"))
                    continue
                event = json.loads(line[len(PREFIX) :])
                unit, label = event["unit"], event["label"]
                bar = bars.get(unit)
                if bar is None or bar.desc != label:
                    if bar is not None:
                        bar.close()
                    bar = bars[unit] = tqdm(
                        desc=label,
                        unit=unit,
                        initial=event["completed"],
                        total=event["total"],
                        dynamic_ncols=True,
                    )
                bar.set_postfix_str(event["detail"], refresh=False)
                bar.update(event["completed"] - bar.n)
        except BaseException:
            # A notebook interrupt must also stop the training subprocess.
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            raise
        finally:
            for bar in reversed(list(bars.values())):
                bar.close()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
