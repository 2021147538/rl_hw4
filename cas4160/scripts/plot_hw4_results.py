import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Run spec '{spec}' must be of the form label=/path/to/logdir"
        )
    label, path = spec.split("=", 1)
    return label, Path(path).expanduser().resolve()


def load_scalar(logdir: Path, scalar_tags: list[str]) -> tuple[str, list[int], list[float]]:
    event_acc = EventAccumulator(str(logdir))
    event_acc.Reload()
    available = set(event_acc.Tags().get("scalars", []))

    for tag in scalar_tags:
        if tag in available:
            scalars = event_acc.Scalars(tag)
            return tag, [s.step for s in scalars], [s.value for s in scalars]

    raise ValueError(
        f"No matching scalar tags found in {logdir}. "
        f"Available scalars: {sorted(available)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot HW4 TensorBoard scalars from one or more log directories."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification in the form label=/absolute/or/relative/logdir. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--scalar-tags",
        nargs="+",
        required=True,
        help="One or more scalar tag candidates. The first tag present in each run will be used.",
    )
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument(
        "--x-label", type=str, default="Environment Steps"
    )
    parser.add_argument("--y-label", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="PNG filename to write inside report_figures/ unless an absolute path is given.",
    )

    args = parser.parse_args()

    if Path(args.output).is_absolute():
        output_path = Path(args.output)
    else:
        output_path = Path("report_figures") / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for run_spec in args.run:
        label, logdir = parse_run_spec(run_spec)
        used_tag, steps, values = load_scalar(logdir, args.scalar_tags)
        plt.plot(steps, values, label=f"{label} ({used_tag})")

    plt.title(args.title)
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
