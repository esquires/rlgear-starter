import tempfile
from pathlib import Path

import plotly.graph_objects as go
import rlgear.postprocess
import rlgear.utils


def name_cb(path: Path) -> str:
    # return name/group of experiment
    return path.parent.parent.parent.name


def write_figure(out_dir: Path, fig: go.Figure, name: str) -> None:
    name = name.replace(" ", "_")
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", showgrid=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black", showgrid=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.write_html(out_dir / f"{name}.html")
    fig.write_image(out_dir / f"{name}.png")
    fig.show()


def main() -> None:
    base_plot_dir = Path("~/ray/plotting/rlgear_starter").expanduser()
    base_plot_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(dir=base_plot_dir))
    out_dir = rlgear.utils.write_metadata("rlgear_starter.yaml", out_dir=out_dir)

    progress_reader = rlgear.postprocess.ProgressReader()

    base_data_dirs = [Path("~/ray/rlgear_starter").expanduser()]
    experiments = rlgear.postprocess.group_experiments(base_data_dirs, name_cb)
    experiments = {k: experiments[k] for k in sorted(experiments)}

    tags = [
        ("Episode Reward", "episode_reward_mean"),
        ("Episode Length", "episode_len_mean"),
    ]

    for name, tag in tags:

        dfs = rlgear.postprocess.get_dataframes(experiments, progress_reader, tag=tag)
        fig = rlgear.postprocess.plot_progress(
            dfs, plot_indiv=True, percentiles=[0.1, 0.9]
        )
        fig.update_layout(xaxis_title="Timesteps", yaxis_title=name)
        write_figure(out_dir, fig, name)


if __name__ == "__main__":
    main()
