from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from pipeline import PipelineError, analyse_clip
from render_utils import RenderOptions, cleanup_old_outputs, render_tactical_view


APP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = APP_ROOT / "outputs"
EXAMPLE_VIDEO = APP_ROOT / "examples" / "sample_clip.mp4"


def _normalise_video_input(video: Any) -> str:
    if isinstance(video, dict):
        video = video.get("video") or video.get("path") or video.get("name")
    if not video:
        raise gr.Error("Please upload an MP4 clip under 10 seconds.")
    return str(video)


def _render(state, selected_team, spotlight, blur_others, team_overlay, show_ids):
    if state is None:
        raise gr.Error("Analyse a clip before changing tactical effects.")

    options = RenderOptions(
        selected_team=int(selected_team),
        spotlight_my_team=bool(spotlight),
        blur_opposition=bool(blur_others),
        team_colour_overlay=bool(team_overlay),
        show_player_ids=bool(show_ids),
    )
    return render_tactical_view(state, options, output_dir=OUTPUT_DIR)


def analyse_and_render(video, selected_team, spotlight, blur_others, team_overlay, show_ids):
    cleanup_old_outputs(OUTPUT_DIR)
    video_path = _normalise_video_input(video)

    try:
        state = analyse_clip(video_path)
        output_path = _render(
            state,
            selected_team,
            spotlight,
            blur_others,
            team_overlay,
            show_ids,
        )
    except PipelineError as exc:
        raise gr.Error(str(exc)) from exc
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    return (
        output_path,
        gr.update(value=output_path, interactive=True),
        state,
        "Analysis complete.",
    )


def rerender(state, selected_team, spotlight, blur_others, team_overlay, show_ids):
    if state is None:
        return (
            None,
            gr.update(value=None, interactive=False),
            "Analyse a clip first.",
        )

    try:
        output_path = _render(
            state,
            selected_team,
            spotlight,
            blur_others,
            team_overlay,
            show_ids,
        )
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    return (
        output_path,
        gr.update(value=output_path, interactive=True),
        "View updated.",
    )


with gr.Blocks(title="Coach's Tactical Lens") as demo:
    gr.Markdown("# Coach's Tactical Lens")

    analysis_state = gr.State()

    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(
                label="Upload video",
                sources=["upload"],
                format="mp4",
                height=320,
            )
            analyse_button = gr.Button("Analyse clip", variant="primary", size="lg")
            output_video = gr.Video(
                label="Output video",
                interactive=False,
                height=420,
            )
            download_button = gr.DownloadButton(
                label="Download output",
                value=None,
                interactive=False,
            )

        with gr.Column(scale=1):
            team_selector = gr.Radio(
                choices=[("Team 0", 0), ("Team 1", 1)],
                value=0,
                label="Team selector",
            )

            spotlight_effect = gr.Checkbox(label="Spotlight my team", value=True)
            blur_effect = gr.Checkbox(label="Blur opposition", value=False)
            overlay_effect = gr.Checkbox(label="Team colour overlay", value=True)
            ids_effect = gr.Checkbox(label="Show player IDs", value=False)

            status = gr.Textbox(
                label="Status",
                value="Ready.",
                interactive=False,
                lines=1,
            )

    analyse_button.click(
        fn=analyse_and_render,
        inputs=[
            video_input,
            team_selector,
            spotlight_effect,
            blur_effect,
            overlay_effect,
            ids_effect,
        ],
        outputs=[output_video, download_button, analysis_state, status],
    )

    for control in [
        team_selector,
        spotlight_effect,
        blur_effect,
        overlay_effect,
        ids_effect,
    ]:
        control.change(
            fn=rerender,
            inputs=[
                analysis_state,
                team_selector,
                spotlight_effect,
                blur_effect,
                overlay_effect,
                ids_effect,
            ],
            outputs=[output_video, download_button, status],
        )

    if EXAMPLE_VIDEO.exists():
        gr.Examples(
            examples=[[str(EXAMPLE_VIDEO)]],
            inputs=[video_input],
        )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo.queue(max_size=4).launch()
