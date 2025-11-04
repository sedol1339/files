from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

DATA_PATH = Path("papers.json")
ALL_TAG_VALUE = "all"
DEFAULT_SEARCH_TEXT = "*"


@dataclass(slots=True, frozen=True)
class Paper:
    body: str
    authors: str
    year: str
    title: str
    url: str
    abbrev: str
    tags: list[str]


papers: list[Paper] = []


def load_papers(path: Path) -> list[Paper]:
    if not path.exists():
        msg = f"Could not find data file at {path.resolve()}."
        raise FileNotFoundError(msg)

    raw_content = path.read_text(encoding="utf-8")
    data = json.loads(raw_content)
    if not isinstance(data, list):
        msg = "JSON data must be a list of paper records."
        raise ValueError(msg)

    papers: list[Paper] = []
    for record in cast(list[Any], data):
        if not isinstance(record, dict):
            msg = "Each paper record must be a JSON object."
            raise ValueError(msg)
        try:
            papers.append(Paper(**record)) # type: ignore
        except TypeError as exc:
            msg = f"Invalid paper record: {record}"
            raise ValueError(msg) from exc
    return papers


def paper_sort_key(paper: Paper) -> tuple[int, str]:
    return (-len(paper.body), paper.title.lower())
    # try:
    #     year_value = int(paper.year)
    # except ValueError:
    #     year_value = -1_000_000_000
    # return (-year_value, paper.title.lower())


def filter_papers(
    papers: list[Paper],
    selected_tag: str | None,
    query: str | None,
) -> list[Paper]:
    active_tag = (selected_tag or ALL_TAG_VALUE).lower()
    normalized_query = (query or "").strip()
    match_all_text = normalized_query in {"", "*"}
    normalized_query_lower = normalized_query.lower()

    filtered: list[Paper] = []
    for paper in papers:
        paper_tags_lower = {tag.lower() for tag in paper.tags}
        if active_tag != ALL_TAG_VALUE and active_tag not in paper_tags_lower:
            continue

        if match_all_text:
            filtered.append(paper)
            continue

        haystacks: tuple[str, ...] = (
            paper.title,
            paper.authors,
            paper.body,
            paper.abbrev,
            paper.year,
        )
        if any(normalized_query_lower in haystack.lower() for haystack in haystacks):
            filtered.append(paper)

    return sorted(filtered, key=paper_sort_key)


def build_paper_button(paper: Paper) -> html.Button:
    return html.Button(
        [
            html.Div(paper.title, style={"fontWeight": 600, "marginBottom": "0.25rem"}),
            html.Div(
                f"{paper.authors} · {paper.year}",
                style={"fontSize": "0.85rem", "color": "#555", "marginBottom": "0.35rem"},
            ),
            html.Div(
                ", ".join(paper.tags),
                style={"fontSize": "0.75rem", "color": "#777"},
            ),
        ],
        id={"type": "paper-button", "key": paper.title},
        n_clicks=0,
        style={
            "width": "100%",
            "border": "1px solid #d0d0d0",
            "borderRadius": "0.5rem",
            "backgroundColor": "#f8f9fa",
            "padding": "0.75rem",
            "textAlign": "left",
            "marginBottom": "0.5rem",
            "cursor": "pointer",
        },
    )


def default_detail_view() -> html.Div:
    return html.Div(
        [
            html.H3("Paper summary"),
            html.P("Select a paper from the list to read its review summary."),
        ],
        style={"padding": "1rem", "color": "#555"},
    )


def build_paper_detail(record: dict[str, Any]) -> html.Div:
    body_text = str(record.get("body", "")).strip()
    paragraphs = [
        html.P(paragraph.strip())
        for paragraph in body_text.split("\n")
        if paragraph.strip()
    ]

    tags = record.get("tags", [])
    tag_text = ", ".join(tags) if tags else "None"

    return html.Div(
        [
            html.H2(record.get("title", "Untitled"), style={"marginBottom": "0.5rem"}),
            html.Div(
                [
                    html.Span(record.get("authors", ""), style={"fontWeight": 500}),
                    html.Span(
                        f" · {record.get('year', '')}",
                        style={"marginLeft": "0.35rem"},
                    ),
                ],
                style={"marginBottom": "0.75rem", "color": "#444"},
            ),
            html.Div(
                html.A(
                    "View paper",
                    href=record.get("url", "#"),
                    target="_blank",
                    rel="noopener noreferrer",
                ),
                style={"marginBottom": "1rem"},
            ),
            html.Div(paragraphs, style={"lineHeight": 1.6}),
            html.Div(
                [
                    html.Strong("Tags:"),
                    html.Span(f" {tag_text}", style={"color": "#555"}),
                ],
                style={"marginTop": "1.5rem"},
            ),
        ],
        style={"padding": "1rem"},
    )


papers = load_papers(DATA_PATH)
UNIQUE_TAGS = sorted({tag for paper in papers for tag in paper.tags})
TAG_DROPDOWN_OPTIONS: list[dict[str, str]] = [{"label": "All tags", "value": ALL_TAG_VALUE}]
TAG_DROPDOWN_OPTIONS.extend(
    {"label": tag, "value": tag} for tag in UNIQUE_TAGS
)

app = Dash(__name__)
server = app.server
app.title = "Paper Review Explorer"

DashChildren = Component | list[Component]

app.layout = html.Div(
    [
        dcc.Store(id="filtered-papers", data=[asdict(paper) for paper in papers]),
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Paper Review Explorer", style={"marginBottom": "1rem"}),
                        html.Label(
                            "Tag filter",
                            htmlFor="tag-dropdown",
                            style={"fontWeight": 600},
                        ),
                        dcc.Dropdown(
                            id="tag-dropdown",
                            options=TAG_DROPDOWN_OPTIONS, # type: ignore
                            value=ALL_TAG_VALUE,
                            clearable=False,
                            style={"marginBottom": "1rem"},
                        ),
                        html.Label(
                            "Search query",
                            htmlFor="search-text",
                            style={"fontWeight": 600},
                        ),
                        dcc.Input(
                            id="search-text",
                            type="text",
                            value=DEFAULT_SEARCH_TEXT,
                            placeholder="Enter search text or * for all",
                            style={
                                "width": "100%",
                                "padding": "0.5rem",
                                "marginTop": "0.25rem",
                                "marginBottom": "0.75rem",
                            },
                        ),
                        html.Button(
                            "Search",
                            id="search-button",
                            n_clicks=0,
                            style={
                                "padding": "0.5rem 1rem",
                                "backgroundColor": "#1976d2",
                                "color": "#fff",
                                "border": "none",
                                "borderRadius": "0.4rem",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(
                            id="search-count",
                            style={"marginTop": "1rem", "color": "#444"},
                        ),
                        html.Div(
                            id="paper-list",
                            style={"marginTop": "1rem"},
                        ),
                    ],
                    style={
                        "flex": "0 0 38%",
                        "maxWidth": "32rem",
                        "padding": "1.5rem",
                        "borderRight": "1px solid #e0e0e0",
                        "overflowY": "auto",
                    },
                ),
                html.Div(
                    default_detail_view(),
                    id="paper-detail",
                    style={
                        "flex": "1",
                        "padding": "1.5rem",
                        "overflowY": "auto",
                    },
                ),
            ],
            style={
                "display": "flex",
                "height": "100vh",
                "fontFamily": "Arial, sans-serif",
                "backgroundColor": "#ffffff",
            },
        ),
    ]
)


@app.callback( # type: ignore
    Output("paper-list", "children"),
    Output("search-count", "children"),
    Output("filtered-papers", "data"),
    Input("search-button", "n_clicks"),
    State("tag-dropdown", "value"),
    State("search-text", "value"),
)
def update_search_results(
    n_clicks: int | None,
    selected_tag: str | None,
    query_text: str | None,
) -> tuple[DashChildren, str, list[dict[str, Any]]]:
    filtered = filter_papers(papers, selected_tag, query_text)
    count_message = f"Found {len(filtered)} paper{'s' if len(filtered) != 1 else ''}."

    if filtered:
        paper_components: list[Component] = [build_paper_button(paper) for paper in filtered]
        payload = [asdict(paper) for paper in filtered]
        return paper_components, count_message, payload

    empty_state = html.Div(
        "No papers match your filters. Adjust the tag or search query and try again.",
        style={"marginTop": "1rem", "color": "#777"},
    )
    return empty_state, count_message, []


@app.callback( # type: ignore
    Output("paper-detail", "children"),
    Input({"type": "paper-button", "key": ALL}, "n_clicks"),
    Input("filtered-papers", "data"),
    prevent_initial_call=True,
)
def display_paper_detail(
    paper_clicks: list[int],
    filtered_payloads: list[dict[str, Any]] | None,
) -> Component:
    triggered = ctx.triggered_id # type: ignore

    if triggered == "filtered-papers":
        return default_detail_view()

    if not isinstance(triggered, dict):
        raise PreventUpdate

    if not isinstance(filtered_payloads, list):
        raise PreventUpdate

    selected_key = triggered.get("key") # type: ignore
    for record in filtered_payloads:
        if record.get("title") == selected_key:
            return build_paper_detail(record)

    raise PreventUpdate


def main() -> None:
    app.run(debug=True) # type: ignore


if __name__ == "__main__":
    main()
