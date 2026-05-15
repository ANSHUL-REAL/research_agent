from __future__ import annotations

from enum import StrEnum

from langchain_core.prompts import PromptTemplate


class Mode(StrEnum):
    GENERAL_RESEARCH = "General Research"
    ROADMAP_VALIDATOR = "Roadmap Validator"
    PROJECT_IDEA_GENERATOR = "Project Idea Generator"
    LEARNING_PLAN_GENERATOR = "Learning Plan Generator"


GENERAL_RESEARCH_PROMPT = """You are an AI research assistant.

Answer the user's question using only the provided context.
Include citations using the source title and URL.
If the context is insufficient, say that the available sources are not enough.
Format the answer as clean Markdown with short paragraphs, ### section headings, and bullet lists where useful.
Do not add a final "Sources" section because the app displays citations separately.

User question:
{question}

Retrieved context:
{context}
"""

ROADMAP_VALIDATOR_PROMPT = """You are an AI career roadmap evaluator.

Analyze the user's roadmap using the retrieved sources.
Check for:
1. Missing topics
2. Incorrect order
3. Outdated tools
4. Beginner-friendliness
5. Practical project coverage

Return:
- Overall rating out of 10
- What is good
- What is missing
- Corrected roadmap
- Estimated timeline
- Recommended resources with citations
Format the answer as clean Markdown with ### section headings.
Do not add a final "Sources" section because the app displays citations separately.

User roadmap:
{question}

Retrieved context:
{context}
"""

PROJECT_IDEA_GENERATOR_PROMPT = """You are an AI project mentor.

Generate practical project ideas based on the user's skills and goals.
Use the retrieved sources for accuracy.
Return at least 3 project ideas.

For each project, include:
- Title
- Problem statement
- Tech stack
- Difficulty
- Key features
- APIs/datasets required
- Implementation steps
- Learning outcomes
- Possible future improvements
Format the answer as clean Markdown with ### section headings and clear bullets.
Do not add a final "Sources" section because the app displays citations separately.

User request:
{question}

Retrieved context:
{context}
"""

LEARNING_PLAN_PROMPT = """You are an AI learning planner.

Create a structured learning plan based on the user request and retrieved sources.

Include:
- Timeline
- Daily/weekly tasks
- Topics
- Resources
- Mini-projects
- Final outcome
- Tips for revision
Format the answer as clean Markdown with ### section headings and clear bullets.
Do not add a final "Sources" section because the app displays citations separately.

User request:
{question}

Retrieved context:
{context}
"""

PROMPTS = {
    Mode.GENERAL_RESEARCH: GENERAL_RESEARCH_PROMPT,
    Mode.ROADMAP_VALIDATOR: ROADMAP_VALIDATOR_PROMPT,
    Mode.PROJECT_IDEA_GENERATOR: PROJECT_IDEA_GENERATOR_PROMPT,
    Mode.LEARNING_PLAN_GENERATOR: LEARNING_PLAN_PROMPT,
}


def coerce_mode(mode: str | Mode) -> Mode:
    if isinstance(mode, Mode):
        return mode
    try:
        return Mode(mode)
    except ValueError:
        return Mode.GENERAL_RESEARCH


def get_prompt_template(mode: str | Mode) -> PromptTemplate:
    selected_mode = coerce_mode(mode)
    return PromptTemplate.from_template(PROMPTS[selected_mode])
