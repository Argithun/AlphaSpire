import csv
import json
import random
from pathlib import Path

from researcher.construct_prompts import build_wq_knowledge_prompt
from researcher.multi_agent import ResearchCommittee, ResearchSettings, SCORE_WEIGHTS, TemplateValidator
from utils.config_loader import ConfigLoader

BASE_DIR = Path(__file__).resolve().parents[1]
POSTS_DIR = BASE_DIR / "data" / "wq_posts" / "helpful_posts"
HYPOTHESIS_DB = BASE_DIR / "data" / "hypothesis_db_v2"
TEMPLATE_DB = BASE_DIR / "data" / "template_db_v2"
RESEARCH_DB = BASE_DIR / "data" / "research_db_v2"


def load_template_validator() -> TemplateValidator:
    enabled = ConfigLoader.get("enabled_field_datasets", [])
    field_ids = set()
    for field_file in (BASE_DIR / "data" / "wq_fields").glob("*.csv"):
        if enabled and field_file.stem not in enabled:
            continue
        with open(field_file, encoding="utf-8") as handle:
            field_ids.update(row["id"] for row in csv.DictReader(handle))
    with open(BASE_DIR / "data" / "wq_template_fields" / "template_fields.json",
              encoding="utf-8") as handle:
        field_types = {}
        for raw_type, field_list in json.load(handle).items():
            clean_type = raw_type.removeprefix("</").removesuffix("/>")
            if not enabled or clean_type.rsplit(":", 1)[-1] in enabled:
                field_types[clean_type] = [field_id for field_id in field_list if field_id in field_ids]
    operator_types = {}
    operator_names = set()
    with open(BASE_DIR / "data" / "wq_template_operators" / "template_operators.csv",
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            operator_names.add(row["name"])
            operator_types.setdefault(row["type"], []).append(row["name"])
    return TemplateValidator(field_types, operator_types, field_ids, operator_names)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def from_post_to_template(post_file: str = None, force: bool = False):
    """Run the specialist committee and return a backward-compatible winning template."""
    if post_file is None:
        posts = list(POSTS_DIR.glob("*.json"))
        if not posts:
            raise FileNotFoundError(f"No helpful posts found in {POSTS_DIR}")
        post_file = random.choice(posts)
    post_path = Path(post_file)
    template_file = TEMPLATE_DB / f"{post_path.stem}_hypotheses_template.json"
    if template_file.exists() and not force:
        print(f"✅ Template already exists: {template_file}")
        return None

    with open(post_path, encoding="utf-8") as handle:
        post = json.load(handle)
    if not isinstance(post, dict) or not any(post.get(key) for key in ("title", "post_body", "post_comments")):
        raise ValueError("Post must contain a title, body or comments")
    settings = ResearchSettings.model_validate(ConfigLoader.get("research", {}))
    committee = ResearchCommittee(build_wq_knowledge_prompt(), load_template_validator(), settings)
    result = committee.run(post)

    hypotheses_file = HYPOTHESIS_DB / f"{post_path.stem}_hypotheses.json"
    audit_file = RESEARCH_DB / f"{post_path.stem}_research.json"
    hypotheses = []
    for agent_id, output in result["hypotheses"].items():
        for hypothesis in output["Hypotheses"]:
            hypotheses.append({
                "AgentId": agent_id,
                "Hypothesis": hypothesis["Hypothesis"],
                "Rationale": hypothesis["Rationale"],
                "Evidence": hypothesis["Evidence"],
                "Potential Fields": hypothesis["PotentialFields"],
                "Potential Operators": hypothesis["PotentialOperators"],
                "Risks": hypothesis["Risks"],
            })
    save_json(hypotheses_file, hypotheses)
    save_json(audit_file, {
        "WorkflowVersion": "multi_agent_v1",
        "SourcePost": str(post_path.resolve()),
        "Settings": settings.model_dump(),
        "ModelName": ConfigLoader.get("openai_model_name"),
        "ScoreWeights": SCORE_WEIGHTS,
        "Agents": [agent.profile.model_dump() for agent in committee.agents],
        "CandidateAuthors": {candidate_id: agent_id
                             for agent_id, candidate_id in committee.candidate_ids.items()},
        **result,
    })
    winner = result["winner"]
    save_json(template_file, {
        **{key: winner[key] for key in ("SelectedHypothesis", "TemplateExpression",
                                      "Description", "ExpectedBehavior")},
        "ResearchMetadata": {
            "WorkflowVersion": "multi_agent_v1",
            "CandidateId": winner["CandidateId"],
            "Ranking": result["ranking"],
            "ScoreMeaning": "Qualitative peer research scores, not backtest performance",
            "AuditFile": str(audit_file.relative_to(BASE_DIR)),
        },
    })
    print(f"🎯 Committee selected {winner['CandidateId']}: {template_file}")
    return template_file


if __name__ == "__main__":
    from_post_to_template()
