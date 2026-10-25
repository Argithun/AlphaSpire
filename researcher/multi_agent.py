import ast
import json
import logging
import math
import re
from random import SystemRandom
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import pstdev
from typing import Callable, List, Literal

import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts" / "multi_agent_research.yaml"
SCORE_WEIGHTS = {
    "Evidence": 0.25,
    "EconomicLogic": 0.25,
    "Feasibility": 0.25,
    "Robustness": 0.15,
    "Novelty": 0.10,
}
logger = logging.getLogger(__name__)


class ResearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    discussion_rounds: int = Field(default=1, ge=1, le=3, strict=True)
    max_workers: int = Field(default=3, ge=1, le=6, strict=True)
    max_retries: int = Field(default=2, ge=0, le=5, strict=True)
    request_timeout: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(default=6000, ge=512, strict=True)
    temperature: float = Field(default=0.2, ge=0, le=2, allow_inf_nan=False)


class AgentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1)
    specialty: str = Field(min_length=1)


class ResearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Hypothesis(ResearchOutput):
    Hypothesis: str = Field(min_length=1)
    Rationale: str = Field(min_length=1)
    Evidence: str = Field(min_length=1)
    PotentialFields: List[str] = Field(min_length=1)
    PotentialOperators: List[str] = Field(min_length=1)
    Risks: str = Field(min_length=1)


class HypothesisOutput(ResearchOutput):
    Hypotheses: List[Hypothesis] = Field(min_length=2, max_length=3)


class TemplateOutput(ResearchOutput):
    SelectedHypothesis: str = Field(min_length=1)
    TemplateExpression: str = Field(min_length=1)
    Description: str = Field(min_length=1)
    ExpectedBehavior: str = Field(min_length=1)
    RevisionNotes: str = Field(min_length=1)


class PeerReview(ResearchOutput):
    CandidateId: str = Field(min_length=1)
    Strengths: str = Field(min_length=1)
    Weaknesses: str = Field(min_length=1)
    Suggestions: str = Field(min_length=1)


class DiscussionOutput(ResearchOutput):
    Reviews: List[PeerReview] = Field(min_length=1)


class Scores(ResearchOutput):
    model_config = ConfigDict(extra="forbid")
    Evidence: float = Field(ge=0, le=10, allow_inf_nan=False, strict=True)
    EconomicLogic: float = Field(ge=0, le=10, allow_inf_nan=False, strict=True)
    Feasibility: float = Field(ge=0, le=10, allow_inf_nan=False, strict=True)
    Robustness: float = Field(ge=0, le=10, allow_inf_nan=False, strict=True)
    Novelty: float = Field(ge=0, le=10, allow_inf_nan=False, strict=True)


def weighted_score(scores: dict) -> float:
    return sum(scores[key] * weight for key, weight in SCORE_WEIGHTS.items())


class Evaluation(ResearchOutput):
    CandidateId: str = Field(min_length=1)
    Scores: Scores
    Reason: str = Field(min_length=1)
    MainRisk: str = Field(min_length=1)


class BallotOutput(ResearchOutput):
    Evaluations: List[Evaluation] = Field(min_length=1)
    PreferredCandidate: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_preference(self):
        scores = {
            item.CandidateId: weighted_score(item.Scores.model_dump())
            for item in self.Evaluations
        }
        if self.PreferredCandidate not in scores:
            raise ValueError("PreferredCandidate must be evaluated on this ballot")
        if not math.isclose(scores[self.PreferredCandidate], max(scores.values())):
            raise ValueError("PreferredCandidate must have the highest weighted score")
        return self


class ResearchState(TypedDict, total=False):
    post: dict
    hypotheses: dict
    candidates: dict
    discussion: list
    round: int
    ballots: dict
    ranking: list
    winner: dict


class TemplateValidator:
    def __init__(self, field_types: dict, operator_types: dict,
                 field_ids: set, operator_names: set):
        self.field_types = field_types
        self.operator_types = operator_types
        self.symbols = field_ids | operator_names | {"true", "false"}

    def validate(self, template: dict):
        expression = template["TemplateExpression"]
        placeholders = re.findall(r"</(.*?)/>", expression)
        if not 1 <= len(placeholders) <= 4:
            raise ValueError("Template must contain 1-4 placeholder occurrences")
        for placeholder in placeholders:
            if not (self.field_types.get(placeholder) or self.operator_types.get(placeholder)):
                raise ValueError(f"Unknown or empty placeholder: {placeholder}")
        remainder = re.sub(r"</.*?/>", "0", expression)
        if any(token in remainder for token in ("</", "/>", "```", ";", "//", "/*", "#")):
            raise ValueError("Use a single expression without comments or malformed placeholders")
        if re.search(r"\b\d+(?:\.\d*)?[eE][+-]?\d+\b", remainder):
            raise ValueError("Scientific notation is not supported")
        parse_expression = re.sub(r"\b(and|or|not)\s*(?=\()", r"wq_\1", remainder)
        try:
            ast.parse(parse_expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Template must be a syntactically well-formed single expression") from exc
        remainder = re.sub(r"\"[^\"\n]*\"|'[^'\n]*'", "", remainder)
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z_0-9]*\b", remainder):
            symbol = match.group()
            if symbol not in self.symbols and not re.match(r"\s*=(?!=)", remainder[match.end():]):
                raise ValueError(f"Unknown field or operator: {symbol}")


def validate_peer_coverage(items: list, expected_ids: set):
    actual_ids = [item["CandidateId"] for item in items]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise ValueError("Evaluate every peer candidate exactly once; self-votes are forbidden")


def render_prompt(template: str, context: dict) -> str:
    def replace(match):
        value = context[match.group(1)]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, template)


class SpecialistAgent:
    def __init__(self, profile: AgentProfile, knowledge: str, prompts: dict,
                 settings: ResearchSettings, model):
        self.profile = profile
        self.prompts = prompts
        self.settings = settings
        system = render_prompt(prompts["system"], {
            "name": profile.name, "specialty": profile.specialty, "knowledge": knowledge
        })
        self.chain = ChatPromptTemplate.from_messages([
            ("system", "{system}"), ("human", "{task}")
        ]) | model | JsonOutputParser()
        self.system = system

    def run(self, phase: str, context: dict, schema: type[BaseModel],
            validator: Callable = None) -> dict:
        task = render_prompt(self.prompts[phase], context)
        task += "\nRequired JSON schema:\n" + json.dumps(schema.model_json_schema())
        error = ""
        for attempt in range(self.settings.max_retries + 1):
            try:
                result = self.chain.invoke({"system": self.system, "task": task + error})
                result = schema.model_validate(result).model_dump()
                if validator:
                    validator(result)
                return result
            except Exception as exc:
                logger.warning("Agent %s phase %s attempt %s failed: %s",
                               self.profile.id, phase, attempt + 1, exc)
                error = f"\nYour previous attempt was invalid: {str(exc)[:2000]}\nReturn corrected JSON."
        raise RuntimeError(f"Agent {self.profile.id} failed during {phase}; research aborted")


class ResearchCommittee:
    def __init__(self, knowledge: str, template_validator: TemplateValidator,
                 settings: ResearchSettings, model=None, prompt_file: Path = PROMPT_FILE):
        with open(prompt_file, encoding="utf-8") as handle:
            self.prompts = yaml.safe_load(handle)
        profiles = [AgentProfile.model_validate(item) for item in self.prompts["agents"]]
        if len(profiles) < 5 or len({profile.id for profile in profiles}) != len(profiles):
            raise ValueError("Research requires at least five specialists with unique IDs")
        self.settings = settings
        self.template_validator = template_validator
        if model is None:
            from utils.config_loader import ConfigLoader

            model = ChatOpenAI(
                base_url=ConfigLoader.get("openai_base_url"),
                api_key=ConfigLoader.get("openai_api_key"),
                model=ConfigLoader.get("openai_model_name"),
                temperature=settings.temperature,
                timeout=settings.request_timeout,
                max_tokens=settings.max_tokens,
                max_retries=0,
            )
        self.agents = [SpecialistAgent(profile, knowledge, self.prompts, settings, model)
                       for profile in profiles]
        identifiers = [f"candidate_{index:02d}" for index in range(1, len(self.agents) + 1)]
        SystemRandom().shuffle(identifiers)
        self.candidate_ids = {agent.profile.id: candidate_id
                              for agent, candidate_id in zip(self.agents, identifiers)}
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        for name, node in (("extract_hypotheses", self._hypotheses), ("proposals", self._proposals),
                           ("peer_discussion", self._discussion), ("revision", self._revision),
                           ("vote", self._vote), ("select", self._select)):
            graph.add_node(name, node)
        graph.add_edge(START, "extract_hypotheses")
        graph.add_edge("extract_hypotheses", "proposals")
        graph.add_edge("proposals", "peer_discussion")
        graph.add_edge("peer_discussion", "revision")
        graph.add_conditional_edges("revision", self._next_phase,
                                    {"discussion": "peer_discussion", "vote": "vote"})
        graph.add_edge("vote", "select")
        graph.add_edge("select", END)
        return graph.compile()

    def _next_phase(self, state: ResearchState) -> Literal["discussion", "vote"]:
        return "discussion" if state["round"] < self.settings.discussion_rounds else "vote"

    def _parallel(self, operation: Callable) -> dict:
        with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
            outputs = list(executor.map(operation, self.agents))
        return {agent.profile.id: output for agent, output in zip(self.agents, outputs)}

    def _context(self, state: ResearchState) -> dict:
        return {"post": state["post"], "hypotheses": state.get("hypotheses", {}),
                "template_rules": self.prompts["template_rules"]}

    def _hypotheses(self, state: ResearchState) -> dict:
        logger.info("Research committee: independent hypotheses")

        def validate(result):
            for hypothesis in result["Hypotheses"]:
                for field_type in hypothesis["PotentialFields"]:
                    if not self.template_validator.field_types.get(re.sub(r"^</|/>$", "", field_type)):
                        raise ValueError(f"Unknown hypothesis field type: {field_type}")
                for operator_type in hypothesis["PotentialOperators"]:
                    if not self.template_validator.operator_types.get(re.sub(r"^</|/>$", "", operator_type)):
                        raise ValueError(f"Unknown hypothesis operator type: {operator_type}")

        outputs = self._parallel(lambda agent: agent.run(
            "hypotheses", {"post": state["post"]}, HypothesisOutput, validate))
        return {"hypotheses": outputs, "discussion": [], "round": 0}

    def _proposals(self, state: ResearchState) -> dict:
        logger.info("Research committee: candidate proposals")
        outputs = self._parallel(lambda agent: agent.run(
            "proposal", self._context(state), TemplateOutput, self.template_validator.validate))
        candidates = {self.candidate_ids[agent_id]: output for agent_id, output in outputs.items()}
        return {"candidates": dict(sorted(candidates.items()))}

    def _peers(self, state: ResearchState, agent: SpecialistAgent) -> dict:
        own_id = self.candidate_ids[agent.profile.id]
        peers = [(candidate_id, candidate) for candidate_id, candidate in state["candidates"].items()
                 if candidate_id != own_id]
        SystemRandom().shuffle(peers)
        return dict(peers)

    def _discussion(self, state: ResearchState) -> dict:
        round_number = state["round"] + 1
        logger.info("Research committee: discussion round %s", round_number)

        def review(agent):
            peers = self._peers(state, agent)
            context = {**self._context(state), "round": str(round_number), "candidates": peers}
            return agent.run("discussion", context, DiscussionOutput,
                             lambda result: validate_peer_coverage(result["Reviews"], set(peers)))

        reviews = self._parallel(review)
        return {"discussion": state["discussion"] + [{
            "Round": round_number, "CandidatesBefore": state["candidates"], "Reviews": reviews
        }], "round": round_number}

    def _revision(self, state: ResearchState) -> dict:
        logger.info("Research committee: simultaneous revisions")

        def revise(agent):
            own_id = self.candidate_ids[agent.profile.id]
            feedback = [{"Reviewer": reviewer, **review}
                        for reviewer, output in state["discussion"][-1]["Reviews"].items()
                        for review in output["Reviews"] if review["CandidateId"] == own_id]
            context = {**self._context(state), "round": str(state["round"]),
                       "own_candidate": state["candidates"][own_id],
                       "candidates": state["candidates"], "feedback": feedback}
            return agent.run("revision", context, TemplateOutput, self.template_validator.validate)

        outputs = self._parallel(revise)
        candidates = dict(sorted((self.candidate_ids[agent_id], output)
                                 for agent_id, output in outputs.items()))
        discussion = state["discussion"][:-1] + [
            {**state["discussion"][-1], "CandidatesAfter": candidates}]
        return {"candidates": candidates, "discussion": discussion}

    def _vote(self, state: ResearchState) -> dict:
        logger.info("Research committee: independent peer ballots")

        def vote(agent):
            peers = self._peers(state, agent)
            anonymous_discussion = []
            for item in state["discussion"]:
                reviews = [review for output in item["Reviews"].values()
                           for review in output["Reviews"] if review["CandidateId"] in peers]
                SystemRandom().shuffle(reviews)
                anonymous_discussion.append({"Round": item["Round"], "Reviews": reviews})
            context = {**self._context(state), "candidates": peers,
                       "discussion": anonymous_discussion}
            return agent.run("vote", context, BallotOutput,
                             lambda result: validate_peer_coverage(result["Evaluations"], set(peers)))

        return {"ballots": self._parallel(vote)}

    def _select(self, state: ResearchState) -> dict:
        ranking = []
        for candidate_id in state["candidates"]:
            evaluations = [evaluation for ballot in state["ballots"].values()
                           for evaluation in ballot["Evaluations"]
                           if evaluation["CandidateId"] == candidate_id]
            scores = [weighted_score(item["Scores"]) for item in evaluations]
            if len(scores) != len(self.agents) - 1:
                raise ValueError("Each candidate must receive one ballot from every other agent")
            ranking.append({
                "CandidateId": candidate_id,
                "MeanScore": sum(scores) / len(scores),
                "FirstChoiceVotes": sum(ballot["PreferredCandidate"] == candidate_id
                                        for ballot in state["ballots"].values()),
                "ScoreStdDev": pstdev(scores),
                "PeerEvaluations": len(scores),
                "DimensionMeans": {key: sum(item["Scores"][key] for item in evaluations) / len(scores)
                                   for key in SCORE_WEIGHTS},
            })
        ranking.sort(key=lambda item: (-round(item["MeanScore"], 10),
                                       -item["FirstChoiceVotes"],
                                       round(item["ScoreStdDev"], 10), item["CandidateId"]))
        winner_id = ranking[0]["CandidateId"]
        winner = {**state["candidates"][winner_id], "CandidateId": winner_id}
        logger.info("Research committee selected %s (score %.3f)", winner_id, ranking[0]["MeanScore"])
        return {"ranking": ranking, "winner": winner}

    def run(self, post: dict) -> dict:
        return self.graph.invoke({"post": post}, {"recursion_limit": 20})
