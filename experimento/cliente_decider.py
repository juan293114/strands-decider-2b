"""Cliente de ejemplo para consumir Strands Decider a través de LiteLLM.

    pip install openai pydantic
    export LITELLM_BASE_URL=http://loadbalancer-litellm-ext-1322139303.us-east-1.elb.amazonaws.com
    export LITELLM_API_KEY=<llave virtual>
    python -m experimento.cliente_decider    (desde la carpeta strands-decider-2b)
"""
import json
import os
from typing import Annotated, Literal, Union

from openai import OpenAI
from pydantic import BaseModel, Field


# ---- Preguntas (entrada) -------------------------------------------------------------
class Noul(BaseModel):
    """Probabilidad de que algo sea cierto (sí/no)."""
    type: Literal["noul"] = "noul"
    instructions: str


class Choice(BaseModel):
    """Una opción entre N (2-255). criteria: opción -> descripción (o None)."""
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str | None]


class Score(BaseModel):
    """Nivel en una escala ordenada ascendente (2-10 niveles)."""
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


Question = Union[Noul, Choice, Score]


class DeciderRequest(BaseModel):
    state: str  # el texto/contexto que se evalúa
    questions: dict[str, Question]


# ---- Respuestas (salida) -------------------------------------------------------------
class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float  # esperanza sobre los niveles; 0 = primer nivel de `criteria`
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class DeciderResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    latency_ms: float


# ---- Cliente -------------------------------------------------------------------------
class DeciderClient:
    def __init__(self, base_url: str, api_key: str, model: str = "strands-decider"):
        # timeout alto: la 1ª petición tras desplegar el endpoint tarda ~42 s
        self._c = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=api_key, timeout=120)
        self._model = model

    def decide(self, req: DeciderRequest) -> DeciderResponse:
        r = self._c.chat.completions.create(
            model=self._model,
            # El decider NO es un LLM: el contenido del mensaje user es el JSON de la petición
            messages=[{"role": "user", "content": req.model_dump_json()}],
        )
        # y la respuesta del decider llega como JSON dentro de message.content
        return DeciderResponse.model_validate(json.loads(r.choices[0].message.content))


if __name__ == "__main__":
    client = DeciderClient(os.environ["LITELLM_BASE_URL"], os.environ["LITELLM_API_KEY"])
    out = client.decide(DeciderRequest(
        state="Help! My payouts have been failing for 3 days!",
        questions={
            "is_urgent": Noul(instructions="Does this convey urgency?"),
            "team": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": None, "sales": None, "retail": None},
            ),
            "frustration": Score(
                instructions="How frustrated is the writer?",
                criteria=["calm", "frustrated", "depressed"],
            ),
        },
    ))
    print("urgente:", out.answers["is_urgent"].noul)
    print("equipo:", out.answers["team"].choice, out.answers["team"].confidence)
    print("frustración:", out.answers["frustration"].score)
    print("latencia_ms:", out.latency_ms)
