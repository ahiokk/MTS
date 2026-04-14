from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.llm import (
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    MAX_REPAIR_ATTEMPTS,
    TaskAnalysis,
    analyze_task,
    run_agent_flow,
)
from app.validator import OUTPUT_FORMATS, validate_lua

OutputFormat = Literal[
    "auto",
    "raw_lua",
    "lowcode_lua_fragment",
    "json_with_lua_fields",
]

GenerateStatus = Literal["completed", "needs_clarification"]

app = FastAPI(
    title="LocalScript API",
    version="1.1.0",
    description="Local AI agent for generating and validating Lua code.",
)


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., description="Task in natural language.")
    context: Any | None = Field(
        default=None,
        description="Optional structured context passed separately from the prompt.",
    )
    existing_code: str | None = Field(
        default=None,
        description="Optional existing Lua code to modify instead of generating from scratch.",
    )
    feedback: str | None = Field(
        default=None,
        description="Optional user feedback for the next iteration or refinement pass.",
    )
    target_output_format: OutputFormat = Field(
        default="auto",
        description="Expected output format for the final artifact.",
    )


class AnalysisResponse(BaseModel):
    mode: Literal["generate", "modify"]
    resolved_output_format: OutputFormat
    needs_clarification: bool
    clarification_question: str | None = None
    referenced_paths: list[str]
    context_attached: bool
    existing_code_attached: bool
    feedback_attached: bool


class GenerateRequest(AnalyzeRequest):
    allow_clarification: bool = Field(
        default=True,
        description="Whether the agent may ask a clarification question instead of generating code immediately.",
    )
    max_repair_attempts: int = Field(
        default=MAX_REPAIR_ATTEMPTS,
        ge=0,
        le=5,
        description="Maximum repair iterations after validation failures.",
    )


class GenerateResponse(BaseModel):
    status: GenerateStatus
    code: str | None = None
    clarification_question: str | None = None
    analysis: AnalysisResponse


class ValidateRequest(BaseModel):
    prompt: str = Field(..., description="Original user prompt.")
    output: str = Field(..., description="Generated code or artifact to validate.")
    expected_output_format: OutputFormat = Field(default="auto")


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_host: str
    luac_supported: bool
    supported_output_formats: list[str]
    default_num_ctx: int
    default_num_predict: int
    default_temperature: float
    default_seed: int
    max_repair_attempts: int


def _analysis_to_response(analysis: TaskAnalysis) -> AnalysisResponse:
    return AnalysisResponse(
        mode=analysis.mode,
        resolved_output_format=analysis.resolved_output_format,  # type: ignore[arg-type]
        needs_clarification=analysis.needs_clarification,
        clarification_question=analysis.clarification_question,
        referenced_paths=analysis.referenced_paths,
        context_attached=analysis.context_attached,
        existing_code_attached=analysis.existing_code_attached,
        feedback_attached=analysis.feedback_attached,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    syntax_probe = validate_lua(
        user_text="return a constant value",
        output="return 1",
        expected_output_format="raw_lua",
    )
    return HealthResponse(
        status="ok",
        model=DEFAULT_MODEL,
        ollama_host=DEFAULT_OLLAMA_HOST,
        luac_supported=bool(syntax_probe["syntax"]["supported"]),
        supported_output_formats=sorted(OUTPUT_FORMATS),
        default_num_ctx=DEFAULT_NUM_CTX,
        default_num_predict=DEFAULT_NUM_PREDICT,
        default_temperature=DEFAULT_TEMPERATURE,
        default_seed=DEFAULT_SEED,
        max_repair_attempts=MAX_REPAIR_ATTEMPTS,
    )


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest) -> AnalysisResponse:
    analysis = analyze_task(
        user_text=request.prompt,
        context=request.context,
        existing_code=request.existing_code,
        feedback=request.feedback,
        target_output_format=request.target_output_format,
    )
    return _analysis_to_response(analysis)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    result = run_agent_flow(
        user_text=request.prompt,
        model=DEFAULT_MODEL,
        context=request.context,
        existing_code=request.existing_code,
        feedback=request.feedback,
        target_output_format=request.target_output_format,
        max_repair_attempts=request.max_repair_attempts,
        allow_clarification=request.allow_clarification,
    )
    return GenerateResponse(
        status=result.status,
        code=result.code,
        clarification_question=result.clarification_question,
        analysis=_analysis_to_response(result.analysis),
    )


@app.post("/validate")
def validate(request: ValidateRequest) -> dict:
    return validate_lua(
        user_text=request.prompt,
        output=request.output,
        expected_output_format=request.expected_output_format,
    )
