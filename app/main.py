from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.llm import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST, MAX_REPAIR_ATTEMPTS, run_agent
from app.validator import OUTPUT_FORMATS, validate_lua

OutputFormat = Literal[
    "auto",
    "raw_lua",
    "lowcode_lua_fragment",
    "json_with_lua_fields",
]

app = FastAPI(
    title="LocalScript API",
    version="1.0.0",
    description="Local AI agent for generating and validating Lua code.",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Task in natural language.")
    target_output_format: OutputFormat = Field(
        default="auto",
        description="Expected output format for generated code.",
    )
    max_repair_attempts: int = Field(
        default=MAX_REPAIR_ATTEMPTS,
        ge=0,
        le=5,
        description="Maximum repair iterations after validation failures.",
    )


class GenerateResponse(BaseModel):
    code: str


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
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    code = run_agent(
        user_text=request.prompt,
        model=DEFAULT_MODEL,
        target_output_format=request.target_output_format,
        max_repair_attempts=request.max_repair_attempts,
    )
    return GenerateResponse(code=code)


@app.post("/validate")
def validate(request: ValidateRequest) -> dict:
    return validate_lua(
        user_text=request.prompt,
        output=request.output,
        expected_output_format=request.expected_output_format,
    )
