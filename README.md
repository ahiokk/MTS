# LocalScript

Локальная агентская система для генерации и валидации Lua-кода без внешних AI API.

## Что умеет

- принимает задачу на русском или английском через HTTP API;
- отдельно анализирует запрос и определяет режим `generate` или `modify`;
- умеет попросить уточнение, если не хватает контекста или текущего Lua-кода;
- генерирует Lua или LowCode JSON-артефакт через локальный `Ollama`;
- использует локальный retrieval по базе few-shot примеров для усиления генерации;
- валидирует ответ доменными правилами и прогоняет синтаксис через `luac`;
- делает repair loop по диагностике валидатора.

## Архитектура

Пайплайн запроса:

`analyze -> clarify_or_generate -> validate -> repair -> return`

Основные компоненты:

- `app/main.py` — FastAPI API;
- `app/llm.py` — policy/analyzer, генерация, clarify/modify flow, repair loop;
- `app/retrieval.py` — lightweight local retrieval и few-shot подбор примеров;
- `app/validator.py` — форматная, доменная и `luac`-валидация;
- `docker-compose.yml` — one-command запуск `app + ollama + model pull`;
- `data/few_shot_examples.json` — локальная база примеров для retrieval;
- `scripts/run_public_regression.py` — regression harness.

## Модель и параметры

Основная модель:

- `ollama pull qwen2.5-coder:7b`

В проекте используется этот же тег по умолчанию:

- `LOCALSCRIPT_MODEL=qwen2.5-coder:7b`

Зафиксированные параметры инференса:

- `num_ctx=4096`
- `num_predict=256`
- `temperature=0`
- `seed=42`
- `OLLAMA_NUM_PARALLEL=1`

Эти параметры задаются в `docker-compose.yml` и `app/llm.py`.

## Быстрый запуск

Требования:

- Docker Desktop или Docker Engine + Compose
- GPU для режима проверки организаторов через Ollama

Запуск одной командой:

```bash
docker compose up --build
```

Что происходит при первом старте:

1. поднимается `ollama`;
2. сервис `ollama-init` автоматически делает `ollama pull qwen2.5-coder:7b`;
3. после успешной загрузки модели стартует `app`.

Если не хочется держать терминал занятым:

```bash
docker compose up --build -d
```

## Проверка после запуска

Health:

```bash
curl http://localhost:8080/health
```

Анализ задачи:

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Из полученного списка email получи последний.",
    "context": {
      "wf": {
        "vars": {
          "emails": ["user1@example.com", "user2@example.com", "user3@example.com"]
        }
      }
    }
  }'
```

Генерация:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Из полученного списка email получи последний.",
    "context": {
      "wf": {
        "vars": {
          "emails": ["user1@example.com", "user2@example.com", "user3@example.com"]
        }
      }
    },
    "target_output_format": "raw_lua"
  }'
```

Пример modify-flow:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Доработай код и добавь поле squared.",
    "existing_code": "{\"num\":\"lua{return tonumber('\''5'\'')}lua\"}",
    "target_output_format": "json_with_lua_fields"
  }'
```

Валидация:

```bash
curl -X POST http://localhost:8080/validate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Отфильтруй элементы из массива, чтобы включить только те, у которых есть значения в полях Discount или Markdown.",
    "output": "local result = _utils.array.new() return result",
    "expected_output_format": "raw_lua"
  }'
```

## OpenAPI baseline

Базовый контракт организаторов сохранен в `data/localscript-openapi.yaml`, но API аккуратно расширен:

- `POST /analyze`
- `POST /generate`
- `POST /validate`
- `GET /health`

`/generate` по-прежнему возвращает поле `code`, а также может вернуть:

- `status=needs_clarification`
- `clarification_question`
- `analysis`

## Локальность и безопасность

- внешние AI API не используются;
- генерация выполняется только через локальный `Ollama`;
- few-shot retrieval работает только по локальной базе `data/few_shot_examples.json`;
- модель подтягивается и запускается внутри docker-контура;
- валидация выполняется локально, включая `luac`;
- исходные данные и код не отправляются наружу.

## Regression

Публичная выборка:

```bash
docker compose exec app python scripts/run_public_regression.py
```

Синтетика:

```bash
docker compose exec app python scripts/run_public_regression.py --cases eval/synthetic_cases_v2.json --request-timeout-seconds 420
```

Примечание:

- на Windows можно использовать те же HTTP-запросы через `PowerShell` и `Invoke-RestMethod`, но для README основной сценарий оставлен в нейтральном `curl`-виде;
- regression теперь можно запускать прямо внутри контейнера, без локального Python и без `.venv`.

## Что еще осталось

- добавить C4-диаграммы в репозиторий;
- подготовить демо-видео и презентацию;
- зафиксировать замер VRAM на целевом GPU под параметры организаторов.
