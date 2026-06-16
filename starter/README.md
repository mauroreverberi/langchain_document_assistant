# Document Assistant

A LangGraph assistant for financial and healthcare documents. It classifies each
request (a question, a summary or a calculation), routes it to the matching
agent, uses tools to look things up in the documents, and keeps context across
turns.

```
classify_intent -> [qa_agent | summarization_agent | calculation_agent] -> update_memory -> END
```

## Run it

```
cp .env.example .env      # add your OpenAI API key
pip install -r requirements.txt
python main.py
```

Commands in the chat: `/docs`, `/help`, `/quit`.

## How it works

- One graph with five nodes. `classify_intent` runs first and routes the request
  to the agent that matches the detected intent.
- Each agent (`qa_agent`, `summarization_agent`, `calculation_agent`) uses the
  document tools and returns a typed response object.
- `update_memory` then writes a short conversation summary and records which
  documents were used, before the graph ends.

## State and memory

- `AgentState` (a `TypedDict`) is shared by all nodes. Each node returns only the
  fields it changes, and LangGraph merges them.
- `messages` uses the `add_messages` reducer and `actions_taken` uses
  `operator.add`, so both accumulate instead of overwriting.
- The graph is compiled with an `InMemorySaver` checkpointer and `thread_id` is
  set to the session id, so a follow-up question still has the earlier context.
- `update_memory` keeps the rolling summary and the list of active documents up
  to date for the next turn.

## Structured output

Every LLM step returns a typed object via `llm.with_structured_output(Schema)`
instead of free text:

- `classify_intent` -> `UserIntent`
- `qa_agent` -> `AnswerResponse`
- `summarization_agent` -> `SummarizationResponse`
- `calculation_agent` -> `CalculationResponse`
- `update_memory` -> `UpdateMemoryResponse`

The schemas also enforce constraints: `confidence` must be between 0 and 1, and
`intent_type` is restricted to `qa`, `summarization`, `calculation` or `unknown`.

## Example conversations

Real output from running `python main.py`:

```
> Who are the two parties in the service agreement contract?
DocDacity Solutions Inc. (Provider) and Healthcare Partners LLC (Client).
intent: qa | tools: document_search, document_reader

> Calculate the sum of all invoice totals
The sum of all invoice totals is $305,800 (INV-001 22,000 + INV-002 69,300 + INV-003 214,500).
intent: calculation | tools: document_search, document_reader, calculator

> Summarize all contracts
Service agreement CON-001: 12 months, $15,000/month ($180,000 total), 60 days notice to terminate.
intent: summarization | tools: document_search, document_reader

> And what is the total of invoice INV-002?
$69,300   (answered from the memory of the previous turn)
```

## Calculator edge cases

The calculator validates the input and handles bad cases instead of crashing.
These are the values the tool returns when called directly (checked in
`test_unit.py`):

```
5/0              -> Error: you can't divide by zero.
(empty)          -> Error: the expression is empty.
2 + abc          -> Error: only numbers and + - * / % ( ) are allowed.
1,000+500        -> 1500   (commas in numbers are fine)
```

## Tests

```
python -m unittest test_unit                           # offline, no API key needed
RUN_LIVE_TESTS=1 python -m unittest test_integration   # live, needs an API key
```
