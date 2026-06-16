# Writeup — Document Assistant

A few notes on how I built the assistant and some runs that show it working. The
code is in `starter/` (start it with `python starter/main.py`), and the runtime
output landed in `starter/logs/` and `starter/sessions/`.

## Architecture & Routing Decisions

It's a LangGraph graph with five nodes:

```
classify_intent -> [qa_agent | summarization_agent | calculation_agent] -> update_memory -> END
```

`classify_intent` always runs first. It asks the LLM to label the request as qa,
summarization, calculation or unknown, and then I set `next_step` to the matching
node (unknown just goes to the Q&A agent). The conditional edge sends the request
to that node from there.

One thing I ran into: the classifier kept getting questions like "what's the
total of invoice XYZ" wrong until I added a few examples to the intent prompt.
After that the routing was solid. I also kept the routing keys equal to the node
names on purpose.

Each agent can use the document tools and the calculator and returns a typed
result. After an agent runs, `update_memory` writes a short summary and notes
which documents were used, and then the graph ends.

## State & Memory

All nodes share one `AgentState` (a TypedDict). A node only returns the fields it
changes and LangGraph merges them in. Two fields use a reducer so they grow
instead of being overwritten: `messages` (add_messages) and `actions_taken`
(operator.add).

For memory I compile the graph with `InMemorySaver` and set `thread_id` to the
session id. So the state is saved per session, and a follow-up question still has
the earlier messages and summary. `update_memory` keeps that summary and the list
of active documents up to date for the next turn.

## Structured Output

Free text wasn't allowed and it is not a good choice, so every LLM step uses
`llm.with_structured_output(...)`:

- classify_intent -> UserIntent
- qa_agent -> AnswerResponse
- summarization_agent -> SummarizationResponse
- calculation_agent -> CalculationResponse
- update_memory -> UpdateMemoryResponse

The schemas also keep things valid: `confidence` has to be between 0 and 1, and
`intent_type` can only be one of the four labels. The calculator returns a
string, checks the input first (so no code can run) and gives a clear error for
things like divide by zero instead of crashing.

## Example Conversations

These are from one real run of `python main.py`. The same session is saved in
`starter/sessions/` and the tool calls in `starter/logs/`.

```
> Who are the two parties in the service agreement contract?
DocDacity Solutions Inc. (Provider) and Healthcare Partners LLC (Client).
[intent: qa | tools: document_search, document_reader]

> Calculate the sum of all invoice totals
$305,800  (22,000 + 69,300 + 214,500) - the calculator tool does the actual sum.
[intent: calculation | tools: document_search, document_reader, calculator]

> Summarize all contracts
Service agreement CON-001: 12 months, $15,000/month ($180,000 total), 60 days notice to cancel.
[intent: summarization]

> What is the total amount in invoice INV-001?
$22,000  ($20,000 + $2,000 tax).

> And what is the total of invoice INV-002?
$69,300 - this still works because the earlier turns stay in the same session.
```
