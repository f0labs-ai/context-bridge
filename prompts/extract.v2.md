You are given the transcript of one phone call between a customer and a support agent of a service provider. The call started at {{started_at}} (UTC).

Write down only what the AGENT says is known about a problem with the service that affects other customers too: an outage, a fault, a degraded service. Use nothing but the transcript. Leave out everything about this customer's own account, device or identity.

Answer with one JSON object and nothing else:

{
  "found": true,
  "problem": "Which service, where, and what customers experience. One sentence.",
  "solution": "What the agent says is known now: the cause, what customers can do meanwhile, the fix and when service is expected back, or that it is fixed. One to three sentences. Keep the agent's own specific words for causes, actions and clock times.",
  "t_start": "When the problem started, as an ISO 8601 UTC date and time such as 2026-06-01T09:10:00Z, or null.",
  "t_eta": "When the agent says service will be back, or was restored, as an ISO 8601 UTC date and time, or null.",
  "confidence": { "problem": 0.0, "solution": 0.0, "time": 0.0 }
}

For t_start and t_eta, give a date and time only when the agent states a clock time; read "today", "yesterday" and "tomorrow" from the call's start. When the agent states no clock time, write null: the JSON value null, without quotation marks.

Each confidence value is between 0 and 1 and says how clearly the agent stated that part: 1.0 when it was stated plainly and specifically, about 0.5 when it was vague or hedged ("it might be", "probably"), 0.0 when it was not stated.

If the agent states no such problem, answer {"found": false}.
