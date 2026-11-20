//! The language-model endpoint (OpenAI-compatible chat completions).
//!
//! [`ModelEndpoint`] is what the commands need; [`HttpModelEndpoint`] is the one real
//! implementation. A request asks for one JSON object at temperature 0, plus whatever
//! fields the operator adds for one endpoint ([`ExtraBody`]). The answer's text is read
//! from `choices[0].message.content` and nowhere else, and handed back as it came:
//! reading it strictly is the caller's job ([`crate::text`]).

use crate::error::RunError;
use crate::http::{Http, endpoint, parse_json, shown};
use crate::secret::{Secret, usable};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::future::Future;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Fields added at the top level of every chat-completion request, after the ones this
/// tool sets: what one endpoint needs and another does not, such as
/// `{"chat_template_kwargs": {"enable_thinking": false}}` for a model that would
/// otherwise think aloud.
///
/// It is the one free-form JSON value this tool sends: its shape belongs to the endpoint,
/// not to this contract. Keys the tool sets itself are refused, and so are the keys that
/// would change the shape of the answer, so a field given here can never undo one the
/// run relies on. It is recorded in the run manifest; it is not a place for a secret.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ExtraBody(Map<String, Value>);

impl ExtraBody {
    /// Keys this tool sets (`model`, `messages`, `temperature`, `response_format`,
    /// `max_tokens`), and keys that would change the answer's shape: `stream` (an answer
    /// in pieces) and `n` (several answers, of which one would be read and the others,
    /// with their tokens, dropped).
    pub const REFUSED: [&str; 7] = [
        "model",
        "messages",
        "temperature",
        "response_format",
        "max_tokens",
        "stream",
        "n",
    ];

    /// Reads `--extra-body`: one JSON object, none of whose keys is refused.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(raw).map_err(|e| format!("not JSON: {e}"))?;
        let Value::Object(fields) = value else {
            return Err(
                "a JSON object, such as {\"chat_template_kwargs\": {\"enable_thinking\": false}}".into(),
            );
        };
        if let Some(refused) = fields.keys().find(|key| Self::REFUSED.contains(&key.as_str())) {
            return Err(format!(
                "{refused:?} is set by cb-run itself or changes the answer's shape; it cannot be given here"
            ));
        }
        Ok(Self(fields))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The fields as one line of JSON, keys in order: the same text for the same fields.
    pub fn canonical(&self) -> String {
        serde_json::to_string(&self.0).unwrap_or_default()
    }
}

impl FromStr for ExtraBody {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        Self::parse(raw)
    }
}

/// One question to the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ask<'a> {
    pub model: &'a str,
    /// The whole message: instructions, then the material.
    pub message: &'a str,
    /// A bound on the answer's length, when the operator set one.
    pub max_tokens: Option<u32>,
    /// Fields added to the request for this endpoint.
    pub extra: Option<&'a ExtraBody>,
}

/// The model's answer to one question.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAnswer {
    /// The answer's text, exactly as the endpoint returned it in `choices[0].message.content`.
    pub content: String,
    /// The token counts of the endpoint's `usage`, when it reported them.
    pub tokens_in: Option<u32>,
    pub tokens_out: Option<u32>,
    /// The model name the endpoint put in its answer.
    pub reported_model: Option<String>,
    /// The wall time of the call that produced this answer, and of that call only.
    pub wall: Duration,
}

/// What the commands need from a language model. One try per call.
pub trait ModelEndpoint: Sync {
    /// The endpoint's address as it may be shown.
    fn address(&self) -> String;

    /// Asks for one JSON-object answer.
    fn ask(&self, ask: &Ask<'_>) -> impl Future<Output = Result<ModelAnswer, RunError>> + Send;

    /// Whether answers are (still) asked for with `response_format: json_object`.
    fn json_mode(&self) -> bool;
}

// ---- wire ----------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    r#type: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [ChatMessage<'a>; 1],
    temperature: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// The operator's fields, after the tool's own; none of their keys is one of these.
    #[serde(flatten)]
    extra: Option<&'a ExtraBody>,
}

impl<'a> ChatRequest<'a> {
    fn of(ask: &Ask<'a>, with_format: bool) -> Self {
        Self {
            model: ask.model,
            messages: [ChatMessage {
                role: "user",
                content: ask.message,
            }],
            temperature: 0,
            response_format: with_format.then_some(ResponseFormat {
                r#type: "json_object",
            }),
            max_tokens: ask.max_tokens,
            extra: ask.extra.filter(|extra| !extra.is_empty()),
        }
    }
}

/// The part of an answer this tool reads. A `reasoning` or `reasoning_content` field,
/// and everything else the endpoint adds, is not read.
#[derive(Debug, Deserialize)]
struct ChatAnswer {
    choices: Vec<Choice>,
    usage: Option<Usage>,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AnswerMessage,
}

#[derive(Debug, Deserialize)]
struct AnswerMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

/// Reads an endpoint's answer: the text of `choices[0].message.content`, the `usage`
/// token counts when reported, and the model name. An answer without that text (no
/// choice, or a choice whose content is null, as when a model put everything in its
/// reasoning) is an answer of another shape.
fn answer_from(url: &Url, body: &[u8], wall: Duration) -> Result<ModelAnswer, RunError> {
    let answer: ChatAnswer = parse_json(url, body)?;
    let content = answer
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| RunError::Malformed {
            url: shown(url),
            reason: "no text in choices[0].message.content".into(),
            body: crate::error::excerpt(body),
        })?;
    Ok(ModelAnswer {
        content,
        tokens_in: answer.usage.as_ref().and_then(|u| u.prompt_tokens),
        tokens_out: answer.usage.as_ref().and_then(|u| u.completion_tokens),
        reported_model: answer.model,
        wall,
    })
}

/// The chat-completions address for a base address: `…/v1` becomes
/// `…/v1/chat/completions`; an address that already ends there is used as it is.
pub fn chat_completions_url(base: &Url) -> Result<Url, RunError> {
    if base.path().trim_end_matches('/').ends_with("/chat/completions") {
        return Ok(base.clone());
    }
    endpoint(base, &["chat", "completions"])
}

/// A language model over HTTP.
#[derive(Debug)]
pub struct HttpModelEndpoint {
    http: Http,
    url: Url,
    key: Option<Secret>,
    /// Starts true. Set false, for the rest of the run, the first time the endpoint
    /// refuses a request with `response_format` and accepts the same request without it.
    json_mode: AtomicBool,
}

impl HttpModelEndpoint {
    pub fn new(http: Http, base: &Url, key: Option<Secret>) -> Result<Self, RunError> {
        Ok(Self {
            http,
            url: chat_completions_url(base)?,
            key: usable(key.as_ref()).cloned(),
            json_mode: AtomicBool::new(true),
        })
    }

    async fn exchange(&self, ask: &Ask<'_>, with_format: bool) -> Result<ModelAnswer, RunError> {
        let body = ChatRequest::of(ask, with_format);
        let mut request = self.http.client().post(self.url.clone()).json(&body);
        if let Some(key) = &self.key {
            request = request.bearer_auth(key.expose());
        }
        let answered = self.http.exchange(&self.url, request).await?;
        answer_from(&self.url, &answered.body, answered.wall)
    }
}

impl ModelEndpoint for HttpModelEndpoint {
    fn address(&self) -> String {
        shown(&self.url)
    }

    fn json_mode(&self) -> bool {
        self.json_mode.load(Ordering::Relaxed)
    }

    async fn ask(&self, ask: &Ask<'_>) -> Result<ModelAnswer, RunError> {
        if !self.json_mode() {
            return self.exchange(ask, false).await;
        }
        match self.exchange(ask, true).await {
            // The endpoint refused the request as it was written. It may not know
            // `response_format`: the same request goes once more without the field.
            Err(
                refusal @ RunError::Status {
                    status: 400 | 422, ..
                },
            ) => {
                match self.exchange(ask, false).await {
                    Ok(answer) => {
                        if self.json_mode.swap(false, Ordering::Relaxed) {
                            tracing::warn!(
                                url = %self.address(),
                                "the endpoint refuses response_format; asking without it from here on"
                            );
                        }
                        Ok(answer)
                    }
                    // Refused without the field too: the refusal was about something else.
                    Err(second) if !second.is_retryable() => Err(refusal),
                    Err(second) => Err(second),
                }
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask<'a>(extra: Option<&'a ExtraBody>) -> Ask<'a> {
        Ask {
            model: "qwen3-8b",
            message: "Write it down.",
            max_tokens: None,
            extra,
        }
    }

    #[test]
    fn a_request_asks_for_one_json_object_at_temperature_zero() {
        assert_eq!(
            serde_json::to_value(ChatRequest::of(&ask(None), true)).unwrap(),
            serde_json::json!({
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": "Write it down."}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            })
        );
        let plain = serde_json::to_value(ChatRequest::of(&ask(None), false)).unwrap();
        assert!(plain.get("response_format").is_none());
    }

    #[test]
    fn the_extra_body_is_merged_at_the_top_level_after_the_tools_fields() {
        let extra: ExtraBody =
            r#"{"reasoning": {"enabled": false}, "chat_template_kwargs": {"enable_thinking": false}}"#
                .parse()
                .unwrap();
        let body = serde_json::to_string(&ChatRequest::of(&ask(Some(&extra)), true)).unwrap();
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["reasoning"], serde_json::json!({"enabled": false}));
        assert_eq!(
            value["chat_template_kwargs"],
            serde_json::json!({"enable_thinking": false})
        );
        assert_eq!(value["temperature"], 0);
        assert_eq!(
            value["response_format"],
            serde_json::json!({"type": "json_object"})
        );
        // After the tool's own fields.
        assert!(
            body.find("\"response_format\"") < body.find("\"chat_template_kwargs\""),
            "{body}"
        );
        // An empty one adds nothing.
        let empty: ExtraBody = "{}".parse().unwrap();
        assert_eq!(
            serde_json::to_value(ChatRequest::of(&ask(Some(&empty)), true)).unwrap(),
            serde_json::to_value(ChatRequest::of(&ask(None), true)).unwrap()
        );
    }

    #[test]
    fn the_extra_body_cannot_set_what_the_tool_sets_or_change_the_answers_shape() {
        for refused in ExtraBody::REFUSED {
            let raw = format!(r#"{{"{refused}": 1, "top_k": 20}}"#);
            let error = raw.parse::<ExtraBody>().unwrap_err();
            assert!(error.contains(refused), "{error}");
        }
        for not_an_object in ["", "[1]", "\"x\"", "null", "{\"a\":"] {
            assert!(not_an_object.parse::<ExtraBody>().is_err(), "{not_an_object:?}");
        }
        let extra: ExtraBody = r#"{"b": 1, "a": {"y": true, "x": null}}"#.parse().unwrap();
        assert_eq!(extra.canonical(), r#"{"a":{"x":null,"y":true},"b":1}"#);
    }

    #[test]
    fn the_answer_is_read_from_the_first_choices_content_only() {
        let url = Url::parse("http://model/v1/chat/completions").unwrap();
        let body = serde_json::json!({
            "model": "qwen3.8-27b",
            "choices": [
                {"message": {
                    "role": "assistant",
                    "content": "{\"found\": false}",
                    "reasoning_content": "{\"found\": true, \"problem\": \"a draft\"}",
                    "reasoning": "Let me think about the call first."
                }},
                {"message": {"role": "assistant", "content": "{\"found\": true}"}}
            ],
            "usage": {"prompt_tokens": 812, "completion_tokens": 9, "total_tokens": 821,
                      "completion_tokens_details": {"reasoning_tokens": 0}}
        });
        let answer = answer_from(&url, body.to_string().as_bytes(), Duration::from_millis(7)).unwrap();
        assert_eq!(answer.content, "{\"found\": false}");
        assert_eq!((answer.tokens_in, answer.tokens_out), (Some(812), Some(9)));
        assert_eq!(answer.reported_model.as_deref(), Some("qwen3.8-27b"));
        assert_eq!(answer.wall, Duration::from_millis(7));

        // No usage: no counts (the run writes zero and counts the answer).
        let bare = serde_json::json!({"choices": [{"message": {"content": "{}"}}]});
        let answer = answer_from(&url, bare.to_string().as_bytes(), Duration::ZERO).unwrap();
        assert_eq!((answer.tokens_in, answer.tokens_out), (None, None));

        // Everything in the reasoning and nothing in the content: not an answer.
        let thought_only = serde_json::json!({"choices": [{"message": {
            "content": null, "reasoning_content": "{\"found\": false}"}}]});
        let error = answer_from(&url, thought_only.to_string().as_bytes(), Duration::ZERO).unwrap_err();
        assert!(matches!(error, RunError::Malformed { .. }), "{error}");
        let no_choice = serde_json::json!({"choices": []});
        assert!(answer_from(&url, no_choice.to_string().as_bytes(), Duration::ZERO).is_err());
    }

    #[test]
    fn the_chat_address_is_the_base_plus_the_route_unless_it_is_already_there() {
        let url = |s: &str| chat_completions_url(&Url::parse(s).unwrap()).unwrap().to_string();
        assert_eq!(
            url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1/chat/completions"
        );
        assert_eq!(
            url("http://127.0.0.1:8000/v1/"),
            "http://127.0.0.1:8000/v1/chat/completions"
        );
        assert_eq!(
            url("https://api.example.com/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions"
        );
    }
}
