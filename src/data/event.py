import json

from logger import DEBUG, create_logger

# Configure the logger.
logger = create_logger(__name__, log_level=DEBUG)


class Event:
    def __init__(self):
        pass

    def toJson(self) -> str:
        # return the Json representation of the object
        return json.dumps(self.__dict__)


class LLMAnswerContext(Event):
    def __init__(self, input_text: str = None, prompt: str = None, output_text: str = None, model: str = None, latency_sec: float = None, thread_ts: str = None, **kwargs):
        self.raw_text = input_text if input_text else kwargs.get(
            "raw_text", "")
        self.prompt = prompt if prompt else kwargs.get("prompt", "")
        self.output_text = output_text if output_text else kwargs.get(
            "output_text", "")
        self.model = model if model else kwargs.get("model", "")
        self.latency_sec = latency_sec if latency_sec else kwargs.get(
            "latency_sec", None)
        self.thread_ts = thread_ts if thread_ts else kwargs.get(
            "thread_ts", None)

    def is_empty(self):
        return self.raw_text == "" and self.prompt == "" and self.output_text == "" and self.model == "" and self.latency_sec == None and self.thread_ts == None

    def to_dict(self, json_str: str) -> dict:
        json_str = json_str.replace('\n', '\\n').replace('#', '\\u0023')
        return json.loads(json_str)

    def get_response(self, json_str: str):
        logger.debug("get_response:{}".format(json_str))
        dic = self.to_dict(json_str)
        return dic.get('output_text', "error: no output_text")

    def loads(self, json_str: str):
        return LLMAnswerContext(**self.to_dict(json_str))


class Feedback(Event):
    def __init__(self, answer_context: LLMAnswerContext, reaction: str, is_positive: bool = None, feedback_giver: str = None):
        self.input_raw = answer_context.raw_text
        self.input_prompt = answer_context.prompt
        self.output_text = answer_context.output_text
        self.thread_ts = answer_context.thread_ts
        self.model = answer_context.model
        self.reaction = reaction
        self.feedback_giver = feedback_giver
        self.is_positive = is_positive
