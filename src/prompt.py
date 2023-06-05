from langchain.prompts.prompt import PromptTemplate

_TEAM_CHAT_TEMPLATE = """Your name is Determinant, simulate co-worker's style of communication in a helpful and collaborative manner. 
Also be specific yet concise in your response. Talk naturally in the conversation. 
I will give you a list of messages, and you shall generate appropriate response when you think it's relevant.

{message_history}
Determinant:"""

TEAM_CHAT_PROMPT = PromptTemplate(
    input_variables=["message_history"], template=_TEAM_CHAT_TEMPLATE
)

_NVC_CHAT_TEMPLATE = """

"""