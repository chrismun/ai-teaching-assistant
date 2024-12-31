# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Annotated, TypedDict, Dict
from langgraph.graph.message import AnyMessage, add_messages
from typing import Callable
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from typing import Annotated, Literal, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig
from src.agent.tools import (
         get_student_overview, ToCourseInfoAssistant, ToHomeworkStatusAssistant, hw_rag, get_student_overview, HandleOtherTalk,
         discussion_board_rag, ToDiscussionBoardAssistant,
         exam_status_rag, ToExamAssistant
)

from src.agent.utils import create_tool_node_with_fallback, get_checkpointer, canonical_rag
from src.common.utils import get_llm, get_prompts

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "aita"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

logger = logging.getLogger(__name__)
prompts = get_prompts()
# TODO get the default_kwargs from the Agent Server API
default_llm_kwargs = {"temperature": 0.2, "top_p": 0.7, "max_tokens": 1024}
    
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    student_info: Dict
    needs_clarification: bool
    clarification_type: str
    reason: str

async def handle_other_talk(state: State, config: RunnableConfig):
    """Handles greetings and queries outside order status, returns, or products, providing polite redirection and explaining chatbot limitations."""

    prompt = prompts.get("other_talk_template", "")

    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", prompt),
        ("placeholder", "{messages}"),
        ]
    )

    # LLM
    llm_settings = config.get('configurable', {}).get("llm_settings", default_llm_kwargs)
    llm = get_llm(**llm_settings)
    llm = llm.with_config(tags=["should_stream"])

    # Chain
    small_talk_chain = prompt | llm
    response = await small_talk_chain.ainvoke(state, config)

    return {"messages": [response]}


def create_entry_node(assistant_name: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " your task is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ]
        }

    return entry_node

# async def ask_clarification(state: State, config: RunnableConfig):

#     # Extract the base prompt
#     base_prompt = prompts.get("ask_clarification")["base_prompt"]
#     previous_conversation = [m for m in state['messages'] if not isinstance(m, ToolMessage)]
#     base_prompt = base_prompt.format(previous_conversation=previous_conversation)

#     purchase_history = state.get("user_purchase_history", [])
#     if state["clarification_type"] == "no_product" and state['reason'].strip():
#         followup_prompt = prompts.get("ask_clarification")["followup"]["no_product"].format(
#             reason=state['reason'],
#             purchase_history=purchase_history
#         )
#     elif not state['reason'].strip():
#         followup_prompt = prompts.get("ask_clarification")["followup"]["default"].format(reason=purchase_history)
#     else:
#         followup_prompt = prompts.get("ask_clarification")["followup"]["default"].format(reason=state['reason'])

#     # Combine base prompt and followup prompt
#     prompt = f"{base_prompt} {followup_prompt}"

#     # LLM
#     llm_settings = config.get('configurable', {}).get("llm_settings", default_llm_kwargs)
#     llm = get_llm(**llm_settings)
#     llm = llm.with_config(tags=["should_stream"])

#     response = await llm.ainvoke(prompt, config)

#     return {"messages": [response]}

async def handle_course_qa(state: State, config: RunnableConfig):

    previous_conversation = [m for m in state['messages'] if not isinstance(m, ToolMessage) and m.content]
    message_type_map = {
        HumanMessage: "user",
        AIMessage: "assistant",
        SystemMessage: "system"
    }

    get_role = lambda x: message_type_map.get(type(x), None)
    previous_conversation_serialized = [{"role": get_role(m), "content": m.content} for m in previous_conversation if m.content]
    last_message = previous_conversation_serialized[-1]['content']
    previous_conversation_serialized.append({"role": "assistant", "content": " "})
    
    logger.info("************ Prev convo serial:")
    logger.info(previous_conversation_serialized)

    retrieved_content = canonical_rag(query=last_message, conv_history=previous_conversation_serialized)
    
    logger.info("************************** Retrieved info:")
    logger.info(retrieved_content)
    
    base_rag_prompt = prompts.get("rag_template")
    rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", base_rag_prompt),
        MessagesPlaceholder("chat_history") + "\n\nCONTEXT:  {context}"
    ]
    )
    rag_prompt = rag_prompt.format(chat_history=previous_conversation, context=retrieved_content)

    llm_settings = config.get('configurable', {}).get("llm_settings", default_llm_kwargs)
    llm = get_llm(**llm_settings)
    llm = llm.with_config(tags=["should_stream"])

    response = await llm.ainvoke(rag_prompt, config)

    return {"messages": [response]}

class Assistant:
    def __init__(self, prompt: str, tools: list):
        self.prompt = prompt
        self.tools = tools

    async def __call__(self, state: State, config: RunnableConfig):
        while True:

            llm_settings = config.get('configurable', {}).get("llm_settings", default_llm_kwargs)
            llm = get_llm(**llm_settings)
            runnable = self.prompt | llm.bind_tools(self.tools)
            last_message = state["messages"][-1]
            messages = []
            if isinstance(last_message, ToolMessage) and last_message.name in ["hw_rag", "return_window_validation", "update_return", "get_purchase_history", "get_recent_return_details"]: # old
                gen = runnable.with_config(
                tags=["should_stream"],
                callbacks=config.get(
                    "callbacks", []
                ),  # <-- Propagate callbacks (Python <= 3.10)
                )
                async for message in gen.astream(state):
                    messages.append(message.content)
                result = AIMessage(content="".join(messages))
            else:
                result = runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                logger.info("$$$$$$$$$$ messages within Assistant:")
                logger.info(messages)
            else:
                break
        return {"messages": result}

### course status Assistant (TODO shouldnt this go below subagents)
homework_status_prompt_template = prompts.get("homework_status_template", "")

homework_status_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            homework_status_prompt_template
        ),
        ("placeholder", "{messages}"),
    ]
)

homework_status_safe_tools = [hw_rag]
homework_status_tools = homework_status_safe_tools

# return_processing_safe_tools = [get_recent_return_details, return_window_validation]
# return_processing_sensitive_tools = [update_return]
# return_processing_tools = return_processing_safe_tools + return_processing_sensitive_tools + [ProductValidation]

primary_assistant_prompt_template = prompts.get("primary_assistant_template", "")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            primary_assistant_prompt_template
        ),
        ("placeholder", "{messages}"),
    ]
)

primary_assistant_tools = [
        HandleOtherTalk,
        ToCourseInfoAssistant,
        ToHomeworkStatusAssistant,
        ToDiscussionBoardAssistant,
        ToExamAssistant
        # ToReturnProcessing,
    ]

# BUILD THE GRAPH
builder = StateGraph(State)

# SUB AGENTS

# Create course_qa Assistant
builder.add_node(
    "enter_course_qa",
    handle_course_qa,
)

builder.add_edge("enter_course_qa", END)

# TODO: course validation?
# builder.add_node("order_validation", validate_product_info)
# builder.add_node("ask_clarification", ask_clarification)

# Create homework_status Assistant
builder.add_node(
    "enter_homework_status", create_entry_node("Homework Assistant")
)
builder.add_node("homework_status", Assistant(homework_status_prompt, homework_status_tools))
builder.add_edge("enter_homework_status", "homework_status")
builder.add_node(
    "homework_status_safe_tools",
    create_tool_node_with_fallback(homework_status_safe_tools),
)


def route_homework_status(
    state: State,
) -> Literal[
    "homework_status_safe_tools",
    # "order_validation",
    "__end__"
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    tool_names = [t.name for t in homework_status_safe_tools]
    # do_product_validation = any(tc["name"] == ProductValidation.__name__ for tc in tool_calls)
    # if do_product_validation:
    #     return "order_validation"
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "homework_status_safe_tools"
    return "homework_status_sensitive_tools"

builder.add_edge("homework_status_safe_tools", "homework_status")
builder.add_conditional_edges("homework_status", route_homework_status)

# Discussion board asst 
discussion_board_tools = [discussion_board_rag]

discussion_board_prompt = prompts.get("discussion_board_template", "")
discussion_board_prompt = ChatPromptTemplate.from_messages([
    ("system", discussion_board_prompt),
    ("placeholder", "{messages}"),
])

builder.add_node(
    "enter_discussion_board",
    create_entry_node("Discussion Board Assistant")
)
builder.add_node(
    "discussion_board",
    Assistant(discussion_board_prompt, discussion_board_tools)
)
builder.add_edge("enter_discussion_board", "discussion_board")

builder.add_node(
    "discussion_board_tools",
    create_tool_node_with_fallback(discussion_board_tools),
)

def route_discussion_board(
    state: State,
) -> Literal["discussion_board_tools", "__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    return "discussion_board_tools"

builder.add_edge("discussion_board_tools", "discussion_board")
builder.add_conditional_edges("discussion_board", route_discussion_board)


# Exam Assistant
exam_status_tools = [exam_status_rag]

exam_status_prompt = prompts.get("exam_status_template", "")
exam_status_prompt = ChatPromptTemplate.from_messages([
    ("system", exam_status_prompt),
    ("placeholder", "{messages}"),
])

# Add nodes
builder.add_node(
    "enter_exam_status",
    create_entry_node("Exam Status Assistant")
)
builder.add_node(
    "exam_status",
    Assistant(exam_status_prompt, exam_status_tools)
)
builder.add_edge("enter_exam_status", "exam_status")

# Add tool node
builder.add_node(
    "exam_status_tools",
    create_tool_node_with_fallback(exam_status_tools),
)

# Add routing function
def route_exam_status(
    state: State,
) -> Literal["exam_status_tools", "__end__"]:
    route = tools_condition(state)
    if route == END:
        return END
    return "exam_status_tools"

builder.add_edge("exam_status_tools", "exam_status")
builder.add_conditional_edges("exam_status", route_exam_status)


# Create return_processing Assistant
# builder.add_node("return_validation", validate_product_info)

# builder.add_node(
#     "enter_return_processing",
#     create_entry_node("Return Processing Assistant"),
# )
# builder.add_node("return_processing", Assistant(return_processing_prompt, return_processing_tools))
# builder.add_edge("enter_return_processing", "return_processing")

# builder.add_node(
#     "return_processing_safe_tools",
#     create_tool_node_with_fallback(return_processing_safe_tools),
# )
# builder.add_node(
#     "return_processing_sensitive_tools",
#     create_tool_node_with_fallback(return_processing_sensitive_tools),
# )


# def route_return_processing(
#     state: State,
# ) -> Literal[
#     "return_processing_safe_tools",
#     "return_processing_sensitive_tools",
#     "return_validation",
#     "__end__",
# ]:
#     route = tools_condition(state)
#     if route == END:
#         return END
#     tool_calls = state["messages"][-1].tool_calls
#     do_product_validation = any(tc["name"] == ProductValidation.__name__ for tc in tool_calls)
#     if do_product_validation:
#         return "return_validation"
#     tool_names = [t.name for t in return_processing_safe_tools]
#     if all(tc["name"] in tool_names for tc in tool_calls):
#         return "return_processing_safe_tools"
#     return "return_processing_sensitive_tools"


# builder.add_edge("return_processing_sensitive_tools", "return_processing")
# builder.add_edge("return_processing_safe_tools", "return_processing")
# builder.add_conditional_edges("return_processing", route_return_processing)


# def user_info(state: State):
#     return {"user_purchase_history": get_purchase_history(state["user_id"]), "current_product": ""}

def fetch_student_info(state: State) -> Dict:
    user_id = state.get("user_id")
    student_data = get_student_overview(user_id)
    return {"student_info": student_data}

builder.add_node("fetch_student_info", fetch_student_info)
builder.add_edge(START, "fetch_student_info")

# builder.add_node("fetch_purchase_history", user_info)
# builder.add_edge(START, "fetch_purchase_history")
# builder.add_edge("ask_clarification", END)

# Primary assistant
builder.add_node("primary_assistant", Assistant(primary_assistant_prompt, primary_assistant_tools))
builder.add_node(
    "other_talk", handle_other_talk
)

def route_primary_assistant(
    state: State,
) -> Literal[
    "enter_course_qa",
    "enter_homework_status",
    "enter_discussion_board",
    "enter_exam_status",
    "other_talk",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToCourseInfoAssistant.__name__:
            return "enter_course_qa"
        elif tool_calls[0]["name"] == ToHomeworkStatusAssistant.__name__:
            return "enter_homework_status"
        elif tool_calls[0]["name"] == ToDiscussionBoardAssistant.__name__:
            return "enter_discussion_board"
        elif tool_calls[0]["name"] == ToExamAssistant.__name__:
            return "enter_exam_status"
        elif tool_calls[0]["name"] == HandleOtherTalk.__name__:
            return "other_talk"
    raise ValueError("Invalid route")


builder.add_edge("other_talk", END)

# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_course_qa": "enter_course_qa",
        "enter_homework_status": "enter_homework_status",
        "enter_discussion_board": "enter_discussion_board",
        "enter_exam_status": "enter_exam_status",
        "other_talk":"other_talk",
        END: END,
    },
)


# def is_order_product_valid(state: State)  -> Literal[
#     "ask_clarification",
#     "order_status"
# ]:
#     """Conditional edge from validation node to decide if we should ask followup questions"""
#     if state["needs_clarification"] == True:
#         return "ask_clarification"
#     return "order_status"

# def is_return_product_valid(state: State)  -> Literal[
#     "ask_clarification",
#     "return_processing"
# ]:
#     """Conditional edge from validation node to decide if we should ask followup questions"""
#     if state["needs_clarification"] == True:
#         return "ask_clarification"
#     return "return_processing"

# builder.add_conditional_edges(
#     "order_validation",
#     is_order_product_valid
# )
# builder.add_conditional_edges(
#     "return_validation",
#     is_return_product_valid
# )

builder.add_edge("fetch_student_info", "primary_assistant")


# Allow multiple async loop togeather
# This is needed to create checkpoint as it needs async event loop
# TODO: Move graph build into a async function and call that to remove nest_asyncio
import nest_asyncio
nest_asyncio.apply()

# To run the async main function
import asyncio

memory = None
pool = None

# TODO: Remove pool as it's not getting used
# WAR: It's added so postgres does not close it's session
async def get_checkpoint():
    global memory, pool
    memory, pool = await get_checkpointer()

asyncio.run(get_checkpoint())

# Compile
graph = builder.compile(checkpointer=memory,
                        # interrupt_before=["return_processing_sensitive_tools"], ## asking human if approved to run tool 
                        #interrupt_after=["ask_human"] 
                        )

try:
    # Generate the PNG image from the graph
    png_image_data = graph.get_graph(xray=True).draw_mermaid_png()
    # Save the image to a file in the current directory
    with open("graph_image.png", "wb") as f:
        f.write(png_image_data)
except Exception as e:
    # This requires some extra dependencies and is optional
    logger.info(f"An error occurred: {e}")