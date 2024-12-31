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
import os
from pydantic import BaseModel, Field
import requests
from datetime import datetime, timedelta
from functools import lru_cache
from langchain_core.tools import tool
import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
import logging

from src.common.utils import get_prompts, get_config

structured_rag_uri = os.getenv('STRUCTURED_RAG_URI', 'http://structured-retriever:8081')
structured_rag_search = f"{structured_rag_uri}/search"

prompts = get_prompts()
logger = logging.getLogger(__name__)


@tool
@lru_cache
def discussion_board_rag(query: str, user_id: str) -> str:
    """Use this for answering questions about discussion boards, including posts, replies, and participation grades."""
    example_query = """
SELECT * FROM discussions 
WHERE LOWER(title) LIKE '%momentum%' 
OR LOWER(content) LIKE '%momentum%';
    """
    
    entry_doc_search = {
        "query": query + f"\n\nExample query: {example_query}",
        "top_k": 4,
        "user_id": user_id
    }
    
    try:
        response = requests.post(structured_rag_search, json=entry_doc_search)
        if response.status_code != 200:
            raise ValueError(f"Error retrieving discussion posts: {response.json()}")
            
        aggregated_content = "\n".join(chunk["content"] for chunk in response.json().get("chunks", []))
        
        if any(x in aggregated_content.lower() for x in ["no records found", "error:"]):
            raise ValueError("No discussion posts found.")
            
        return aggregated_content
    except Exception as e:
        logger.info(f"Error in discussion_board_rag: {e}")
        return "No discussion board data available."


@tool
@lru_cache
def hw_rag(query: str, user_id: str) -> str:
    """Use this for answering questions about homework, including grades, due dates, assignment details. Match homework number based on name e.g. 'Homework 1' rather than using the id."""
    example_query = f"""
SELECT *
FROM homework
WHERE student_id = {user_id};
"""

    entry_doc_search = {"query": query + f"\n\nIf asked for specific homework, match homework number based on name e.g. 'Homework 1' rather than using the id. Example query: {example_query}", "top_k": 4, "user_id": user_id}
    aggregated_content = ""
    try:
        response = requests.post(structured_rag_search, json=entry_doc_search)
        logger.info(f"Actual Structured Response : {response}")
        if response.status_code != 200:
            raise ValueError(f"Error while retireving docs: {response.json()}")
        
        aggregated_content = "\n".join(chunk["content"] for chunk in response.json().get("chunks", []))
        logger.info("&&&&&&& structured rag results")
        logger.info(aggregated_content)
        if any(x in aggregated_content.lower() for x in ["no records found", "error:"]):
            raise ValueError("No records found for the specified criteria.")
        return aggregated_content
    except Exception as e:
        logger.info(f"Some error within the structured_rag {e}, sending student_overview")
        return get_student_overview(user_id)

@tool
@lru_cache
def exam_status_rag(query: str, user_id: str) -> str:
    """Fetch exam information including grades, dates, and course details."""
    example_query = f"""
    SELECT exam_id, exam_name, exam_date, grade, course_name 
    FROM exams 
    WHERE student_id = {user_id}
    ORDER BY exam_date DESC;
    """
    
    entry_doc_search = {
        "query": query + f"\n\nExample query: {example_query}",
        "top_k": 5,
        "user_id": user_id
    }
    
    try:
        response = requests.post(structured_rag_search, json=entry_doc_search)
        if response.status_code != 200:
            raise ValueError(f"Error retrieving exam data: {response.json()}")
            
        return response.json().get("content", "No exam data available.")
    except Exception as e:
        logger.error(f"Error in exam_status_rag: {e}")
        return "Unable to fetch exam information."

class ToExamAssistant(BaseModel):
    """Handles specific exam-related queries including grades. Remember that the course_qa tool has the syllabus for more general questions."""
    query: str = Field(
        description="An exam-related question about grades, dates, or performance."
    )
    user_id: str = Field(
        description="The unique identifier of the student making the query."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What was my grade in the last Fluid Dynamics exam?",
                "user_id": "12345"
            }
        }



def get_student_overview(user_id: str) -> dict:
    """
    Retrieves a comprehensive overview of a student's data from the `students` table.
    This includes the student's basic info and excludes homework details.
    """

    STUDENT_INFO_QUERY = f"""
    SELECT student_id, first_name, last_name, email, major, enrollment_year
    FROM public.students
    WHERE student_id = {user_id};
    """
    app_database_url = get_config().database.url
    parsed_url = urlparse(f"//{app_database_url}", scheme='postgres')
    host = parsed_url.hostname
    port = parsed_url.port

    db_params = {
        'dbname': os.getenv("CUSTOMER_DATA_DB", 'customer_data'),
        'user': os.getenv("POSTGRES_USER_READONLY", 'postgres'),
        'password': os.getenv("POSTGRES_PASSWORD_READONLY", 'password'),
        'host': host,
        'port': port
    }

    overview = {
        "student_info": {}
    }

    with psycopg2.connect(**db_params) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(STUDENT_INFO_QUERY)
            student_info = cur.fetchone()
            if student_info:
                overview["student_info"] = dict(student_info)

    return overview




# @tool
# @lru_cache
# def get_recent_return_details(user_id: str) -> str:
#     """Retrieves the recent return details for a user, including order ID, product name, return status, and relevant dates."""

#     return get_purchase_history(user_id)


@tool
@lru_cache
def return_window_validation(order_date: str) -> str:
    """Use this to check the return window for validation. Use 'YYYY-MM-DD' for the order date."""

    return_window_time=os.environ.get('RETURN_WINDOW_THRESHOLD_DAYS', 15)
    try:
        # Parse the order date
        order_date = datetime.strptime(order_date, "%Y-%m-%d")

        # Get today's date
        today = os.environ.get('RETURN_WINDOW_CURRENT_DATE', "")

        if today:
            today = datetime.strptime(today, "%Y-%m-%d")
        else:
            today = datetime.now()

        # Parse the return window time
        return_days = int(return_window_time)

        # Calculate the return window end date
        return_window_end = order_date + timedelta(days=return_days)

        # Check if the product is within the return window
        if today <= return_window_end:
            days_left = (return_window_end - today).days
            return f"The product is eligible for return. {days_left} day(s) left in the return window."
        else:
            days_passed = (today - return_window_end).days
            return f"The return window has expired. It ended {days_passed} day(s) ago."
    except ValueError:
        return "Invalid date format. Please use 'YYYY-MM-DD' for the order date."

@tool
@lru_cache
def update_return(user_id: str, current_product: str, order_id: str) -> str:
    """Use this to update return status in the database."""

    # Query to retrieve the order details
    SELECT_QUERY = f"""
    SELECT order_id, product_name, order_date, order_status
    FROM public.customer_data
    WHERE customer_id='{user_id}' AND product_name='{current_product}' AND order_id='{order_id}'
    ORDER BY order_date DESC
    LIMIT 1;
    """

    # Query to update the return_status
    UPDATE_QUERY = f"""
    UPDATE public.customer_data
    SET return_status = 'Requested'
    WHERE customer_id='{user_id}' AND product_name='{current_product}' AND order_id='{order_id}';
    """

    app_database_url = get_config().database.url

    # Parse the URL
    parsed_url = urlparse(f"//{app_database_url}", scheme='postgres')

    # Extract host and port
    host = parsed_url.hostname
    port = parsed_url.port

    db_params = {
        'dbname': os.getenv("CUSTOMER_DATA_DB",'customer_data'),
        'user': os.getenv('POSTGRES_USER', None),
        'password': os.getenv('POSTGRES_PASSWORD', None),
        'host': host,
        'port': port
    }

    # Using context manager for connection and cursor
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Execute the SELECT query to verify the order details
            cur.execute(SELECT_QUERY)
            result = cur.fetchone()

            # If the order exists, update the return status
            if result:
                cur.execute(UPDATE_QUERY)
                conn.commit()  # Commit the transaction to apply the update
                return f"Return status for order_id {order_id} has been updated to 'Requested'."
            else:
                return f"No matching order found for user_id {user_id}, product_name {current_product}, and order_id {order_id}."

class ToProductQAAssistant(BaseModel):
    """
    Transfers work to a specialized assistant to handle Product QA. 
    Answers generic queries about products, including their descriptions, specifications, warranties, usage instructions, and troubleshooting issues.
    Can also address queries based on product manuals, product catalogs, FAQs, policy documents, and general product-related inquiries.
    Can also answer queries about the NVIDIA Gear Store's product offerings, policies, order management, shipping information, payment methods, returns, and customer service contacts.
    """
    query: str = Field(
        description="The question or issue related to the product. This can involve asking about product specifications, usage guidelines, troubleshooting, warranty details, or other product-related concerns."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the warranty period for this model, and does it cover screen burn-in issues?"
            }
        }

class ToCourseInfoAssistant(BaseModel):
    """
    Transfers work to a specialized assistant that handles queries specifically related to course content that can be found in a syllabus. 
    This assistant can answer queries about syllabus structure, reading lists, lecture topics, 
    and general academic guidelines for the course, as opposed to the student's performance such as homework.
    """
    query: str = Field(
        description="A course content-related question, such as topics covered in the syllabus, reading recommendations, assignment descriptions, or lecture details."
    )

    user_id: str = Field(
        description="The unique identifier of the student making the query."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What topics are covered in the syllabus for Computer Science 101?",
                "user_id": "12345"
            }
        }


class ToHomeworkStatusAssistant(BaseModel):
    """
    Delegates queries related to a student's homework status and performance within a course to a specialized assistant.
    This includes inquiries about completed homework, grades on assignments, upcoming deadlines, 
    pending submissions, and other homework-related progress metrics tied to the user's academic record.
    It leverages data generated and stored from the relevant database schema.
    """

    query: str = Field(
        description=(
            "A homework-related question, such as completed assignments, "
            "homework-specific grades, pending tasks, upcoming deadlines, or performance on recent submissions."
        )
    )
    user_id: str = Field(
        description="The unique identifier of the student making the query."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is my grade on Homework 3 for Physics 101, and are there any pending assignments?",
                "user_id": "12345"
            }
        }


class ToDiscussionBoardAssistant(BaseModel):
    """
    Delegates queries related to discussion boards to a specialized assistant.
    Handles inquiries about posts, replies, participation grades, and deadlines.
    """
    query: str = Field(
        description="A discussion board related question about posts, replies, or participation."
    )
    user_id: str = Field(
        description="The unique identifier of the student making the query."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many posts have I made in the Week 3 discussion?",
                "user_id": "12345"
            }
        }


# class ToOrderStatusAssistant(BaseModel):
#     """
#     Delegates queries specifically related to orders or purchase history to a specialized assistant.
#     This assistant handles inquiries regarding Order ID, Order Date, Quantity, Order Amount, Order Status, 
#     and any other questions related to the user's purchase history.
#     """

#     query: str = Field(
#         description="The specific query regarding the order or purchase history, such as order status, delivery updates, or historical purchase information."
#     )
#     user_id: str = Field(
#         description="The unique identifier of the user."
#     )

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "query": "What is the current status of my order?",
#                 "user_id": "1"
#             },
#              "example 2": {
#                 "query": "How many items were ordered on 2024-10-15?",
#                 "user_id": "2"
#             }
#         }


class ToReturnProcessing(BaseModel):
    """
    Transfers work to a specialized assistant which handles processing of a product return request.
    This assistant handles inquiries regarding return transactions, including return status, relevant dates, 
    reasons for return, notes, and any other questions related to return processing.
    """

    query: str = Field(
        description="The specific return-related query, such as the status of the return, refund details, or return policy."
    )
    user_id: str = Field(
        description="The unique identifier of the user requesting the return."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I wanted to return my product?",
                "user_id": "1"
            }
        }

class HandleOtherTalk(BaseModel):
    """Handles greetings and other absurd queries by offering polite redirection and clearly explaining the limitations of the chatbot."""

    message: str  # The message sent by the customer.

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello",
            }
        }

class ProductValidation(BaseModel):
    """
    Utilize this to identify the specific order or product the user is referring to in the conversation, 
    especially when the current product is unclear or unknown.
    """

    message: str  # The message sent by the customer.

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the order status for my RTX",
            }
        }