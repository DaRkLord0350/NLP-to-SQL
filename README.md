# Natural Language to SQL with Google Gemini & LangChain

This project provides an end-to-end solution for translating natural language questions into executable MySQL queries. It leverages Google's Gemini LLM, LangChain, and a semantic few-shot prompting technique to achieve high accuracy. The entire application is exposed through an interactive Gradio web interface.

This repository is designed to be a robust starting point for building sophisticated data-querying tools that empower users to interact with complex databases without writing a single line of SQL.

## 🎯 The Problem

In any data-driven organization, a significant gap exists between domain experts (who have questions) and the data itself (which requires SQL to access). This bottleneck limits data accessibility, slows down decision-making, and places a heavy burden on data analytics teams.

## 💡 The Solution

This application bridges that gap by providing a conversational interface for data retrieval. It empowers non-technical users to query a `walmart_sales` database in plain English.

The key to its accuracy is a **semantic few-shot prompting** mechanism. Instead of just basic instructions, the AI is dynamically fed relevant examples from a vector database. This "in-context learning" allows the model to handle ambiguous column names and complex, multi-part questions with significantly greater success than standard prompting.

## ✨ Core Features

* **Intuitive Natural Language Interface:** Users can ask questions like "Compare the average gross income between male and female customers" or "What were the top 3 product lines by total revenue in January?"
* **Advanced LLM-Powered Translation:** Utilizes Google's `gemini-2.5-flash` model via `langchain-google-genai` for a balance of high performance, speed, and cost-efficiency.
* **Semantic Few-Shot Prompting:** Implements a `ChromaDB` vector store with `Sentence-Transformer` embeddings. When a user asks a question, the system first finds the most similar *question-query* pairs from its example set and injects them into the prompt, guiding the LLM to produce a correct and syntactically valid MySQL query.
* **Interactive Gradio Dashboard:** The application is wrapped in a clean, user-friendly Gradio UI that displays the generated SQL, the raw data results in a table, and a final, clear natural language explanation of the findings.
* **Scalable & Modular Architecture:** The code is logically separated into modules for configuration (`config.py`), database connection (`database.py`), LLM setup (`llm_setup.py`), core logic (`chain_logic.py`), and the UI (`app.py`), making it easy to maintain, debug, and extend.

## 🛠️ Technical Architecture

The application's request-response pipeline works as follows:

1.  **UI (Gradio):** The user submits a question through the `app.py` interface.
2.  **Semantic Example Selector:** The question is vectorized, and `ChromaDB` performs a similarity search to retrieve the `k` most relevant few-shot examples.
3.  **Dynamic Prompt Assembly:** A `FewShotPromptTemplate` in `chain_logic.py` constructs the full prompt, injecting the user's question, the retrieved examples, and the database schema.
4.  **LLM (Gemini):** The `create_sql_query_chain` invokes the Gemini model to generate the SQL query.
5.  **Database Execution:** The generated query is safely executed against the MySQL database using LangChain's `SQLDatabase` utility.
6.  **Response Synthesis:** The raw data result is passed back to the LLM, which generates a final, easy-to-understand natural language answer.
7.  **UI (Gradio):** The generated SQL, the data table, and the final answer are all displayed to the user.

