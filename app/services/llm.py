import os
import google.generativeai as genai
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX

def build_llm_and_prompt(db):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing")
    genai.configure(api_key=api_key)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    few_shots = [
        {'Question': "Find total revenue generated across all transactions.",
         'SQLQuery': "SELECT ROUND(SUM(Total), 2) AS total_revenue FROM walmart_sales;",
         'SQLResult': "Result of the SQL query",
         'Answer': "The total revenue across all transactions is calculated by summing the Total column."},
        {'Question': "Count the total number of sales transactions per branch.",
         'SQLQuery': "SELECT Branch, COUNT(`Invoice ID`) AS total_transactions FROM walmart_sales GROUP BY Branch;",
         'SQLResult': "Result of the SQL query",
         'Answer': "The number of transactions per branch is found by counting Invoice IDs grouped by Branch."},
        {'Question': "Calculate average customer rating per product line.",
         'SQLQuery': "SELECT `Product line`, ROUND(AVG(Rating), 2) AS avg_rating FROM walmart_sales GROUP BY `Product line`;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Average rating per product line is calculated by averaging Rating grouped by Product line."},
        {'Question': "Find the number of transactions done using each payment method.",
         'SQLQuery': "SELECT Payment, COUNT(*) AS total_payments FROM walmart_sales GROUP BY Payment;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Transactions per payment method by counting and grouping by Payment."},
        {'Question': "Retrieve total quantity sold for each product line.",
         'SQLQuery': "SELECT `Product line`, SUM(Quantity) AS total_quantity FROM walmart_sales GROUP BY `Product line`;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Sum Quantity grouped by Product line."},
        {'Question': "Find average revenue per transaction for each branch.",
         'SQLQuery': "SELECT Branch, ROUND(SUM(Total)/COUNT(`Invoice ID`), 2) AS avg_revenue_per_txn FROM walmart_sales GROUP BY Branch;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Total sales divided by transaction count per branch."},
        {'Question': "Determine which product line has the highest total sales in each city.",
         'SQLQuery': "SELECT City, `Product line`, SUM(Total) AS total_sales FROM walmart_sales GROUP BY City, `Product line` ORDER BY City, total_sales DESC;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Group by City/Product line, order by total sales."},
        {'Question': "Calculate monthly total sales trend.",
         'SQLQuery': "SELECT DATE_FORMAT(STR_TO_DATE(Date, '%Y-%m-%d'), '%Y-%m') AS month, SUM(Total) AS total_sales FROM walmart_sales GROUP BY month ORDER BY month;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Extract month and sum Total."},
        {'Question': "Find top 3 most popular product lines by number of transactions.",
         'SQLQuery': "SELECT `Product line`, COUNT(*) AS transactions FROM walmart_sales GROUP BY `Product line` ORDER BY transactions DESC LIMIT 3;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Count and order, limit 3."},
        {'Question': "Compare average gross income of male vs. female customers.",
         'SQLQuery': "SELECT Gender, ROUND(AVG(`gross income`), 2) AS avg_gross_income FROM walmart_sales GROUP BY Gender;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Average gross income grouped by Gender."},
        {'Question': "Find top-performing product line per city using ranking.",
         'SQLQuery': "SELECT City, `Product line`, SUM(Total) AS total_revenue, RANK() OVER (PARTITION BY City ORDER BY SUM(Total) DESC) AS rank_in_city FROM walmart_sales GROUP BY City, `Product line`;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Window rank partitioned by City."},
        {'Question': "Identify the hour of the day with the highest total sales for each branch.",
         'SQLQuery': "SELECT Branch, HOUR(STR_TO_DATE(Time, '%H:%i:%s')) AS hour_of_day, SUM(Total) AS total_sales FROM walmart_sales GROUP BY Branch, hour_of_day ORDER BY Branch, total_sales DESC;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Extract hour and sum Total."},
        {'Question': "Calculate revenue contribution percentage by gender and customer type.",
         'SQLQuery': "SELECT Gender, `Customer type`, ROUND(SUM(Total) * 100 / (SELECT SUM(Total) FROM walmart_sales), 2) AS revenue_percent FROM walmart_sales GROUP BY Gender, `Customer type` ORDER BY revenue_percent DESC;",
         'SQLResult': "Result of the SQL query",
         'Answer': "Share of total revenue."},
        {'Question': "Compute monthly sales growth rate percentage over previous month.",
         'SQLQuery': "SELECT DATE_FORMAT(STR_TO_DATE(Date, '%Y-%m-%d'), '%Y-%m') AS month, SUM(Total) AS total_sales, ROUND((SUM(Total) - LAG(SUM(Total)) OVER (ORDER BY DATE_FORMAT(STR_TO_DATE(Date, '%Y-%m-%d'), '%Y-%m'))) / LAG(SUM(Total)) OVER (ORDER BY DATE_FORMAT(STR_TO_DATE(Date, '%Y-%m-%d'), '%Y-%m')) * 100, 2) AS growth_percent FROM walmart_sales GROUP BY month ORDER BY month;",
         'SQLResult': "Result of the SQL query",
         'Answer': "LAG to compare months."},
    ]

    to_vectorize = [" ".join(str(v) for v in ex.values()) for ex in few_shots]
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma.from_texts(texts=to_vectorize, embedding=embeddings, metadatas=few_shots)

    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)

    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.

Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.

Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.

Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run (without any markdown formatting or code blocks)
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only provide the SQL query without any additional text, explanations, or markdown code blocks.
"""

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"]
    )

    query_chain = create_sql_query_chain(llm, db, prompt=few_shot_prompt)
    return llm, query_chain
