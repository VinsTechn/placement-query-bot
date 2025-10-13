from groq import Groq
import os
import re
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()

GROQ_MODEL = os.getenv('GROQ_MODEL')

db_path = "placement_data.db"

client_sql = Groq()

sql_prompt = """You are an expert in understanding the database schema and generating a single SQLite SQL query for a natural language question about the data. The schema is provided in the schema tags. Always return ONLY the SQL query wrapped between <SQL>...</SQL> tags and nothing else.

<schema>
table: placements

fields:
Company - string (name of the company)
AIML - integer (number of AIML students placed)         # count
CSE - integer (number of CSE students placed)           # count
ISE - integer (number of ISE students placed)           # count
ECE - integer (number of ECE students placed)           # count
EEE - integer (number of EEE students placed)           # count
Mech - integer (number of Mech students placed)         # count
Total Placed - integer (total number of students placed) # count
Salary LPA - float (salary/package for that placement record, in LPA) # numeric
year - integer (placement year)                         # numeric
</schema>

Important interpretation rules (use these to decide which SQL to generate):
1. Branch columns (AIML, CSE, ISE, ECE, EEE, Mech) are COUNTS (how many students from that branch were placed for that company/row). They are NOT salary values.
2. "Salary LPA" is the salary value (float). When the user asks about an **average package for a particular branch** (e.g., "avg placement for CSE in 2023", "average package for CSE 2023", "avg salary for CSE"), compute the **weighted average package** for that branch:
   - Weighted average formula: SUM("Salary LPA" * <branch_col>) / SUM(<branch_col>)
   - Guard against division-by-zero using CASE WHEN SUM(<branch_col>) = 0 THEN NULL ELSE ... END.
3. When the user explicitly asks for **counts/placements** (words: "placements", "placed", "how many", "total placed for CSE"), use SUM(<branch_col>) (or SUM("Total Placed") for overall placements). If the user asks for "average placements" of counts, return AVG(<branch_col>).
4. If the user asks for a general "average salary" or "avg package" across companies (no branch mentioned), use AVG("Salary LPA"). If they explicitly request weighting by number of students, use SUM("Salary LPA" * "Total Placed") / SUM("Total Placed").
5. Map common synonyms of branches case-insensitively (e.g., 'cse' → CSE, 'aiml' or 'ai/ml' → AIML, 'mechanical' → Mech).
6. For company name matching, always use case-insensitive pattern matching with SQL `LIKE` and wildcards: e.g. `Company LIKE '%<value>%'`. **Do not use ILIKE**. Use `%LIKE%` semantics in instructions to the model.
7. When column names include spaces (e.g., `Total Placed`, `Salary LPA`), always quote them using double quotes in SQL: e.g., `"Total Placed"`, `"Salary LPA"`.
8. Use SQLite-compatible SQL. Return a single query only. Use aggregation functions (SUM, AVG) and CASE WHEN for safety as needed.

Formatting rules for generated SQL:
- If the question requests an aggregate/summary (avg, sum, count), the SELECT should contain the appropriate aggregate expressions (and optional alias), e.g. `SELECT CASE WHEN SUM(CSE)=0 THEN NULL ELSE SUM("Salary LPA"*CSE)/SUM(CSE) END AS avg_salary_cse FROM placements WHERE year = 2023;`
- If the question requests row-level details or filtering (no aggregation asked), use `SELECT * FROM placements WHERE ...` with appropriate WHERE conditions.
- Always reference the table name `placements`.
- Use a single SQL statement (no commentary, no multiple statements).

Examples (these illustrate the intended behavior):

1) User: "avg placement for cse in 2023"
   -> Interpret as average package for CSE students in 2023 (weighted by CSE counts).
   Example SQL to produce:
   <SQL>SELECT CASE WHEN SUM(CSE) = 0 THEN NULL ELSE SUM("Salary LPA" * CSE) * 1.0 / SUM(CSE) END AS avg_salary_cse FROM placements WHERE year = 2023;</SQL>

2) User: "total placements for CSE in 2022"
   -> Interpret as total number of CSE students placed in 2022.
   Example SQL to produce:
   <SQL>SELECT SUM(CSE) AS total_cse_placements FROM placements WHERE year = 2022;</SQL>

3) User: "list companies where salary is above 20 LPA in 2024"
   -> Row-level result; produce SELECT * with a WHERE filter:
   <SQL>SELECT * FROM placements WHERE "Salary LPA" > 20 AND year = 2024;</SQL>

Follow the interpretation rules above strictly so column semantics (counts vs. salary) are used correctly."""




comprehension_prompt = """You are an expert in understanding the context of the question and replying naturally based on the placement data provided. 
You will be given a QUESTION: and DATA:. The data will be in the form of a number, record, array, or dataframe. 
Always answer only from the given DATA and nothing else. 

Guidelines:
1. Do not use technical phrases like "Based on the data". Just answer directly in plain natural language. 
2. If the answer is a single number (e.g., average salary, total placements), phrase it in context with the question. 
   Example: 
   - Question: "What is the average package for CSE in 2023?" 
   - Data: "6.8" 
   - Response: "The average package for CSE in 2023 was 6.8 LPA."
3. If the DATA is a row or multiple rows (dataframe/dict/array), list them in a clean and human-readable format. Each company should be on its own line. 
   Include company name, salary offered, placements (by branch if relevant), total placed, and year if available.
   Example:
   1. Company: Infosys | Salary: 6.5 LPA | CSE: 40 | AIML: 12 | Total Placed: 52 | Year: 2023  
   2. Company: TCS | Salary: 7.2 LPA | CSE: 30 | ISE: 10 | Total Placed: 40 | Year: 2023
4. If only some columns are present in the DATA, include only those available (do not invent missing details). 
5. Numbers should be expressed with proper units: "students" for counts, "LPA" for salary, and mention year if it exists in the row. 
6. Always be natural and concise. Do not repeat the question, only answer it directly in a way that makes sense to a student or recruiter reading it.
"""



def generate_sql_query(question):
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sql_prompt,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        model=os.environ['GROQ_MODEL'],
        temperature=0.2,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



def run_query(query):
    if query.strip().upper().startswith('SELECT'):
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df


def data_comprehension(question, context):
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": comprehension_prompt,
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}. DATA: {context}",
            }
        ],
        model=os.environ['GROQ_MODEL'],
        temperature=0.2,
        # max_tokens=1024
    )

    return chat_completion.choices[0].message.content



def sql_chain(question):
    sql_query = generate_sql_query(question)
    pattern = "<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, sql_query, re.DOTALL)

    if len(matches) == 0:
        return "Sorry, LLM is not able to generate a query for your question"

    #print(matches[0].strip())

    response = run_query(matches[0].strip())
    if response is None:
        return "Sorry, there was a problem executing SQL query"

    context = response.to_dict(orient='records')

    answer = data_comprehension(question, context)
    return answer


if __name__ == "__main__":
    while True:
        question=input("Ask bot: ")
        if question=='exit':
            break
        answer=sql_chain(question)
        print(answer)
