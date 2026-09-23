# Intelligent Analytics Query Engine

This project builds an intelligent analytics assistant that converts natural-language questions into SQL-like analytical logic, runs them against a sales dataset, and returns a structured result with a confidence score and explanation.

## Project goal

The system is designed to help users ask business questions like:

- What is the total revenue by region?
- Which product category contributes the most revenue?
- Compare APAC revenue against target for February.
- Show the top 2 categories in each region by revenue.

The engine maps the question to a dataset-aware SQL/query pattern, calculates the result, and returns a useful explanation.

## Features

- Natural-language to analytical logic conversion
- SQL-style generation for common business queries
- Execution over the supplied dataset
- Confidence score per output
- Explanation of query mapping and result generation
- Feedback-aware confidence scoring
- Optional OpenAI-based explanation layer
- Manual interactive command-line mode
- Simple Flask web UI for browser-based usage

## Folder structure

- `analytics_engine.py` — main logic for query parsing, SQL generation, execution, and explanation
- `main.py` — entry point for CLI execution
- `web_app.py` — Flask web application
- `app.py` — simple app wrapper
- `templates/index.html` — browser interface
- `static/styles.css` — UI styling
- `dataset/` — all data files used by the engine
- `tests/test_engine.py` — validation checks

## Dataset files

- `dataset/sales_data.csv`
- `dataset/targets.csv`
- `dataset/data_dictionary.json`
- `dataset/nl_queries.json`
- `dataset/feedback_log.csv`

## Setup

1. Open the project folder.
2. Install dependencies:

```bash
C:/Users/acer/AppData/Local/Programs/Python/Python314/python.exe -m pip install pandas openai python-dotenv flask pytest
```

3. Add your API key in the `.env` file if needed:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

A sample template is also provided in `.env.example`.

## Run from terminal

### Sample queries

```bash
cd "C:\Users\acer\Desktop\IIC project"
C:/Users/acer/AppData/Local/Programs/Python/Python314/python.exe main.py
```

Then choose:
- `1` for sample queries
- `2` for manual interactive chat

### Manual interactive chat

Type any sales question such as:

```text
Which product category contributes the most revenue?
```

The system will generate SQL, execute it, and show the result.

## Run the web UI

```bash
cd "C:\Users\acer\Desktop\IIC project"
C:/Users/acer/AppData/Local/Programs/Python/Python314/python.exe web_app.py
```

Then open:

```text
http://localhost:5000/
```

## Example output

```json
{
  "query": "What is the total revenue by region?",
  "generated_sql": "SELECT region, ROUND(SUM(quantity * unit_price * (1 - discount)), 2) AS revenue FROM sales_data GROUP BY region ORDER BY revenue DESC;",
  "result": [
    {"region": "NA", "revenue": 3262.4},
    {"region": "EMEA", "revenue": 2398.0},
    {"region": "APAC", "revenue": 474.0}
  ],
  "confidence_score": 0.94,
  "explanation": "The chatbot mapped the request to a region-level revenue aggregation. It converted each row to revenue using quantity × unit_price × (1 - discount), then grouped by region."
}
```

## Notes on GenAI usage

The project uses OpenAI only as an optional enhancement layer, mainly for SQL generation and high-level natural-language explanations. If the API is unavailable, rate-limited, or out of credit, the app falls back to local rule-based logic and still works correctly on the provided dataset.

## Validation

The implementation has been verified with the project test suite:

```bash
cd "C:\Users\acer\Desktop\IIC project"
C:/Users/acer/AppData/Local/Programs/Python/Python314/python.exe -m pytest -q
```

This currently passes successfully.

## Optional improvements if given more time

- Add a larger schema-aware SQL generator
- Add more synonyms and query normalization
- Add authentication and API endpoints
- Improve multi-table joins and nested business logic
- Add deployment support for Streamlit or FastAPI
- Support CSV export of results
