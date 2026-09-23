import json

from analytics_engine import QueryEngine


def test_region_revenue_query():
    engine = QueryEngine(data_dir="dataset")
    output = engine.answer_query("What is the total revenue by region?")
    assert output["result"]["APAC"] == 474.0
    assert output["result"]["EMEA"] == 2398.0
    assert output["confidence_score"] >= 0.8


def test_top_category_query():
    engine = QueryEngine(data_dir="dataset")
    output = engine.answer_query("Which product category contributes the most revenue?")
    assert output["result"]["category"] == "Technology"
    assert output["result"]["revenue"] == 5402.0


def test_target_comparison_query():
    engine = QueryEngine(data_dir="dataset")
    output = engine.answer_query("Compare APAC revenue against target for February.")
    assert output["result"]["actual_revenue"] == 75.0
    assert output["result"]["target_revenue"] == 6000
    assert output["result"]["status"] == "Below target"


def test_top_n_within_group_query():
    engine = QueryEngine(data_dir="dataset")
    output = engine.answer_query("Show the top 2 product categories in each region by revenue.")
    assert isinstance(output["result"], dict)
    assert "APAC" in output["result"]
    assert len(output["result"]["APAC"]) == 2


def test_batch_queries():
    engine = QueryEngine(data_dir="dataset")
    queries = json.load(open("dataset/nl_queries.json"))
    responses = engine.answer_queries(queries)
    assert len(responses) == len(queries)
    assert all(item["confidence_score"] >= 0.0 for item in responses)


def test_sql_generation_and_execution():
    engine = QueryEngine(data_dir="dataset")
    sql = engine.generate_sql_for_query("What is the total revenue by region?")
    assert "SELECT" in sql.upper()
    rows = engine.execute_generated_sql("What is the total revenue by region?")
    assert isinstance(rows, list)
    assert len(rows) >= 1


def test_chat_answer_format():
    engine = QueryEngine(data_dir="dataset")
    response = engine.chat_answer("Which product category contributes the most revenue?")
    assert "generated_sql" in response
    assert "result" in response
    assert response["confidence_score"] >= 0.0
