import os
import json
from datetime import datetime

# Import LLM and Tools
from core.agent.llm import call_llm, get_session_history, add_message_to_history
import core.agent.tools as tools
from routers.dataset import get_clean_df

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIVE_DATASET_PATH = os.path.join(BASE_DIR, "data", "active_dataset.csv")

def get_product_lookup_str() -> str:
    """
    Scans the active dataset and builds a product-name-to-ID lookup string
    to guide the LLM classifier when matching product queries.
    """
    if not os.path.exists(ACTIVE_DATASET_PATH):
        return "No active dataset loaded."
        
    try:
        df_clean = get_clean_df()
        # Dedupe on product_id alone (keeping the first name/category seen for that ID).
        # Some datasets have inconsistent category values for the same product_id across
        # different rows (e.g. category recorded per-transaction instead of per-product);
        # deduping on all three columns then produces one lookup line per (id, name, category)
        # combination instead of one per product — with enough products this bloats the
        # classifier's prompt to the point where the LLM stops routing to tools correctly.
        unique_prods = df_clean[['product_id', 'product_name', 'category']].drop_duplicates(subset=['product_id'])

        lookup_lines = []
        for _, row in unique_prods.iterrows():
            lookup_lines.append(f"- ID: '{row['product_id']}' | Name: '{row['product_name']}' | Category: '{row['category']}'")
        return "\n".join(lookup_lines)
    except Exception as e:
        return f"Error building product list: {str(e)}"

def resolve_product_id(input_str: str) -> str:
    """
    Resolves product queries fuzzy-style when the LLM extracts slightly malformed
    names, synonyms, or lowercase references instead of exact IDs.
    Returns the mapped active catalog ID, or the original string as a fallback.
    """
    if not input_str:
        return ""
    input_str = input_str.strip().lower()

    if not os.path.exists(ACTIVE_DATASET_PATH):
        return input_str
        
    try:
        df_clean = get_clean_df()
        df_unique = df_clean[['product_id', 'product_name']].drop_duplicates()
        
        # 1. Exact case-insensitive match on product_id
        for _, row in df_unique.iterrows():
            pid = str(row['product_id'])
            if pid.lower() == input_str:
                return pid
                
        # 2. Case-insensitive substring match (e.g. "milk" -> "PRD_01")
        for _, row in df_unique.iterrows():
            pid = str(row['product_id'])
            pname = str(row['product_name']).lower()
            if input_str in pname or pname in input_str:
                return pid
                
        # 3. Mapped ID within the input string
        for _, row in df_unique.iterrows():
            pid = str(row['product_id'])
            if pid.lower() in input_str:
                return pid
    except Exception:
        pass
        
    return input_str

async def run_agent_query(user_message: str, session_id: str) -> dict:
    """
    Executes the AI Analyst workflow:
    1. Classifies user query into tool selection + arguments.
    2. Executes the selected tool.
    3. Feeds structured tool data + memory history to LLM for final response.

    Returns {"response": str, "chart": dict | None} — chart is populated only when the
    classifier routed to generate_chart_spec and it succeeded; the LLM never sees or
    touches the chart's data, only its own natural-language caption.
    """
    product_lookup = get_product_lookup_str()
    
    # --- STEP 1: REQUEST CLASSIFICATION ---
    classifier_system_prompt = f"""
You are the routing and classification agent for InsightForge (AI Retail Decision Support System).
Your task is to analyze the user's query and decide if we need to call a backend tool, and extract its arguments.

AVAILABLE BACKEND TOOLS:
1. `list_products()`: Use when asked what products, items, categories, or catalog options are available to forecast or audit.
2. `top_selling_products(limit: int)`: Use ONLY when the user wants a plain-text LIST or ranking with no visual requested (e.g. "which products sell best?", "what are my top sellers?"). If the phrasing includes "show", "chart", "graph", "plot", "visualize", or names a chart type, use `generate_chart_spec` instead — see tool 11. (Default limit: 5)
3. `low_stock_products(threshold_days: int)`: Use when asked about items running low, running out of stock, needing replenishment, or restock alerts. (Default threshold_days: 10)
4. `inventory_health()`: Use when asked about overall stock status, overstock rate, understock rate, or inventory health count.
5. `sales_summary()`: Use when asked for general sales volume, estimated revenues, and product category breakdowns.
6. `compare_sales(product_id: str)`: Use when comparing sales of a specific product over time (e.g. this month vs last month).
7. `forecast_product(product_id: str)`: Use when asked to show or forecast future sales/predictions for a specific product.
8. `explain_forecast_decomposition(product_id: str)`: Use when asked what drives the forecast, what factors influence demand, how much promotions or weekly seasonality affect future sales, or to show a forecast breakdown.
9. `model_comparison(product_id: str)`: Use when asked why a model was selected, or to compare validation metrics (MAE, MAPE, R2) for a specific product.
10. `generate_business_insights()`: Use when asked for business advice, slow-moving items, general recommendations, or business insights.
11. `generate_chart_spec(chart_type, metric, dimension, product_names, categories, recent_days, limit)`: PREFER THIS TOOL whenever the phrasing implies a visual — "show", "plot", "chart", "graph", "visualize", "compare X and Y [sales/revenue]", or names a chart type like "pie chart"/"bar chart". This takes priority over tools 2/5/6 when both could apply. Examples: "show me a chart of...", "plot revenue by category", "compare Milk and Bread sales", "pie chart of category revenue", "show top 10 products", "show revenue for the last 3 months". Arguments:
    - chart_type: one of "bar", "line", "donut", "area" (default "bar"; a "pie chart" request means "donut")
    - metric: one of "units_sold", "revenue", "stock", "profit" (default "units_sold"; "profit and loss"/"profit margin" requests mean "profit")
    - dimension: one of "product", "category", "date", "day_of_week", "month" (default "category") — what the chart is broken down BY
    - product_names: array of product names/references mentioned by the user, e.g. ["Milk", "Bread"] — empty array if none mentioned
    - categories: array of category names mentioned by the user — empty array if none mentioned
    - recent_days: integer for date-relative requests (e.g. "last 3 months" -> 90, "last 30 days" -> 30), or null if no time range was mentioned
    - limit: integer, default 10, for "top N" requests

PRODUCT LOOKUP DICTIONARY (Match queries to these exact IDs):
{product_lookup}

ROUTING RULES:
- Output a single JSON object with keys: "tool_name" and "arguments".
- "tool_name" MUST be one of the 11 tools listed above, or "none" if the query is a general greeting, about machine learning concepts, or general help.
- "arguments" is a dictionary of parameters for the tool. For example, if product is milk, map it to ID 'PRD_01' based on the lookup list.
- If the user asks about 'this product' or 'the current product' and didn't specify one, check if a product is mentioned in your lookup or default to the top-selling product ID.

OUTPUT FORMAT (JSON ONLY, do not explain):
{{"tool_name": "forecast_product", "arguments": {{"product_id": "PRD_01"}}}}
OR
{{"tool_name": "none", "arguments": {{}}}}
"""
    
    # We call the classifier. Since it needs to be 100% JSON, we request JSON mode.
    classification_messages = [
        {"role": "system", "content": classifier_system_prompt},
        {"role": "user", "content": f"Query to classify: {user_message}"}
    ]
    
    try:
        raw_classification = await call_llm(classification_messages, require_json=True)
        # Parse JSON
        classification = json.loads(raw_classification)
        tool_name = classification.get("tool_name", "none")
        tool_args = classification.get("arguments", {})
    except Exception as e:
        print(f"Classification failed, falling back to none. Error: {str(e)}")
        tool_name = "none"
        tool_args = {}
        
    # --- STEP 2: TOOL EXECUTION ---
    tool_output = None
    if tool_name != "none":
        print(f"Agent classified query. Executing tool '{tool_name}' with args '{tool_args}'...")
        try:
            # Map tool name to function execution
            if tool_name == "list_products":
                tool_output = tools.list_products()
            elif tool_name == "top_selling_products":
                limit = int(tool_args.get("limit", 5))
                tool_output = tools.top_selling_products(limit=limit)
            elif tool_name == "low_stock_products":
                days = int(tool_args.get("threshold_days", 10))
                tool_output = tools.low_stock_products(threshold_days=days)
            elif tool_name == "inventory_health":
                tool_output = tools.inventory_health()
            elif tool_name == "sales_summary":
                tool_output = tools.sales_summary()
            elif tool_name in ["compare_sales", "forecast_product", "model_comparison", "explain_forecast_decomposition"]:
                pid = resolve_product_id(str(tool_args.get("product_id", "")))
                
                if tool_name == "compare_sales":
                    tool_output = tools.compare_sales(product_id=pid)
                elif tool_name == "forecast_product":
                    tool_output = tools.forecast_product(product_id=pid)
                elif tool_name == "model_comparison":
                    tool_output = tools.model_comparison(product_id=pid)
                elif tool_name == "explain_forecast_decomposition":
                    tool_output = tools.explain_forecast_decomposition(product_id=pid)
            elif tool_name == "generate_business_insights":
                tool_output = tools.generate_business_insights()
            elif tool_name == "generate_chart_spec":
                # Resolve any free-text product names the classifier extracted into real
                # catalog IDs, same fuzzy resolver used for every other product-aware tool —
                # tools.py itself only ever accepts already-resolved IDs.
                raw_names = tool_args.get("product_names") or []
                resolved_ids = [resolve_product_id(str(n)) for n in raw_names if str(n).strip()]
                tool_output = tools.generate_chart_spec(
                    chart_type=str(tool_args.get("chart_type", "bar")),
                    metric=str(tool_args.get("metric", "units_sold")),
                    dimension=str(tool_args.get("dimension", "category")),
                    product_ids=resolved_ids or None,
                    categories=tool_args.get("categories") or None,
                    recent_days=tool_args.get("recent_days"),
                    limit=tool_args.get("limit", 10),
                )
            else:
                tool_output = {
                    "status": "error",
                    "message": f"Tool '{tool_name}' is not supported by the backend."
                }
        except Exception as e:
            print(f"Tool execution failed: {str(e)}")
            tool_output = {"status": "error", "message": f"Failed to execute tool {tool_name}: {str(e)}"}
            
    # --- STEP 3: FINAL RESPONSE GENERATION ---
    # Retrieve short-term memory history
    history = get_session_history(session_id)
    
    # Construct final LLM prompt
    analyst_system_prompt = """
You are the Lead AI Retail Analyst for the InsightForge Decision Support System.
Your job is to answer user questions about sales, inventory, and demand forecasts.

RULES:
1. Always base your numbers, statistics, and forecasts on the provided "Structured Data Context". Do not invent/hallucinate numbers.
2. If a tool was executed and returned data, translate that structured JSON data into a clear, friendly, and professional natural language response.
3. Keep recommendations strictly evidence-based. Do not speculate on external factors (such as competitor actions, consumer tastes, or economic conditions) that are not present in the data.
4. When explaining a forecast decomposition (trend, seasonality, promo percentages), use the exact calculations provided in the context. Explain them clearly, highlighting which driver is the strongest.
5. When explaining metrics (like MAE, MAPE, R²), reference the 'human_friendly_explanation' template in the context to explain what they mean in simple business terms.
6. If the structured context indicates an error or no active dataset, politely tell the user to upload a dataset or load the demo.
7. If the context's metric_type is "chart_spec", a chart is already being rendered separately in the UI — write only a short (1-2 sentence) caption highlighting the standout value(s), do not restate every number from the chart.
8. Default to plain business language, not ML/statistics jargon — say "how far off predictions typically are" rather than "MAE", "how much of the demand pattern the model explains" rather than "R²", unless the user's own question uses that technical term or specifically asks how the model works. You are talking to a retail manager making a decision, not a data scientist.
9. This is an Indian retail business. Always write currency amounts with the ₹ symbol (e.g. ₹1,070,847.06), never $ or USD.
"""
    
    # Formulate prompt messages
    final_messages = [
        {"role": "system", "content": analyst_system_prompt}
    ]
    
    # Add history
    for msg in history:
        final_messages.append(msg)
        
    # Add current context
    context_str = ""
    if tool_output:
        context_str = f"\n\n[Structured Data Context from Backend Tool: '{tool_name}']\n{json.dumps(tool_output, indent=2)}"
    else:
        context_str = "\n\n[Structured Data Context]: No backend tool was called. Answer general retail or forecasting questions."
        
    final_messages.append({"role": "user", "content": f"{user_message}{context_str}"})
    
    # Call final summarization
    try:
        final_response = await call_llm(final_messages)
    except Exception as e:
        final_response = (
            f"I apologize, but I could not contact the LLM completions service. "
            f"Here is the raw data computed from our backend tools instead:\n"
            f"{json.dumps(tool_output, indent=2) if tool_output else 'No data context available.'}"
        )
        
    # Update session memory
    add_message_to_history(session_id, "user", user_message)
    add_message_to_history(session_id, "assistant", final_response)

    # Pass the chart spec straight through from the tool's real output to the API
    # response — never round-tripped through the LLM, so it can't be altered/fabricated.
    chart = None
    if tool_name == "generate_chart_spec" and isinstance(tool_output, dict) and tool_output.get("status") == "success":
        chart = tool_output.get("chart")

    return {"response": final_response, "chart": chart}
