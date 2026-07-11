import os
import json
import pandas as pd
from datetime import datetime

# Import LLM and Tools
from core.agent.llm import call_llm, get_session_history, add_message_to_history
import core.agent.tools as tools
from core.forecasting.preprocessor import clean_dataset

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
        df = pd.read_csv(ACTIVE_DATASET_PATH)
        df_clean = clean_dataset(df)
        unique_prods = df_clean[['product_id', 'product_name', 'category']].drop_duplicates()
        
        lookup_lines = []
        for _, row in unique_prods.iterrows():
            lookup_lines.append(f"- ID: '{row['product_id']}' | Name: '{row['product_name']}' | Category: '{row['category']}'")
        return "\n".join(lookup_lines)
    except Exception as e:
        return f"Error building product list: {str(e)}"

def run_agent_query(user_message: str, session_id: str) -> str:
    """
    Executes the AI Analyst workflow:
    1. Classifies user query into tool selection + arguments.
    2. Executes the selected tool.
    3. Feeds structured tool data + memory history to LLM for final response.
    """
    product_lookup = get_product_lookup_str()
    
    # --- STEP 1: REQUEST CLASSIFICATION ---
    classifier_system_prompt = f"""
You are the routing and classification agent for InsightForge (AI Retail Decision Support System).
Your task is to analyze the user's query and decide if we need to call a backend tool, and extract its arguments.

AVAILABLE BACKEND TOOLS:
1. `top_selling_products(limit: int)`: Use when asked about highest selling, most popular, or top products by volume. (Default limit: 5)
2. `low_stock_products(threshold_days: int)`: Use when asked about items running low, running out of stock, needing replenishment, or restock alerts. (Default threshold_days: 10)
3. `inventory_health()`: Use when asked about overall stock status, overstock rate, understock rate, or inventory health count.
4. `sales_summary()`: Use when asked for general sales volume, estimated revenues, and product category breakdowns.
5. `compare_sales(product_id: str)`: Use when comparing sales of a specific product over time (e.g. this month vs last month).
6. `forecast_product(product_id: str)`: Use when asked to show, explain, or forecast future demand/sales for a specific product.
7. `model_comparison(product_id: str)`: Use when asked why a model was selected, or to compare metrics (MAE, MAPE, R2) for a specific product.
8. `generate_business_insights()`: Use when asked for business advice, slow-moving items, general recommendations, or business insights.

PRODUCT LOOKUP DICTIONARY (Match queries to these exact IDs):
{product_lookup}

ROUTING RULES:
- Output a single JSON object with keys: "tool_name" and "arguments".
- "tool_name" MUST be one of the 8 tools listed above, or "none" if the query is a general greeting, about machine learning concepts, or general help.
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
        raw_classification = call_llm(classification_messages, require_json=True)
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
            if tool_name == "top_selling_products":
                limit = int(tool_args.get("limit", 5))
                tool_output = tools.top_selling_products(limit=limit)
            elif tool_name == "low_stock_products":
                days = int(tool_args.get("threshold_days", 10))
                tool_output = tools.low_stock_products(threshold_days=days)
            elif tool_name == "inventory_health":
                tool_output = tools.inventory_health()
            elif tool_name == "sales_summary":
                tool_output = tools.sales_summary()
            elif tool_name == "compare_sales":
                pid = str(tool_args.get("product_id", ""))
                tool_output = tools.compare_sales(product_id=pid)
            elif tool_name == "forecast_product":
                pid = str(tool_args.get("product_id", ""))
                tool_output = tools.forecast_product(product_id=pid)
            elif tool_name == "model_comparison":
                pid = str(tool_args.get("product_id", ""))
                tool_output = tools.model_comparison(product_id=pid)
            elif tool_name == "generate_business_insights":
                tool_output = tools.business_insights()
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
1. Always base your numbers, statistics, and forecasts on the provided "Structured Data Context". Do not invent numbers.
2. If a tool was executed and returned data, translate that structured JSON data into a clear, friendly, and professional natural language response.
3. Keep recommendations strictly evidence-based. Avoid speculation. Use standard business and ML terms.
4. When explaining metrics (like MAE, MAPE, R²), explain them simply so a store manager can understand them.
5. If the structured context indicates an error or no active dataset, politely tell the user to upload a dataset or load the demo.
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
        final_response = call_llm(final_messages)
    except Exception as e:
        final_response = (
            f"I apologize, but I could not contact the LLM completions service. "
            f"Here is the raw data computed from our backend tools instead:\n"
            f"{json.dumps(tool_output, indent=2) if tool_output else 'No data context available.'}"
        )
        
    # Update session memory
    add_message_to_history(session_id, "user", user_message)
    add_message_to_history(session_id, "assistant", final_response)
    
    return final_response
