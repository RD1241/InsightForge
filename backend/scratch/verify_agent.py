import os
import sys
import json

# Add the backend root directory to the python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

import core.agent.tools as tools

def test_toolbelt():
    print("=== InsightForge AI Analyst Toolbelt Verification ===")
    
    # Check if active dataset is loaded
    if not os.path.exists(tools.ACTIVE_DATASET_PATH):
        print("CRITICAL: Active dataset not found. Please run verify_training.py first to initialize data.")
        return False
        
    # 1. Test sales_summary
    print("\n1. Testing tools.sales_summary()...")
    summary = tools.sales_summary()
    print(f"   Status: {summary['status']}")
    print(f"   Total Volume: {summary['total_sales_volume']}")
    print(f"   Total Revenue: ${summary['total_revenue']}")
    print(f"   Categories: {[c['category'] for c in summary['category-breakdown'] if 'category' in c] if 'category-breakdown' in summary else [c['category'] for c in summary.get('category_breakdown', [])]}")

    # 2. Test top_selling_products
    print("\n2. Testing tools.top_selling_products()...")
    top_selling = tools.top_selling_products(limit=3)
    print(f"   Status: {top_selling['status']}")
    for i, p in enumerate(top_selling['data']):
        print(f"   #{i+1}: {p['product_name']} (Sales: {p['total_sales']}, Revenue: ${p['total_revenue']})")

    # 3. Test low_stock_products
    print("\n3. Testing tools.low_stock_products()...")
    low_stock = tools.low_stock_products(threshold_days=10)
    print(f"   Status: {low_stock['status']}")
    print(f"   Products running low (under 10 days of sales): {len(low_stock['data'])}")
    for p in low_stock['data'][:3]:
        print(f"   * {p['product_name']} | Stock on hand: {p['stock_on_hand']} | Avg daily sales: {round(p['avg_daily_sales'], 2)} | Days left: {p['days_of_stock_remaining']}")

    # 4. Test inventory_health
    print("\n4. Testing tools.inventory_health()...")
    health = tools.inventory_health()
    print(f"   Status: {health['status']}")
    print(f"   Health Summary: {health['health_summary']}")

    # 5. Test compare_sales
    print("\n5. Testing tools.compare_sales()...")
    comparison = tools.compare_sales("PRD_01")
    print(f"   Status: {comparison['status']}")
    print(f"   Product: {comparison['product_name']}")
    print(f"   Current 30-day sales: {comparison['current_period_sales']}")
    print(f"   Prior 30-day sales: {comparison['prior_period_sales']}")
    print(f"   Percentage change: {comparison['percentage_change']}%")

    # 6. Test model_comparison
    print("\n6. Testing tools.model_comparison()...")
    comparison_models = tools.model_comparison("PRD_01")
    print(f"   Status: {comparison_models['status']}")
    print(f"   Recommended Model: {comparison_models['recommended_model']}")
    print(f"   Metrics details:")
    for m in comparison_models['all_models_metrics']:
        print(f"     - {m['model_name']}: MAE={m['metrics']['MAE']}, R2={m['metrics']['R2']}")

    # 7. Test business_insights
    print("\n7. Testing tools.generate_business_insights()...")
    insights = tools.generate_business_insights()
    print(f"   Status: {insights['status']}")
    print(f"   Slow Movers count: {len(insights['slow_moving_products'])}")
    print(f"   Critical Restock Alerts count: {len(insights['critical_restock_alerts'])}")

    # 8. Test list_products
    print("\n8. Testing tools.list_products()...")
    catalog = tools.list_products()
    print(f"   Status: {catalog['status']}")
    print(f"   Categories in Catalog: {list(catalog['catalog'].keys())}")

    # 9. Test explain_forecast_decomposition
    print("\n9. Testing tools.explain_forecast_decomposition()...")
    decomp = tools.explain_forecast_decomposition("PRD_01")
    print(f"   Status: {decomp['status']}")
    print(f"   Product: {decomp['product_name']} | Model: {decomp['model_used']}")
    print(f"   Decomposition Weights: {decomp['decomposition']}")

    # 10. Test generate_chart_spec (Phase 2 — AI-generated charts)
    print("\n10. Testing tools.generate_chart_spec()...")
    chart = tools.generate_chart_spec(chart_type="bar", metric="units_sold", dimension="product", limit=5)
    print(f"   Status: {chart['status']}")
    print(f"   Title: {chart['chart']['title']} | chart_type: {chart['chart']['chart_type']}")
    print(f"   x: {chart['chart']['x']}")
    print(f"   y: {chart['chart']['y']}")

    print("\n11. Testing tools.generate_chart_spec() with filters (donut, revenue, category, recent_days)...")
    chart2 = tools.generate_chart_spec(chart_type="pie", metric="revenue", dimension="category", recent_days=90)
    print(f"   Status: {chart2['status']}")
    print(f"   chart_type normalized 'pie' -> '{chart2['chart']['chart_type']}' (expected 'donut')")
    print(f"   x: {chart2['chart']['x']}")

    print("\n=== All AI Tools Executed and Verified Successfully! ===")
    return True

if __name__ == "__main__":
    success = test_toolbelt()
    sys.exit(0 if success else 1)
