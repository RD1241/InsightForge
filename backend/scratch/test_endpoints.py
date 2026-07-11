import time
import httpx
import sys

def run_tests():
    print("=== Testing FastAPI Backend Endpoints ===")
    
    url = "http://127.0.0.1:8000"
    
    # 1. Health check
    try:
        response = httpx.get(f"{url}/api/health", timeout=5.0)
        print(f"Health Check: Status {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"CRITICAL: Failed to connect to server: {str(e)}")
        return False
        
    # 2. Load Demo Dataset
    print("\nTriggering Load Demo Dataset (/api/dataset/load-demo)...")
    try:
        response = httpx.post(f"{url}/api/dataset/load-demo", timeout=15.0)
        print(f"Load Demo Status: {response.status_code}")
        data = response.json()
        print(f"Message: {data.get('message')}")
    except Exception as e:
        print(f"Error loading demo: {str(e)}")
        return False
        
    # 3. Train Models
    print("\nTriggering Model Training Pipeline (/api/forecast/train)...")
    try:
        response = httpx.post(f"{url}/api/forecast/train", timeout=30.0)
        print(f"Train Status: {response.status_code}")
        data = response.json()
        report = data.get("report", {})
        print(f"Message: {data.get('message')}")
        print(f"Products Trained: {report.get('total_products_trained')}")
        print(f"Average MAE: {report.get('average_mae')}")
        print(f"Average MAPE: {report.get('average_mape')}%")
    except Exception as e:
        print(f"Error running training: {str(e)}")
        return False
        
    # 4. Check Report
    print("\nFetching Training Run Report (/api/forecast/report)...")
    try:
        response = httpx.get(f"{url}/api/forecast/report", timeout=5.0)
        print(f"Report Status: {response.status_code}")
        data = response.json()
        print(f"Report Timestamp: {data.get('timestamp')}")
        print(f"Products in Report: {len(data.get('products', {}))}")
    except Exception as e:
        print(f"Error getting report: {str(e)}")
        return False
        
    # 5. Predict Forecast
    print("\nRequesting 30-day Future Forecast for PRD_01 (/api/forecast/predict)...")
    try:
        response = httpx.get(f"{url}/api/forecast/predict?product_id=PRD_01&horizon_days=30", timeout=10.0)
        print(f"Predict Status: {response.status_code}")
        data = response.json()
        print(f"Product Name: {data.get('product_name')}")
        print(f"Model Used: {data.get('model_used')}")
        print(f"First 3 predictions: {data.get('forecast', {}).get('predictions', [])[:3]}")
        print(f"Recommendation Reason: {data.get('recommendation_reason')}")
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        return False
        
    # 6. Compare Models metrics
    print("\nRequesting Model Comparison Metrics for PRD_01 (/api/forecast/compare)...")
    try:
        response = httpx.get(f"{url}/api/forecast/compare?product_id=PRD_01", timeout=5.0)
        print(f"Compare Status: {response.status_code}")
        data = response.json()
        print("Trained Models and their metrics:")
        for m in data:
            print(f"  - {m['model_name']}: {m['metrics']}")
    except Exception as e:
        print(f"Error comparing models: {str(e)}")
        return False
        
    print("\n=== All Endpoint Tests Completed Successfully ===")
    return True

if __name__ == "__main__":
    # Wait 2 seconds for uvicorn to fully startup
    time.sleep(2.0)
    success = run_tests()
    sys.exit(0 if success else 1)
