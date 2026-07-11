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
        print(f"Report Is Valid: {data.get('report', {}).get('is_valid')}")
    except Exception as e:
        print(f"Error loading demo: {str(e)}")
        return False
        
    # 3. Check status
    print("\nChecking Dataset Status (/api/dataset/status)...")
    try:
        response = httpx.get(f"{url}/api/dataset/status", timeout=5.0)
        print(f"Status Endpoint: {response.status_code}")
        data = response.json()
        print(f"Loaded: {data.get('loaded')}")
        print(f"Summary: {data.get('profile_summary')}")
    except Exception as e:
        print(f"Error getting status: {str(e)}")
        return False
        
    # 4. Check EDA
    print("\nRequesting EDA Report (/api/dataset/eda)...")
    try:
        response = httpx.get(f"{url}/api/dataset/eda", timeout=15.0)
        print(f"EDA Endpoint Status: {response.status_code}")
        data = response.json()
        print("EDA Keys verified:")
        for k in data.keys():
            print(f"  - {k}")
        print(f"Outliers count in report: {len(data.get('outliers', []))}")
    except Exception as e:
        print(f"Error getting EDA: {str(e)}")
        return False
        
    print("\n=== All Endpoint Tests Completed Successfully ===")
    return True

if __name__ == "__main__":
    # Wait 2 seconds for uvicorn to fully startup
    time.sleep(2.0)
    success = run_tests()
    sys.exit(0 if success else 1)
