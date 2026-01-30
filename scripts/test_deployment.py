import requests
import time
import subprocess
import os
import signal
import sys

def test_api():
    print("🚀 Starting FastAPI server for deployment testing...")
    
    # Start the server in a subprocess
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "fastapi_app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    time.sleep(5) # Give it time to load the model
    
    try:
        # 1. Health Check
        print("🔍 Checking API health...")
        health_resp = requests.get("http://127.0.0.1:8000/health")
        print(f"Health Response: {health_resp.json()}")
        assert health_resp.status_code == 200
        
        # 2. Prediction Test
        print("🔮 Testing Sales Prediction endpoint...")
        payload = {
            "Store": 1,
            "Date": "2015-09-17",
            "Promo": 1,
            "StateHoliday": "0",
            "SchoolHoliday": 0
        }
        pred_resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if pred_resp.status_code == 200:
            result = pred_resp.json()
            print(f"✅ Prediction Successful!")
            print(f"   Store: {result['Store']}")
            print(f"   Date: {result['Date']}")
            print(f"   Predicted Sales: €{result['PredictedSales']:.2f}")
        else:
            print(f"❌ Prediction Failed: {pred_resp.text}")
            
        assert pred_resp.status_code == 200
        
    except Exception as e:
        print(f"💥 Error during API test: {str(e)}")
        # Print server logs if it failed
        out, err = server_process.communicate(timeout=1)
        print(f"Server STDOUT: {out.decode()}")
        print(f"Server STDERR: {err.decode()}")
        raise e
    finally:
        print("🛑 Shutting down server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)

if __name__ == "__main__":
    test_api()
