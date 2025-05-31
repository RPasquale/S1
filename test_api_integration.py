"""
Test CFA Integration with Server API

This script tests the CFA embedding training integration with the server API.
"""

import requests
import json
import time

# Server URL
BASE_URL = "http://localhost:8000"

def test_cfa_documents_endpoint():
    """Test the CFA documents endpoint"""
    print("=== Testing CFA Documents Endpoint ===")
    
    try:
        response = requests.get(f"{BASE_URL}/api/cfa-documents")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['document_count']} CFA documents")
            print(f"📁 Document folder: {data['doc_folder']}")
            print(f"📄 Document files: {data['document_files'][:3]}..." if len(data['document_files']) > 3 else f"📄 Document files: {data['document_files']}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_cfa_training_endpoint():
    """Test the CFA training endpoint"""
    print("\n=== Testing CFA Training Endpoint ===")
    
    try:
        # Start training with minimal parameters for testing
        payload = {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 2e-5
        }
        
        response = requests.post(f"{BASE_URL}/api/train/cfa", params=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training started: {data['message']}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_training_status():
    """Test the training status endpoint"""
    print("\n=== Testing Training Status ===")
    
    try:
        response = requests.get(f"{BASE_URL}/api/training/status")
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Training status: {data.get('status_message', 'Unknown')}")
            print(f"🔄 Progress: {data.get('progress', 0)}%")
            print(f"🏃 Is training: {data.get('is_training', False)}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing CFA Integration with Server API\n")
    
    # Test 1: Check CFA documents
    docs_ok = test_cfa_documents_endpoint()
    
    if docs_ok:
        # Test 2: Start training
        training_ok = test_cfa_training_endpoint()
        
        if training_ok:
            # Test 3: Monitor status for a bit
            print("\n⏱️  Monitoring training status for 30 seconds...")
            for i in range(6):
                time.sleep(5)
                test_training_status()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    main()
