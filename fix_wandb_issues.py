#!/usr/bin/env python3
"""
WandB Connection Fixer and Diagnostic Tool

This script helps diagnose and fix common WandB connection issues,
especially the [WinError 10054] connection reset error during cleanup.
"""

import os
import sys
import time
import json
import shutil
import subprocess
import psutil
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def check_wandb_processes():
    """Check for running WandB processes"""
    print_header("WandB Process Check")
    
    wandb_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        try:
            if proc.info['name'] and 'wandb' in proc.info['name'].lower():
                wandb_processes.append(proc.info)
            elif proc.info['cmdline']:
                cmdline_str = ' '.join(proc.info['cmdline']).lower()
                if 'wandb' in cmdline_str:
                    wandb_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if wandb_processes:
        print(f"🔍 Found {len(wandb_processes)} WandB-related processes:")
        for proc in wandb_processes:
            print(f"  - PID {proc['pid']}: {proc['name']} ({proc['status']})")
        return wandb_processes
    else:
        print("✅ No WandB processes found")
        return []

def kill_wandb_processes():
    """Kill all WandB processes"""
    print_header("Killing WandB Processes")
    
    processes = check_wandb_processes()
    if not processes:
        print("ℹ️  No WandB processes to kill")
        return
    
    killed_count = 0
    for proc_info in processes:
        try:
            proc = psutil.Process(proc_info['pid'])
            proc.terminate()
            proc.wait(timeout=5)
            print(f"✅ Killed process {proc_info['pid']}")
            killed_count += 1
        except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied) as e:
            print(f"⚠️  Could not kill process {proc_info['pid']}: {e}")
    
    print(f"🎯 Killed {killed_count} WandB processes")

def clean_wandb_cache():
    """Clean WandB cache and temporary files"""
    print_header("Cleaning WandB Cache")
    
    # Common WandB directories
    wandb_dirs = [
        os.path.expanduser("~/.cache/wandb"),
        os.path.expanduser("~/.wandb"),
        os.path.expanduser("~/wandb"),
        "./wandb"
    ]
    
    cleaned_dirs = 0
    for wandb_dir in wandb_dirs:
        if os.path.exists(wandb_dir):
            try:
                print(f"🧹 Cleaning: {wandb_dir}")
                
                # Remove temporary files and logs
                for root, dirs, files in os.walk(wandb_dir):
                    for file in files:
                        if any(pattern in file.lower() for pattern in ['.tmp', '.log', '.lock', 'debug']):
                            file_path = os.path.join(root, file)
                            try:
                                os.remove(file_path)
                                print(f"  📄 Removed: {file}")
                            except Exception as e:
                                print(f"  ⚠️  Could not remove {file}: {e}")
                
                cleaned_dirs += 1
            except Exception as e:
                print(f"⚠️  Error cleaning {wandb_dir}: {e}")
        else:
            print(f"ℹ️  Directory not found: {wandb_dir}")
    
    print(f"✅ Cleaned {cleaned_dirs} WandB directories")

def check_network_connectivity():
    """Check network connectivity to WandB servers"""
    print_header("Network Connectivity Check")
    
    # Test WandB API connectivity
    try:
        import requests
        response = requests.get("https://api.wandb.ai/", timeout=10)
        if response.status_code == 200:
            print("✅ WandB API is reachable")
        else:
            print(f"⚠️  WandB API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach WandB API: {e}")
    
    # Test DNS resolution
    try:
        import socket
        socket.gethostbyname("api.wandb.ai")
        print("✅ DNS resolution working")
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")

def check_wandb_config():
    """Check WandB configuration"""
    print_header("WandB Configuration Check")
    
    # Check for API key
    api_key = os.environ.get('WANDB_API_KEY')
    if api_key:
        print(f"✅ WANDB_API_KEY found (length: {len(api_key)})")
    else:
        print("⚠️  WANDB_API_KEY not found in environment")
    
    # Check WandB config file
    config_path = os.path.expanduser("~/.netrc")
    if os.path.exists(config_path):
        print(f"✅ WandB config file found: {config_path}")
    else:
        print("ℹ️  No .netrc file found")
    
    # Check WandB settings
    settings_path = os.path.expanduser("~/.config/wandb/settings")
    if os.path.exists(settings_path):
        print(f"✅ WandB settings found: {settings_path}")
        try:
            with open(settings_path, 'r') as f:
                settings = f.read()
                print(f"📄 Settings preview: {settings[:200]}...")
        except Exception as e:
            print(f"⚠️  Could not read settings: {e}")
    else:
        print("ℹ️  No WandB settings file found")

def fix_wandb_permissions():
    """Fix WandB file permissions on Windows"""
    print_header("Fixing WandB Permissions")
    
    try:
        # Find WandB directories
        wandb_dirs = [
            os.path.expanduser("~/.cache/wandb"),
            os.path.expanduser("~/.wandb"),
            "./wandb"
        ]
        
        for wandb_dir in wandb_dirs:
            if os.path.exists(wandb_dir):
                print(f"🔧 Fixing permissions for: {wandb_dir}")
                
                # On Windows, use attrib command to remove read-only attributes
                try:
                    subprocess.run(['attrib', '-R', f'{wandb_dir}\\*.*', '/S'], 
                                 capture_output=True, check=False)
                    print(f"  ✅ Permissions fixed for {wandb_dir}")
                except Exception as e:
                    print(f"  ⚠️  Permission fix failed: {e}")
        
        print("✅ Permission fixes completed")
        
    except Exception as e:
        print(f"❌ Permission fix error: {e}")

def create_wandb_offline_backup():
    """Create offline backup of WandB data"""
    print_header("Creating WandB Offline Backup")
    
    try:
        wandb_dir = "./wandb"
        if os.path.exists(wandb_dir):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = f"wandb_backup_{timestamp}"
            
            print(f"💾 Creating backup: {backup_dir}")
            shutil.copytree(wandb_dir, backup_dir)
            print(f"✅ Backup created successfully")
            
            # Create backup summary
            summary = {
                "backup_time": timestamp,
                "original_dir": wandb_dir,
                "backup_dir": backup_dir,
                "files_backed_up": sum(1 for _, _, files in os.walk(backup_dir) for _ in files)
            }
            
            with open(f"{backup_dir}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📄 Backup summary saved: {backup_dir}_summary.json")
            return backup_dir
        else:
            print("ℹ️  No WandB directory found to backup")
            return None
            
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None

def test_wandb_connection():
    """Test WandB connection with a minimal run"""
    print_header("Testing WandB Connection")
    
    try:
        import wandb
        
        print("🧪 Starting minimal WandB test run...")
        
        # Test with offline mode first
        test_run = wandb.init(
            project="connection-test",
            name=f"test_{int(time.time())}",
            mode="offline"
        )
        
        # Log some test data
        test_run.log({"test_metric": 1.0})
        
        # Try to finish
        test_run.finish()
        
        print("✅ WandB offline test passed")
        
        # Now test online mode
        print("🌐 Testing online mode...")
        
        online_run = wandb.init(
            project="connection-test",
            name=f"online_test_{int(time.time())}",
            mode="online"
        )
        
        online_run.log({"online_metric": 2.0})
        online_run.finish()
        
        print("✅ WandB online test passed")
        return True
        
    except Exception as e:
        print(f"❌ WandB connection test failed: {e}")
        return False

def main():
    """Main diagnostic and fix function"""
    print("🔧 WandB Connection Fixer and Diagnostic Tool")
    print("=" * 60)
    print("This tool will help diagnose and fix WandB connection issues.")
    print("Especially useful for fixing [WinError 10054] connection reset errors.")
    
    # Step 1: Check for running processes
    check_wandb_processes()
    
    # Step 2: Kill any hanging processes
    kill_wandb_processes()
    
    # Step 3: Clean cache
    clean_wandb_cache()
    
    # Step 4: Fix permissions
    fix_wandb_permissions()
    
    # Step 5: Check network connectivity
    check_network_connectivity()
    
    # Step 6: Check configuration
    check_wandb_config()
    
    # Step 7: Create backup
    backup_dir = create_wandb_offline_backup()
    
    # Step 8: Test connection
    connection_ok = test_wandb_connection()
    
    # Final summary
    print_header("Fix Summary")
    
    if connection_ok:
        print("✅ WandB connection is working properly!")
        print("🎯 You can now run your training scripts safely.")
    else:
        print("⚠️  WandB connection issues persist.")
        print("💡 Recommended next steps:")
        print("   1. Check your internet connection")
        print("   2. Verify your WandB API key")
        print("   3. Try running training in offline mode")
        print("   4. Contact WandB support if issues continue")
    
    if backup_dir:
        print(f"💾 WandB data backed up to: {backup_dir}")
    
    print("\n🎯 To prevent future issues:")
    print("   • Use the wandb_utils.py module in your training scripts")
    print("   • Always use safe_wandb_finish() instead of wandb.finish()")
    print("   • Run this diagnostic tool before long training sessions")

if __name__ == "__main__":
    main()
