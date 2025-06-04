#!/usr/bin/env python3
"""
Expert Matching System - Test Script
Tests basic functionality and setup
"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_imports():
    """Test if all required packages are available"""
    print("🧪 Testing imports...")
    
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('streamlit', 'Streamlit'), 
        ('sklearn', 'scikit-learn'),
        ('pandas', 'pandas'),
        ('numpy', 'NumPy'),
        ('requests', 'requests'),
        ('plotly', 'Plotly'),
        ('uvicorn', 'Uvicorn')
    ]
    
    missing_packages = []
    
    for package_name, display_name in required_packages:
        try:
            importlib.import_module(package_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} - Not installed")
            missing_packages.append(package_name)
    
    # Test optional packages
    optional_packages = [
        ('nltk', 'NLTK'),
        ('spacy', 'spaCy')
    ]
    
    print("\n🔍 Optional packages:")
    for package_name, display_name in optional_packages:
        try:
            importlib.import_module(package_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ⚠️  {display_name} - Not installed (optional)")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All required packages are available!")
        return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\n📁 Testing project structure...")
    
    required_files = [
        'app/main.py',
        'app/models.py',
        'app/database.py',
        'app/processor.py',
        'app/matcher.py',
        'streamlit_app.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'app',
        'data'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ - Missing directory")
            missing_items.append(dir_name)
    
    # Check files
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} - Missing file")
            missing_items.append(file_name)
    
    if missing_items:
        print(f"\n❌ Missing items: {', '.join(missing_items)}")
        return False
    else:
        print("\n✅ Project structure is correct!")
        return True

def test_app_imports():
    """Test if app modules can be imported"""
    print("\n🐍 Testing app module imports...")
    
    app_modules = [
        'app.main',
        'app.models',
        'app.database', 
        'app.processor',
        'app.matcher'
    ]
    
    failed_imports = []
    
    for module_name in app_modules:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name} - Import failed: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"  ⚠️  {module_name} - Import warning: {e}")
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All app modules imported successfully!")
        return True

def test_database_creation():
    """Test database creation"""
    print("\n💾 Testing database creation...")
    
    try:
        from app.database import DatabaseManager
        
        # Create test database
        test_db_path = "data/test_experts.db"
        db = DatabaseManager(test_db_path)
        
        # Test basic operations
        stats = db.get_system_stats()
        print(f"  ✅ Database created and connected")
        print(f"  📊 Initial stats: {stats}")
        
        # Clean up test database
        import os
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print(f"  🧹 Test database cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        return False

def test_cv_processing():
    """Test CV processing functionality"""
    print("\n📄 Testing CV processing...")
    
    try:
        from app.processor import CVProcessor
        
        processor = CVProcessor()
        
        # Test CV text
        test_cv = """
        John Doe
        Software Engineer with 5 years of experience in Python, JavaScript, and machine learning.
        Experience with React, Django, and AWS cloud services.
        PhD in Computer Science from MIT.
        Email: john.doe@example.com
        """
        
        result = processor.process_cv(test_cv, "John Doe", "john.doe@example.com")
        
        print(f"  ✅ CV processed successfully")
        print(f"  📝 Extracted {len(result['skills'])} skills")
        print(f"  💼 Detected {result['experience_years']} years experience")
        print(f"  🎓 Education levels: {result['education']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CV processing test failed: {e}")
        return False

def test_matching_engine():
    """Test matching engine functionality"""
    print("\n🎯 Testing matching engine...")
    
    try:
        from app.matcher import ExpertMatcher
        
        # Create test matcher with temporary database
        test_db_path = "data/test_matching.db"
        matcher = ExpertMatcher(test_db_path)
        
        # Add a test expert
        test_cv = "Software engineer with Python and machine learning experience."
        expert_id = matcher.add_expert_from_cv("Test Expert", "test@example.com", test_cv)
        
        print(f"  ✅ Expert added with ID: {expert_id}")
        
        # Test matching
        matches = matcher.find_best_matches(
            "Looking for Python developer with ML experience",
            ["Python", "Machine Learning"],
            top_k=1
        )
        
        print(f"  ✅ Found {len(matches)} matches")
        if matches:
            print(f"  🎯 Best match score: {matches[0]['combined_score']:.3f}")
        
        # Clean up
        import os
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print(f"  🧹 Test database cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Matching engine test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SYSTEM TEST COMPLETED!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1️⃣  Start the system:")
    print("   Windows: run_system.bat")
    print("   Linux/Mac: ./run_system.sh")
    print("   Manual: python start_backend.py (then python start_frontend.py)")
    
    print("\n2️⃣  Access the application:")
    print("   🎨 Frontend: http://localhost:8501")
    print("   🔧 API Docs: http://localhost:8000/docs")
    
    print("\n3️⃣  Seed the database:")
    print("   Go to Admin page and click 'Seed Database'")
    print("   Or use API: POST http://localhost:8000/data/seed")
    
    print("\n4️⃣  Start matching experts to tasks!")
    print("\n💡 Tips:")
    print("   - Upload CV files in the Experts section")
    print("   - Create tasks in the Tasks section")
    print("   - Use the Matching page for ad-hoc matches")
    print("   - Check Analytics for insights")

def main():
    """Run all tests"""
    print("🎯 Expert Matching System - System Test")
    print("="*50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("App Module Imports", test_app_imports),
        ("Database Creation", test_database_creation),
        ("CV Processing", test_cv_processing),
        ("Matching Engine", test_matching_engine)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    # Summary
    print("\n" + "="*60)
    print(f"📊 TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    print("="*60)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        print_next_steps()
    else:
        print("❌ Some tests failed. Please fix the issues before using the system.")
        print("\n💡 Common fixes:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Check file permissions and project structure")
        print("   - Ensure Python 3.8+ is installed")

if __name__ == "__main__":
    main() 