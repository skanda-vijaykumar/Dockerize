import pytest
import asyncio
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestBasicFunctionality:
    def test_python_version(self):
        assert sys.version_info >= (3, 10), "Python version should be 3.10 or higher"
    
    def test_imports(self):
        try:
            import fastapi
            import psycopg
            import ollama
            import langchain
            import llama_index
            assert True
        except ImportError as e:
            pytest.fail(f"Critical import failed: {e}")
    ## Hello test env, u there?    
    def test_environment_variables(self):
        required_vars = ['POSTGRES_HOST','POSTGRES_USER','POSTGRES_PASSWORD','POSTGRES_DB']
        
        for var in required_vars:
            value = os.getenv(var)
            if value is not None:
                assert len(value) > 0, f"Environment variable {var} is empty"

class TestAsyncFunctionality:
    @pytest.mark.asyncio
    async def test_async_basic(self):
        async def dummy_async_function():
            await asyncio.sleep(0.01)
            return "success"
        
        result = await dummy_async_function()
        assert result == "success"
## Hello db, are you there?
class TestDatabaseConnection:
    def test_psycopg_connection_format(self):
        host = os.getenv('POSTGRES_HOST', 'localhost')
        user = os.getenv('POSTGRES_USER', 'postgres') 
        password = os.getenv('POSTGRES_PASSWORD', 'password')
        db = os.getenv('POSTGRES_DB', 'test_db')        
        conn_string = f"postgresql://{user}:{password}@{host}/{db}"
        assert "postgresql://" in conn_string
        assert host in conn_string
        assert user in conn_string
        assert db in conn_string
## directory there or no?
class TestApplicationStructure:
    def test_main_files_exist(self):
        expected_files = ['route18.py','requirements.txt','Dockerfile','docker-compose.yml']
        for file in expected_files:
            if os.path.exists(file):
                assert os.path.isfile(file), f"{file} should be a file"
                assert os.path.getsize(file) > 0, f"{file} should not be empty"
    
    def test_static_directory(self):
        if os.path.exists('static'):
            assert os.path.isdir('static'), "static should be a directory"
    def test_templates_directory(self):
        if os.path.exists('templates'):
            assert os.path.isdir('templates'), "templates should be a directory"

class TestBasicConfiguration:
    def test_docker_configuration(self):
        if os.path.exists('docker-compose.yml'):
            with open('docker-compose.yml', 'r') as f:
                content = f.read()
                assert 'postgres' in content.lower(), "Docker compose should include postgres"
                assert 'ollama' in content.lower(), "Docker compose should include ollama"
                assert 'app' in content.lower(), "Docker compose should include app service"

# Simple smoke test for testing pytest
def test_smoke():
    assert True

## Test runs only in CI environment
def test_ci_environment():
    ci_indicators = [
        'CI',
        'GITLAB_CI', 
        'GITHUB_ACTIONS',
        'CONTINUOUS_INTEGRATION'
    ]
    
    is_ci = any(os.getenv(indicator) for indicator in ci_indicators)
    if is_ci:
        print("Running in CI environment")
        assert True
    else:
        print("Running in local environment")
        assert True

if __name__ == "__main__":
    ## running tests directly
    pytest.main([__file__, "-v"])
