
# чтобы запустить тесты ввести в коммандной строке python -m pytest
# установить pytest.   pip pytest

from utils import validate_horsepower

def test_valid_hp():
	assert validate_horsepower(150) == True 

def test_low_hp():
	assert validate_horsepower(30) == False

def test_high_hp():
	assert validate_horsepower(350) == False


