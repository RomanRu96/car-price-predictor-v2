
def validate_horsepower(hp):
	"""Проверяет, что мощность в допустимом диапазоне"""
	if hp < 50 or hp > 300:
		return False
	return True