def mean(numbers):
    """Return the mean (average) of a list of numbers."""
    if not numbers:
        raise ValueError("List is empty.")
    return sum(numbers) / len(numbers)

def median(numbers):
    """Return the median of a list of numbers."""
    if not numbers:
        raise ValueError("List is empty.")
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:
        return sorted_numbers[mid]
    
def mode(numbers):
    """Return the mode of a list of numbers."""
    if not numbers:
        raise ValueError("List is empty.")
    frequency = {}
    for number in numbers:
        frequency[number] = frequency.get(number, 0) + 1
    max_freq = max(frequency.values())
    modes = [num for num, freq in frequency.items() if freq == max_freq]
    if len(modes) == len(numbers):
        return None  # No mode found
    return modes

def variance(numbers):
    """Return the variance of a list of numbers."""
    if not numbers:
        raise ValueError("List is empty.")
    mean_value = mean(numbers)
    return sum((x - mean_value) ** 2 for x in numbers) / len(numbers)