def vacation_days(months_employed: int) -> int:
    if months_employed < 0:
        raise ValueError("months_employed cannot be negative")
    if months_employed < 12:
        return 0
    if months_employed < 60:
        return 10
    return 15
