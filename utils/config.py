"""
config.py
"""

def str_to_bool(value):
    """
    parse strings to read argparse flag entries in as True/False
    INPUTS:
        value: input value
    OUTPUTS:
        either True, False, or raises error
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')