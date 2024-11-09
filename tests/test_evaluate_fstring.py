import pytest
from autolevels import evaluate_fstring


def test_basic_functionality():
    """Test basic string formatting cases"""
    assert evaluate_fstring("f'Hello {name}!'", "World") == "Hello World!"
    assert evaluate_fstring("f'Number: {num}'", 42) == "Number: 42"
    assert evaluate_fstring('f"Value: {val}"', True) == "Value: True"


def test_format_specifications():
    """Test various format specifications"""
    assert evaluate_fstring("f'IMG_{fn:04d}.jpg'", 5) == "IMG_0005.jpg"
    assert evaluate_fstring("f'{num:>10}'", 42) == "        42"
    assert evaluate_fstring("f'{val:.2f}'", 3.14159) == "3.14"
    assert evaluate_fstring("f'{text:>10.5}'", "hello world") == "     hello"
    assert evaluate_fstring('"{x}"', '42') == "42"


def test_automatic_str_conversion():
    """Test str to int conversion according to format specification"""
    assert evaluate_fstring("f'1-{fn:03d}.jpg'", '5') == "1-005.jpg"


def test_sanitize_fstring():
    """Test adding missing quotes and leading f"""
    assert evaluate_fstring("f1-{fn:03d}.jpg", '5') == "1-005.jpg"
    assert evaluate_fstring("1-{fn:03d}.jpg", '5') == "1-005.jpg"


def test_security_threats():
    """Test potential security threats and code injection attempts"""
    dangerous_inputs = [
        # Attempted code injection
        "f'{x.__import__('os').system('rm -rf /')}'",
        "f'{x.__class__.__bases__[0].__subclasses__()}'",
        "f'{x.__init__.__globals__}'",
        'f"{__import__("os").system("ls")}"',
        # Format string injection
        "f'{x}'.format(system('rm -rf /'))",
        "f'{x}}{system('rm -rf /')}'",
        # Attribute access attempts
        "f'{x.__dict__}'",
        "f'{x.__class__}'",
        # Nested format attempts
        "f'{x:{y}}'",
        # Multiple variables
        "f'{x}{y}'",
        # DoS via massive precision
        "f'{x:.{2**1000000}f}'",
        # Memory exhaustion
        "f'{x:100000000000000000d}'",
        # Attempt to exploit with complex expression
        'f"{x.__class__.__bases__[0].__subclasses__()}"',
    ]
    for dangerous_input in dangerous_inputs:
        with pytest.raises(ValueError):
            evaluate_fstring(dangerous_input, 42)


def test_edge_cases():
    """Test edge cases and potential error conditions"""
    invalid_inputs = [
        # Invalid f-string syntax
        "not an fstring",
        "f'{}'",
        "f'{ }'",
        # Invalid variable names
        "f'{123var}'",
        "f'{@var}'",
        "f'{var-name}'",
        # Invalid format specs
        "f'{var!r}'",  # Python's !r representation
        "f'{var!s}'",  # Python's !s representation
        "f'{var!a}'",  # Python's !a representation
        'f"{var:invalid}"',
        # Empty or whitespace
        "f''",
        "f' '",
        # Mismatched quotes
        'f"test\'',
        'f"{x}',
        # Multiple variables with similar names
        "f'{var}{variable}'",
    ]
    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError):
            evaluate_fstring(invalid_input, 42)
            evaluate_fstring(invalid_input, 0.0)
            evaluate_fstring(invalid_input, '42')


def test_whitespace_handling():
    """Test handling of whitespace in various positions"""
    assert evaluate_fstring("f'{ var }'", 42) == "42"
    with pytest.raises(ValueError):
        evaluate_fstring("f'{ var:>5 }'", 42)


def test_error_messages():
    """Test that appropriate error messages are raised"""
    with pytest.raises(ValueError, match="The f-string is improperly formatted or missing quotes"):
        evaluate_fstring("not an fstring", 42)

    with pytest.raises(ValueError, match="No valid variable symbol"):
        evaluate_fstring("f'{}'", 42)

    with pytest.raises(ValueError, match='Format is "d", but "not a number" is not a number'):
        evaluate_fstring("f'{var:d}'", "not a number")
