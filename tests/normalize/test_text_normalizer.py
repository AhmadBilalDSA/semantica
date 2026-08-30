import unittest
from unittest.mock import MagicMock, patch

from semantica.normalize.text_normalizer import (
    SpecialCharacterProcessor,
    TextNormalizer,
    UnicodeNormalizer,
    WhitespaceNormalizer,
)


class TestTextNormalizer(unittest.TestCase):
    """
    Test suite for the TextNormalizer class.
    """

    def setUp(self):
        """Set up mocks"""

        self.logger_patcher = patch("semantica.normalize.text_normalizer.get_logger")
        self.tracker_patcher = patch(
            "semantica.normalize.text_normalizer.get_progress_tracker"
        )
        self.cleaner_patcher = patch("semantica.normalize.text_normalizer.TextCleaner")

        self.mock_logger = self.logger_patcher.start()
        self.mock_tracker = self.tracker_patcher.start()
        self.mock_cleaner_cls = self.cleaner_patcher.start()

        # config mocks
        self.mock_tracker_instance = MagicMock()
        self.mock_tracker_instance.enabled = True
        self.mock_tracker.return_value = self.mock_tracker_instance

        self.mock_cleaner_instance = MagicMock()
        self.mock_cleaner_cls.return_value = self.mock_cleaner_instance

        # init normalization

        self.normalizer = TextNormalizer()

    def tearDown(self):
        """Stop all patches."""
        self.logger_patcher.stop()
        self.tracker_patcher.stop()
        self.cleaner_patcher.stop()

    def test_init(self):
        """Test initialization"""
        self.mock_cleaner_cls.assert_called_once()
        self.assertTrue(hasattr(self.normalizer, "unicode_normalizer"))
        self.assertTrue(hasattr(self.normalizer, "whitespace_normalizer"))
        self.assertTrue(hasattr(self.normalizer, "special_char_processor"))

        self.assertTrue(self.normalizer.progress_tracker.enabled)

    def test_normalize_text_basic(self):
        """Test basic text normalization"""
        text = "Hello World"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Hello World")

        # progress bar insurance

        self.mock_tracker_instance.start_tracking.assert_called()
        self.mock_tracker_instance.stop_tracking.assert_called_with(
            self.mock_tracker_instance.start_tracking.return_value, status="completed"
        )

    def test_normalize_empty_string(self):
        """Test 'nothingness'"""
        self.assertEqual(self.normalizer.normalize_text(""), "")
        self.assertEqual(self.normalizer.normalize_text(None), "")

    def test_normalize_case_options(self):
        """Test case normalization"""
        text = "HeLLo WoRLd"

        self.assertEqual(
            self.normalizer.normalize_text(text, case="lower"), "hello world"
        )

        self.assertEqual(
            self.normalizer.normalize_text(text, case="upper"), "HELLO WORLD"
        )
        self.assertEqual(
            self.normalizer.normalize_text(text, case="title"), "Hello World"
        )

        # preserve test ---- default

        self.assertEqual(
            self.normalizer.normalize_text(text, case="preserve"), "HeLLo WoRLd"
        )

    def test_normalize_delegation(self):
        """Verify that normalize_text correctly delegates to subcomponents."""

        self.normalizer.unicode_normalizer.normalize_unicode = MagicMock(
            return_value="U"
        )
        self.normalizer.whitespace_normalizer.normalize_whitespace = MagicMock(
            return_value="W"
        )
        self.normalizer.special_char_processor.process_special_chars = MagicMock(
            return_value="S"
        )

        result = self.normalizer.normalize_text(
            "input",
            unicode_form="NFD",
            line_break_type="windows",
            normalize_diacritics=True,
        )

        self.normalizer.unicode_normalizer.normalize_unicode.assert_called_with(
            "input", form="NFD"
        )
        self.normalizer.whitespace_normalizer.normalize_whitespace.assert_called_with(
            "U", line_break_type="windows"
        )
        self.normalizer.special_char_processor.process_special_chars.assert_called_with(
            "W", normalize_diacritics=True
        )

        self.assertEqual(result, "S")

    def test_clean_text(self):
        """Test delegation to TextCleaner"""
        text = "<html>body</html>"
        self.mock_cleaner_instance.clean.return_value = "body"
        result = self.normalizer.clean_text(text, remove_html=True)

        self.mock_cleaner_instance.clean.assert_called_with(text, remove_html=True)
        self.assertEqual(result, "body")

    def test_standardize_format(self):
        """Test format standardization option"""
        text = "  one   two  "

        self.assertEqual(
            self.normalizer.standardize_format(text, format_type="compact"), "one two"
        )

        self.assertEqual(
            self.normalizer.standardize_format(text, format_type="preserve"),
            "one   two",
        )

    def test_process_batch(self):
        """Test batch processing"""
        texts = ["TEST 1", "Test 2"]
        results = self.normalizer.process_batch(texts, case="lower")
        self.assertEqual(results, ["test 1", "test 2"])

    def test_normalize_overloaded_method(self):
        """Test generic normalize method"""
        self.assertEqual(self.normalizer.normalize("TEST", case="lower"), "test")

        # dict

        docs = [
            {"id": 1, "content": "DOC 1"},
            {"id": 2, "content": "DOC 2", "other": "meta"},
            {"id": 3, "nocontent": "skip"},
        ]

        results = self.normalizer.normalize(docs, case="lower")

        self.assertEqual(results[0]["content"], "doc 1")
        self.assertEqual(results[1]["content"], "doc 2")
        self.assertEqual(results[1]["other"], "meta")

        self.assertIn("skip", results[2])

    def test_normalize_error_handling(self):
        """Test error handling"""

        self.normalizer.unicode_normalizer.normalize_unicode = MagicMock(
            side_effect=Exception("Test Error")
        )

        with self.assertRaises(Exception):
            self.normalizer.normalize_text("input")
        self.mock_tracker_instance.stop_tracking.assert_called_with(
            self.mock_tracker_instance.start_tracking.return_value,
            status="failed",
            message="Test Error",
        )


class TestUnicodeNormalizer(unittest.TestCase):
    """Test suite for UniCodeNormalizer class"""

    def setUp(self):
        self.normalizer = UnicodeNormalizer()

    def test_normalize_unicode_forms(self):
        """Test diff unicode normalization forms"""

        text_nfc = "\u00e9"
        text_nfd = "\u0065\u0301"

        self.assertEqual(self.normalizer.normalize_unicode(text_nfd, "NFC"), text_nfc)
        self.assertEqual(self.normalizer.normalize_unicode(text_nfc, "NFD"), text_nfd)

    def test_normalize_none(self):
        """Test empty input"""

        self.assertEqual(self.normalizer.normalize_unicode(None), "")
        self.assertEqual(self.normalizer.normalize_unicode(""), "")

    def test_normalize_failure_fallback(self):
        """Test that it returns og text if unicode fails"""

        with patch("unicodedata.normalize", side_effect=Exception("Boom")):
            result = self.normalizer.normalize_unicode("test")
            self.assertEqual(result, "test")

    def test_handle_encoding(self):
        """Test encoding handling"""
        self.assertEqual(self.normalizer.handle_encoding("test", "utf-8"), "test")

        # bytes in

        byte_data = "test".encode("utf-8")
        self.assertEqual(self.normalizer.handle_encoding(byte_data, "utf-8"), "test")

        # cross encoding

        latin_bytes = "café".encode("latin-1")
        result = self.normalizer.handle_encoding(latin_bytes, "latin-1", "utf-8")
        self.assertEqual(result, "café")

        # broken bites

        bad_bytes = b"\xff"
        self.assertIsInstance(self.normalizer.handle_encoding(bad_bytes, "utf-8"), str)

    def test_process_special_chars_replacement(self):
        """Test unicode character replacement"""
        input_text = "\u2018single\u2019 \u201Cdouble\u201D \u2013 \u2014 \u2026"
        expected = "'single' \"double\" - -- ..."
        self.assertEqual(self.normalizer.process_special_chars(input_text), expected)


class TestWhitespaceNormalizer(unittest.TestCase):
    """Test suite for WhitespaceNormalizer class"""

    def setUp(self):
        self.normalizer = WhitespaceNormalizer()

    def test_normalize_whitespace_basic(self):
        """Test basic whitespace cleanup"""
        text = "Hello   World\tTest"

        self.assertEqual(self.normalizer.normalize_whitespace(text), "Hello World Test")

    def test_handle_line_breaks(self):
        """Test line break conversion"""
        text = "Row1\r\nRow2\rRow3\n"

        self.assertEqual(
            self.normalizer.handle_line_breaks(text, "unix"), "Row1\nRow2\nRow3\n"
        )

        res_windows = self.normalizer.handle_line_breaks("Row1\nRow2", "windows")
        self.assertEqual(res_windows, "Row1\r\nRow2")

    def test_process_indentation(self):
        """Test indentation conversion"""

        spaces = "    Code"
        self.assertEqual(self.normalizer.process_indentation(spaces, "tabs"), "\tCode")

        tabs = "\tCode"
        self.assertEqual(
            self.normalizer.process_indentation(tabs, "spaces"), "    Code"
        )


class TestSpecialCharacterProcessor(unittest.TestCase):
    """Test suite for SpecialCharacterProcessor class."""

    def setUp(self):
        self.processor = SpecialCharacterProcessor()

    def test_normalize_punctuation(self):
        """Test punctuation cleanup"""

        text = "“Hello” ‘World’ – …"
        expected = "\"Hello\" 'World' - ..."

        self.assertEqual(self.processor.normalize_punctuation(text), expected)

    def test_process_diacritics_remove(self):
        """Test removing diacritics"""

        text = "Crème Brûlée"
        expected = "Creme Brulee"
        result = self.processor.process_diacritics(text, remove_diacritics=True)

        self.assertEqual(result, expected)

    def test_process_diacritics_normalize(self):
        """Test normalizing diacritics"""

        text = "e\u0301"  # NFD ~~ this wastes memory

        expected = "\u00e9"  # should become NFC which is uh precomposed single char
        result = self.processor.process_diacritics(text, remove_diacritics=False)
        self.assertEqual(result, expected)

    def test_process_special_chars_integration(self):
        """Test the main processing method integration"""
        text = "\u201cCr\u00e8me\u201d"

        result = self.processor.process_special_chars(
            text, normalize_diacritics=True, remove_diacritics=True
        )
        self.assertEqual(result, '"Creme"')


class TestFencedCodeBlockPreservation(unittest.TestCase):
    """
    Test suite ensuring whitespace normalization preserves indentation
    inside fenced code blocks (``` and ~~~ delimiters).
    """

    def setUp(self):
        self.normalizer = WhitespaceNormalizer()

    def test_python_code_4_space_indentation_backtick_fence(self):
        """4-space indented Python code inside ```python fences is preserved."""
        text = (
            "Some text before\n"
            "\n"
            "```python\n"
            "def hello():\n"
            "    print('hi')\n"
            "    if True:\n"
            "        return 1\n"
            "```\n"
            "\n"
            "Some text after"
        )
        result = self.normalizer.normalize_whitespace(text)
        self.assertIn("    print('hi')", result)
        self.assertIn("        return 1", result)
        self.assertIn("def hello():", result)

    def test_python_code_4_space_indentation_tilde_fence(self):
        """4-space indented Python code inside ~~~python fences is preserved."""
        text = (
            "Before\n"
            "\n"
            "~~~python\n"
            "class Foo:\n"
            "    def method(self):\n"
            "        pass\n"
            "~~~\n"
            "\n"
            "After"
        )
        result = self.normalizer.normalize_whitespace(text)
        self.assertIn("    def method(self):", result)
        self.assertIn("        pass", result)

    def test_nested_lists_outside_code_block(self):
        """Nested list indentation outside code blocks is collapsed normally."""
        text = (
            "A list:\n"
            "\n"
            "  - item one\n"
            "  - item two\n"
            "    - nested item\n"
            "  - item three"
        )
        result = self.normalizer.normalize_whitespace(text)
        # The leading spaces on list items get collapsed
        self.assertNotIn("  - item", result)
        self.assertIn("- item", result)

    def test_mixed_code_and_normal_text(self):
        """Code block is preserved while surrounding text is normalized."""
        text = (
            "  Leading   spaces   in   prose\n"
            "\n"
            "```js\n"
            "const   x   =   1;\n"
            "```\n"
            "\n"
            "  Trailing   spaces   in   prose"
        )
        result = self.normalizer.normalize_whitespace(text)
        # Prose multiple spaces collapsed
        self.assertIn("Leading spaces in prose", result)
        self.assertIn("Trailing spaces in prose", result)
        # Code block whitespace preserved
        self.assertIn("const   x   =   1;", result)

    def test_multiple_code_blocks(self):
        """Multiple code blocks are each preserved."""
        text = (
            "First\n"
            "```\n"
            "    a    b\n"
            "```\n"
            "Middle\n"
            "```\n"
            "    c    d\n"
            "```\n"
            "Last"
        )
        result = self.normalizer.normalize_whitespace(text)
        self.assertIn("    a    b", result)
        self.assertIn("    c    d", result)
        self.assertIn("First", result)
        self.assertIn("Middle", result)
        self.assertIn("Last", result)

    def test_unclosed_fence_preserves_content(self):
        """An unclosed fence still preserves content as code."""
        text = (
            "Before\n"
            "```\n"
            "    indented code\n"
            "    still code"
        )
        result = self.normalizer.normalize_whitespace(text)
        self.assertIn("    indented code", result)
        self.assertIn("    still code", result)

    def test_fence_with_language_tag(self):
        """Fence lines with language tags are handled correctly."""
        text = (
            "Text\n"
            "```python\n"
            "    x = 1\n"
            "```\n"
            "More text"
        )
        result = self.normalizer.normalize_whitespace(text)
        self.assertIn("    x = 1", result)

    def test_tab_to_space_conversion_in_code(self):
        """Tabs inside code blocks are converted to spaces but not collapsed."""
        text = (
            "```\n"
            "\tdef foo():\n"
            "\t\tpass\n"
            "```"
        )
        result = self.normalizer.normalize_whitespace(text)
        # Tabs converted to single space each
        self.assertIn(" def foo():", result)
        self.assertIn("  pass", result)

    def test_empty_code_block(self):
        """Empty code blocks are handled."""
        text = "Before\n```\n```\nAfter"
        result = self.normalizer.normalize_whitespace(text)
        self.assertIn("Before", result)
        self.assertIn("After", result)

    def test_code_block_only(self):
        """Text that is entirely a code block."""
        text = "```\n    only code\n```"
        result = self.normalizer.normalize_whitespace(text)
        self.assertIn("    only code", result)

    def test_empty_input(self):
        """Empty string returns empty string."""
        self.assertEqual(self.normalizer.normalize_whitespace(""), "")

    def test_real_world_python_example(self):
        """Full realistic example with prose, code, and nested lists."""
        text = (
            "Here   is   an   example:\n"
            "\n"
            "```python\n"
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
            "```\n"
            "\n"
            "And   some   notes:\n"
            "\n"
            "  - First   point\n"
            "  - Second   point\n"
            "    - Sub-point"
        )
        result = self.normalizer.normalize_whitespace(text)
        # Code block fully preserved
        self.assertIn("    if n <= 1:", result)
        self.assertIn("        return 1", result)
        self.assertIn("    return n * factorial(n - 1)", result)
        # Prose normalized
        self.assertIn("Here is an example", result)
        self.assertIn("And some notes", result)

    def test_split_code_blocks_helper(self):
        """Test the _split_code_blocks helper directly."""
        text = "abc\n```\ncode\n```\ndef"
        segments = self.normalizer._split_code_blocks(text)
        # Before (non-code) + code block (code) + after (non-code)
        non_code = [s for s in segments if not s[0]]
        code = [s for s in segments if s[0]]
        self.assertEqual(len(non_code), 2)
        self.assertIn("abc", non_code[0][1])
        self.assertIn("def", non_code[1][1])
        self.assertTrue(len(code) >= 1)
        # All code segments combined contain the full code block
        code_text = "".join(seg for _, seg in code)
        self.assertIn("```", code_text)
        self.assertIn("code", code_text)

    def test_very_long_fenced_block(self):
        """Deeply indented code (8+ spaces) inside a code block is preserved."""
        text = (
            "text\n"
            "```\n"
            "        eight spaces\n"
            "                sixteen spaces\n"
            "```\n"
            "more text"
        )
        result = self.normalizer.normalize_whitespace(text)
        self.assertIn("        eight spaces", result)
        self.assertIn("                sixteen spaces", result)


class TestTextNormalizerFencedCodeIntegration(unittest.TestCase):
    """
    Integration tests for TextNormalizer.normalize_text with fenced code blocks.
    """

    def setUp(self):
        self.logger_patcher = patch("semantica.normalize.text_normalizer.get_logger")
        self.tracker_patcher = patch(
            "semantica.normalize.text_normalizer.get_progress_tracker"
        )
        self.cleaner_patcher = patch("semantica.normalize.text_normalizer.TextCleaner")

        self.mock_logger = self.logger_patcher.start()
        self.mock_tracker = self.tracker_patcher.start()
        self.mock_cleaner_cls = self.cleaner_patcher.start()

        self.mock_tracker_instance = MagicMock()
        self.mock_tracker_instance.enabled = True
        self.mock_tracker.return_value = self.mock_tracker_instance

        self.mock_cleaner_instance = MagicMock()
        self.mock_cleaner_cls.return_value = self.mock_cleaner_instance

        self.normalizer = TextNormalizer()

    def tearDown(self):
        self.logger_patcher.stop()
        self.tracker_patcher.stop()
        self.cleaner_patcher.stop()

    def test_normalize_text_preserves_code_block_indentation(self):
        """Full normalize_text pipeline preserves code block indentation."""
        text = (
            "  Some   text  \n"
            "```python\n"
            "def greet(name):\n"
            "    print(f'Hello, {name}')\n"
            "```\n"
            "  More   text  "
        )
        result = self.normalizer.normalize_text(text)
        self.assertIn("    print(f'Hello, {name}')", result)
        self.assertIn("Some text", result)
        self.assertIn("More text", result)

    def test_normalize_text_with_tilde_fence(self):
        """normalize_text preserves indentation inside ~~~ fences."""
        text = (
            "Intro\n"
            "~~~\n"
            "    indented\n"
            "~~~\n"
            "Outro"
        )
        result = self.normalizer.normalize_text(text)
        self.assertIn("    indented", result)

    def test_normalize_text_with_smart_quotes_and_code(self):
        """Smart quotes are replaced outside code blocks, code is preserved."""
        text = (
            "\u201cquoted\u201d text\n"
            "```\n"
            "    code    here\n"
            "```\n"
            "\u201cmore\u201d"
        )
        result = self.normalizer.normalize_text(text)
        self.assertIn('"quoted" text', result)
        self.assertIn("    code    here", result)
        self.assertIn('"more"', result)


if __name__ == "__main__":
    unittest.main()
