from typing import Any, List, Optional
import re

class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        keep_separator: bool = True,
        **kwargs: Any
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._keep_separator = keep_separator

    def _length_function(self, text: str) -> int:
        # Define your own length function here
        return len(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge the splits into chunks based on the chunk size and overlap.
        
        Parameters:
            splits (List[str]): List of strings to merge into chunks.
            separator (str): The separator used to join the splits.
        
        Returns:
            List[str]: List of merged chunks.
        """
        separator_len = self._length_function(separator)
        docs = []
        current_doc = []
        total = 0

        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    print(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = separator.join(current_doc)
                    docs.append(doc)
                    while (
                        total > self._chunk_overlap
                        or (total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size and total > 0)
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        
        doc = separator.join(current_doc)
        docs.append(doc)
        return docs

    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError("split_text method must be implemented in subclasses")

class CharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        separator: str = "\n\n",
        is_separator_regex: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )
        splits = self._split_text_with_regex(text, separator)
        separator = "" if self._keep_separator else self._separator
        return self._merge_splits(splits, separator)

    def _split_text_with_regex(
        self, text: str, separator: str
    ) -> List[str]:
        if separator:
            if self._keep_separator:
                _splits = re.split(f"({separator})", text)
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if len(_splits) % 2 == 1:
                    splits.append(_splits[-1])
                splits = [_splits[0]] + splits
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)
    
    def _split_text_with_regex(
    self, text: str, separator: str, keep_separator: bool
    ) -> List[str]:
        # Now that we have the separator, split the text
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = [_splits[0]] + splits
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks


def example():
    # Create an instance of RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=10
    )

    # Example text to be split
    text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """

    # Split the text
    chunks = splitter.split_text(text)

    # Print the resulting chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(chunk)
        print("-" * 30)
