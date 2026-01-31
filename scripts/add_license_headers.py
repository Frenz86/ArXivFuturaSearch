#!/usr/bin/env python3
"""Script to add Apache 2.0 license header to all Python files."""

import os
from pathlib import Path

# Apache 2.0 license header
LICENSE_HEADER = '''# Copyright 2025 ArXivFuturaSearch Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''

# Marker to check if header already exists
HEADER_MARKER = "# Copyright 2025 ArXivFuturaSearch Contributors"


def add_license_header(file_path: Path) -> bool:
    """
    Add license header to a Python file if it doesn't already have one.

    Args:
        file_path: Path to the Python file

    Returns:
        True if header was added, False if already present
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if license header already exists
        lines = content.split('\n')
        if len(lines) >= 10 and HEADER_MARKER in content:
            print(f"[OK] {file_path}: Header already exists")
            return False

        # Find the first non-comment, non-empty line after the shebang
        insert_pos = 0
        lines = content.split('\n')

        # Skip shebang if present
        if lines and lines[0].startswith('#!'):
            insert_pos = 1

        # Skip empty lines and existing docstring
        while insert_pos < len(lines):
            line = lines[insert_pos].strip()
            if line and not line.startswith('"""') and not line.startswith("'''"):
                break
            insert_pos += 1

        # Prepare new content with header
        new_lines = []
        new_lines.extend(lines[:insert_pos])
        new_lines.append("")  # Empty line after shebang
        new_lines.append(LICENSE_HEADER.strip())
        new_lines.append("")
        new_lines.extend(lines[insert_pos:])

        new_content = '\n'.join(new_lines)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"[OK] {file_path}: Header added")
        return True

    except Exception as e:
        print(f"[ERROR] {file_path}: Error - {e}")
        return False


def main():
    """Add license headers to all Python files in the app directory."""
    app_dir = Path("app")

    if not app_dir.exists():
        print("Error: app directory not found")
        return

    # Find all Python files
    py_files = list(app_dir.rglob("*.py"))

    print(f"Found {len(py_files)} Python files")
    print("Adding Apache 2.0 license headers...\n")

    added_count = 0
    for file_path in sorted(py_files):
        if add_license_header(file_path):
            added_count += 1

    print(f"\n✓ Completed! Added headers to {added_count} files")


if __name__ == "__main__":
    main()
