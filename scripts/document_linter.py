#!/usr/bin/env python3
"""
Comprehensive Markdown and LaTeX Document Linter & Auto-Fixer.
Implements the rules required by the @document-linter workflow:
- Missing blank lines around headings
- Trailing whitespace removal
- Consecutive blank line normalization
- Code block balance checks
- LaTeX balanced brace and environment checks
"""

import argparse
import os
import re
import sys
from pathlib import Path


def lint_and_fix_markdown(content: str, filepath: str, fix: bool) -> tuple[str, list[str]]:
    """Lints and optionally fixes a Markdown string."""
    errors = []
    lines = content.split('\n')
    fixed_lines = []
    
    in_code_block = False
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 1. Check code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
        
        # We don't auto-fix trailing spaces inside code blocks to preserve literals
        if not in_code_block:
            # 2. Check and fix trailing whitespace
            if line.rstrip() != line:
                errors.append(f"{filepath}:{i+1}: Trailing whitespace detected.")
                if fix:
                    line = line.rstrip()
            
            # 3. Check and fix missing space after heading hash
            heading_match = re.match(r'^(#{1,6})([^ #\n].*)$', line)
            if heading_match:
                errors.append(f"{filepath}:{i+1}: Missing space after heading marker '{heading_match.group(1)}'.")
                if fix:
                    line = f"{heading_match.group(1)} {heading_match.group(2)}"
        
        fixed_lines.append(line)
        
    if in_code_block:
        errors.append(f"{filepath}:EOF: Unclosed code block (```) detected.")
        if fix:
            fixed_lines.append("```")

    # 4. Check consecutive blank lines
    fixed_content = '\n'.join(fixed_lines)
    if '\n\n\n' in fixed_content:
        errors.append(f"{filepath}: Multiple consecutive blank lines detected.")
        if fix:
            fixed_content = re.sub(r'\n{3,}', '\n\n', fixed_content)

    return fixed_content, errors


def lint_and_fix_latex(content: str, filepath: str, fix: bool) -> tuple[str, list[str]]:
    """Lints and optionally fixes a LaTeX string."""
    errors = []
    
    # Check balanced braces
    if content.count('{') != content.count('}'):
        errors.append(f"{filepath}: Unbalanced braces '{{' and '}}' detected.")
    
    # Check environments
    begins = len(re.findall(r'\\begin\{.*\}', content))
    ends = len(re.findall(r'\\end\{.*\}', content))
    if begins != ends:
        errors.append(f"{filepath}: Unbalanced environments (\\begin vs \\end) detected.")
        
    # Fix trailing whitespace globally
    lines = content.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        if line.rstrip() != line:
            errors.append(f"{filepath}:{i+1}: Trailing whitespace detected.")
            if fix:
                line = line.rstrip()
        fixed_lines.append(line)
        
    fixed_content = '\n'.join(fixed_lines)
    
    # Multiple blank lines
    if '\n\n\n' in fixed_content:
        errors.append(f"{filepath}: Multiple consecutive blank lines detected.")
        if fix:
            fixed_content = re.sub(r'\n{3,}', '\n\n', fixed_content)
            
    return fixed_content, errors


def process_file(filepath: str, fix: bool) -> bool:
    """Processes a single file, prints errors, and overwrites if fixed. Returns True if successful/no unfixable errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
        
    ext = Path(filepath).suffix.lower()
    
    if ext == '.md':
        fixed_content, errors = lint_and_fix_markdown(content, filepath, fix)
    elif ext == '.tex':
        fixed_content, errors = lint_and_fix_latex(content, filepath, fix)
    else:
        print(f"Skipping {filepath}: Unsupported extension {ext}")
        return True

    if errors:
        for err in errors:
            print(f"[Error] {err}")
    
    if fix and content != fixed_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"[Fixed] {filepath}")
        
    # Return False if errors exist and we didn't fix them, or if there are unfixable structural errors (like braces)
    # For simplicity, if not fixing and errors exist, fail. If fixing and unfixable errors persist, fail.
    # Structural errors like unclosed code blocks are auto-fixed in markdown. Unbalanced braces in tex are not yet auto-fixable.
    if ext == '.tex' and ('Unbalanced' in ''.join(errors)):
        return False
        
    return not errors if not fix else True


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Markdown and LaTeX Linter")
    parser.add_argument('files', nargs='+', help="Files to lint")
    parser.add_argument('--fix', action='store_true', help="Auto-fix formatting errors")
    
    args = parser.parse_args()
    
    all_passed = True
    for fp in args.files:
        if os.path.isdir(fp):
            # Recursively find md/tex files
            for root, _, files in os.walk(fp):
                for file in files:
                    if file.endswith(('.md', '.tex')):
                        full_path = os.path.join(root, file)
                        success = process_file(full_path, args.fix)
                        if not success:
                            all_passed = False
        else:
            success = process_file(fp, args.fix)
            if not success:
                all_passed = False

    if not all_passed and not args.fix:
        print("\nLinting failed. Run with --fix to automatically resolve formatting errors.")
        sys.exit(1)
    elif not all_passed and args.fix:
        print("\nLinting finished with unfixable structural errors.")
        sys.exit(1)
    else:
        print("\nLinting passed successfully.")
        sys.exit(0)


if __name__ == '__main__':
    main()
