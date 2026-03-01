"""
Built-in tools for common operations.

This module provides pre-built tools for common tasks like
calculations, file operations, and system utilities.
"""

from typing import Any, Optional
import math
import os
import subprocess
import sys
from datetime import datetime
import json

from torchagentic.tools.base import Tool, ToolParameter


class CalculatorTool(Tool):
    """
    Tool for performing mathematical calculations.
    
    Supports basic arithmetic and common mathematical functions.
    """
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Supports basic arithmetic (+, -, *, /), "
                       "powers (^ or **), square root (sqrt), logarithms (log, log10), "
                       "trigonometric functions (sin, cos, tan), and more.",
            tags=["math", "calculation"],
        )
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="The mathematical expression to evaluate",
                required=True,
            ),
        ]
    
    async def execute(self, expression: str) -> str:
        """Execute a mathematical expression."""
        try:
            # Safe evaluation of mathematical expressions
            # Replace common math notation
            expr = expression.lower()
            expr = expr.replace("^", "**")
            expr = expr.replace("√", "sqrt")
            
            # Create safe namespace with math functions
            safe_dict = {
                name: getattr(math, name)
                for name in dir(math)
                if not name.startswith("_")
            }
            safe_dict.update({
                "abs": abs,
                "round": round,
                "sum": sum,
                "min": min,
                "max": max,
            })
            
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"


class DateTimeTool(Tool):
    """
    Tool for getting current date and time information.
    """
    
    def __init__(self):
        super().__init__(
            name="get_datetime",
            description="Get the current date and time. Can return full datetime or specific components.",
            tags=["time", "date", "utility"],
        )
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="format",
                type="string",
                description="Optional format string (e.g., '%Y-%m-%d %H:%M:%S'). "
                           "If not provided, returns ISO format.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="component",
                type="string",
                description="Specific component to return: 'date', 'time', 'year', 'month', "
                           "'day', 'hour', 'minute', 'second', 'weekday', or 'full'",
                required=False,
                default="full",
                enum=["date", "time", "year", "month", "day", "hour", "minute", "second", 
                      "weekday", "full"],
            ),
        ]
    
    async def execute(
        self,
        format: Optional[str] = None,
        component: str = "full",
    ) -> str:
        """Get current date/time information."""
        now = datetime.now()
        
        if format:
            return now.strftime(format)
        
        if component == "date":
            return now.strftime("%Y-%m-%d")
        elif component == "time":
            return now.strftime("%H:%M:%S")
        elif component == "year":
            return str(now.year)
        elif component == "month":
            return str(now.month)
        elif component == "day":
            return str(now.day)
        elif component == "hour":
            return str(now.hour)
        elif component == "minute":
            return str(now.minute)
        elif component == "second":
            return str(now.second)
        elif component == "weekday":
            return now.strftime("%A")
        else:
            return now.isoformat()


class FileReadTool(Tool):
    """
    Tool for reading files from the filesystem.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(
            name="read_file",
            description="Read the contents of a file. Supports text files and can read "
                       "specific line ranges.",
            tags=["file", "io"],
        )
        self.base_dir = base_dir or os.getcwd()
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to read (relative or absolute)",
                required=True,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="Starting line number (0-indexed). If not provided, reads from beginning.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="Ending line number (exclusive). If not provided, reads to end.",
                required=False,
                default=None,
            ),
        ]
    
    async def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read a file's contents."""
        try:
            # Resolve path
            if not os.path.isabs(path):
                full_path = os.path.join(self.base_dir, path)
            else:
                full_path = path
            
            # Security check - ensure path is within base_dir
            real_path = os.path.realpath(full_path)
            real_base = os.path.realpath(self.base_dir)
            if not real_path.startswith(real_base):
                return f"Error: Access denied. Path must be within {self.base_dir}"
            
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Apply line range if specified
            if start_line is not None or end_line is not None:
                lines = lines[start_line:end_line]
            
            return "".join(lines)
        
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"


class FileWriteTool(Tool):
    """
    Tool for writing files to the filesystem.
    """
    
    def __init__(self, base_dir: Optional[str] = None, allow_overwrite: bool = True):
        super().__init__(
            name="write_file",
            description="Write content to a file. Can create new files or overwrite existing ones.",
            tags=["file", "io"],
        )
        self.base_dir = base_dir or os.getcwd()
        self.allow_overwrite = allow_overwrite
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to write (relative or absolute)",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write to the file",
                required=True,
            ),
            ToolParameter(
                name="mode",
                type="string",
                description="Write mode: 'w' for overwrite, 'a' for append",
                required=False,
                default="w",
                enum=["w", "a"],
            ),
        ]
    
    async def execute(
        self,
        path: str,
        content: str,
        mode: str = "w",
    ) -> str:
        """Write content to a file."""
        try:
            # Resolve path
            if not os.path.isabs(path):
                full_path = os.path.join(self.base_dir, path)
            else:
                full_path = path
            
            # Security check
            real_path = os.path.realpath(full_path)
            real_base = os.path.realpath(self.base_dir)
            if not real_path.startswith(real_base):
                return f"Error: Access denied. Path must be within {self.base_dir}"
            
            # Check if file exists and overwrite not allowed
            if os.path.exists(full_path) and not self.allow_overwrite and mode == "w":
                return "Error: File exists and overwrite is not allowed"
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {path}"
        
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class SearchTool(Tool):
    """
    Tool for searching through text or data.
    """
    
    def __init__(self):
        super().__init__(
            name="search",
            description="Search for patterns or content within text. Supports regex patterns.",
            tags=["search", "text"],
        )
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="string",
                description="The text to search within",
                required=True,
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="The pattern to search for (supports regex)",
                required=True,
            ),
            ToolParameter(
                name="case_sensitive",
                type="boolean",
                description="Whether the search should be case-sensitive",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="return_matches",
                type="boolean",
                description="Whether to return all matches or just count",
                required=False,
                default=True,
            ),
        ]
    
    async def execute(
        self,
        text: str,
        pattern: str,
        case_sensitive: bool = False,
        return_matches: bool = True,
    ) -> str:
        """Search for a pattern in text."""
        import re
        
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            matches = re.findall(pattern, text, flags)
            
            if return_matches:
                if matches:
                    return f"Found {len(matches)} match(es): {json.dumps(matches)}"
                return "No matches found"
            else:
                return f"Found {len(matches)} match(es)"
        
        except re.error as e:
            return f"Invalid regex pattern: {e}"


class PythonREPLTool(Tool):
    """
    Tool for executing Python code in a REPL-like environment.
    
    WARNING: This tool executes arbitrary Python code. Use with caution.
    """
    
    def __init__(self, safe_mode: bool = True):
        super().__init__(
            name="python_repl",
            description="Execute Python code and return the output. "
                       "Useful for calculations, data processing, and scripting. "
                       "WARNING: Executes arbitrary code - use with caution.",
            tags=["code", "python", "execution"],
        )
        self.safe_mode = safe_mode
        self._globals = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "json": __import__("json"),
            "re": __import__("re"),
            "datetime": __import__("datetime"),
        }
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type="string",
                description="Python code to execute",
                required=True,
            ),
        ]
    
    async def execute(self, code: str) -> str:
        """Execute Python code."""
        try:
            # Capture stdout
            import io
            from contextlib import redirect_stdout
            
            output = io.StringIO()
            
            with redirect_stdout(output):
                exec(code, self._globals)
            
            return output.getvalue() or "Code executed successfully (no output)"
        
        except Exception as e:
            return f"Error executing code: {e}"


class ShellCommandTool(Tool):
    """
    Tool for executing shell commands.
    
    WARNING: This tool executes arbitrary shell commands. Use with extreme caution.
    """
    
    def __init__(self, allowed_commands: Optional[list[str]] = None):
        super().__init__(
            name="shell_command",
            description="Execute a shell command and return the output. "
                       "WARNING: Executes arbitrary shell commands - use with extreme caution.",
            tags=["shell", "command", "execution"],
        )
        self.allowed_commands = allowed_commands
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="Shell command to execute",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout in seconds (default: 30)",
                required=False,
                default=30,
            ),
        ]
    
    async def execute(self, command: str, timeout: int = 30) -> str:
        """Execute a shell command."""
        # Check allowed commands
        if self.allowed_commands:
            cmd_base = command.split()[0] if command.split() else ""
            if cmd_base not in self.allowed_commands:
                return f"Error: Command '{cmd_base}' is not in the allowed list"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nstderr: {result.stderr}"
            
            return output or f"Command executed (exit code: {result.returncode})"
        
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {e}"
