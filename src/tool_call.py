#!/usr/bin/env python3
# src/tool_call.py
"""
Tool calling implementation for CSV files using Gemini.
Allows querying CSV data through function calls without embedding.
"""

import os
import json
import pandas as pd
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Try to import google.genai for Gemini client
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from src import tokens


class CSVToolCaller:
    """Handles tool calling for CSV files using Gemini."""
    
    # Define available tools for CSV analysis
    TOOLS = [
        {
            "name": "analyze_csv",
            "description": "Load and analyze a CSV file. Use this first to understand the data structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file"},
                    "rows": {"type": "integer", "description": "Number of rows to preview (default: 10)"}
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "filter_csv",
            "description": "Filter CSV data based on column values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file"},
                    "column": {"type": "string", "description": "Column name to filter by"},
                    "value": {"type": "string", "description": "Value to filter for"},
                    "operator": {"type": "string", "description": "Comparison operator: ==, !=, >, <, >=, <=, contains"}
                },
                "required": ["file_path", "column", "value"]
            }
        },
        {
            "name": "aggregate_csv",
            "description": "Calculate aggregations (sum, avg, count, min, max) on a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file"},
                    "column": {"type": "string", "description": "Column to aggregate"},
                    "operation": {"type": "string", "description": "Operation: sum, avg, count, min, max"}
                },
                "required": ["file_path", "column", "operation"]
            }
        },
        {
            "name": "group_by_csv",
            "description": "Group data by a column and calculate aggregates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file"},
                    "group_column": {"type": "string", "description": "Column to group by"},
                    "agg_column": {"type": "string", "description": "Column to aggregate"},
                    "operation": {"type": "string", "description": "Aggregation: sum, avg, count, min, max"}
                },
                "required": ["file_path", "group_column", "agg_column", "operation"]
            }
        },
        {
            "name": "sort_csv",
            "description": "Sort CSV data by a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file"},
                    "column": {"type": "string", "description": "Column to sort by"},
                    "ascending": {"type": "boolean", "description": "Sort ascending (default: True)"}
                },
                "required": ["file_path", "column"]
            }
        },
        {
            "name": "get_column_info",
            "description": "Get information about columns in the CSV (types, null counts, unique values).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the CSV file"}
                },
                "required": ["file_path"]
            }
        }
    ]
    
    def __init__(self, csv_files: list[str]):
        """Initialize with CSV files to work with."""
        if not HAS_GEMINI:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")
        
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.csv_files = csv_files
        self.csv_data = {}  # Cache for loaded CSV data
        self.conversation_history = []  # Store as list of Content objects
        
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        if file_path not in self.csv_data:
            self.csv_data[file_path] = pd.read_csv(file_path)
        return self.csv_data[file_path]
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        try:
            df = self._load_csv(arguments["file_path"])
            
            if tool_name == "analyze_csv":
                rows = arguments.get("rows", 10)
                return f"CSV has {len(df)} rows and {len(df.columns)} columns.\nColumns: {list(df.columns)}\nFirst {rows} rows:\n{df.head(rows).to_string()}"
            
            elif tool_name == "filter_csv":
                column = arguments["column"]
                value = arguments["value"]
                operator = arguments.get("operator", "==")
                
                if column not in df.columns:
                    return f"Error: Column '{column}' not found. Available columns: {list(df.columns)}"
                
                try:
                    if df[column].dtype in ['int64', 'float64']:
                        value = float(value)
                except:
                    pass
                
                if operator == "==":
                    filtered = df[df[column] == value]
                elif operator == "!=":
                    filtered = df[df[column] != value]
                elif operator == ">":
                    filtered = df[df[column] > value]
                elif operator == "<":
                    filtered = df[df[column] < value]
                elif operator == ">=":
                    filtered = df[df[column] >= value]
                elif operator == "<=":
                    filtered = df[df[column] <= value]
                elif operator == "contains":
                    filtered = df[df[column].astype(str).str.contains(value, na=False)]
                else:
                    return f"Unknown operator: {operator}"
                
                return f"Found {len(filtered)} rows matching '{column} {operator} {value}':\n{filtered.head(20).to_string()}"
            
            elif tool_name == "aggregate_csv":
                column = arguments["column"]
                operation = arguments["operation"]
                
                if column not in df.columns:
                    return f"Error: Column '{column}' not found."
                
                if operation == "sum":
                    result = df[column].sum()
                elif operation == "avg":
                    result = df[column].mean()
                elif operation == "count":
                    result = df[column].count()
                elif operation == "min":
                    result = df[column].min()
                elif operation == "max":
                    result = df[column].max()
                else:
                    return f"Unknown operation: {operation}"
                
                return f"{operation.upper()} of '{column}': {result}"
            
            elif tool_name == "group_by_csv":
                group_col = arguments["group_column"]
                agg_col = arguments["agg_column"]
                operation = arguments["operation"]
                
                if group_col not in df.columns or agg_col not in df.columns:
                    return f"Error: One or more columns not found."
                
                if operation == "sum":
                    result = df.groupby(group_col)[agg_col].sum()
                elif operation == "avg":
                    result = df.groupby(group_col)[agg_col].mean()
                elif operation == "count":
                    result = df.groupby(group_col)[agg_col].count()
                elif operation == "min":
                    result = df.groupby(group_col)[agg_col].min()
                elif operation == "max":
                    result = df.groupby(group_col)[agg_col].max()
                else:
                    return f"Unknown operation: {operation}"
                
                return f"Group by '{group_col}' with {operation} of '{agg_col}':\n{result.to_string()}"
            
            elif tool_name == "sort_csv":
                column = arguments["column"]
                ascending = arguments.get("ascending", True)
                
                if column not in df.columns:
                    return f"Error: Column '{column}' not found."
                
                sorted_df = df.sort_values(by=column, ascending=ascending)
                return f"Sorted by '{column}' ({'ascending' if ascending else 'descending'}):\n{sorted_df.head(20).to_string()}"
            
            elif tool_name == "get_column_info":
                info = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "null_counts": df.isnull().sum().to_dict(),
                    "shape": df.shape
                }
                return f"CSV Info: {json.dumps(info, indent=2)}"
            
            return "Unknown tool"
            
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def _text_content(self, text: str, role: str = "user") -> types.Content:
        """Create a Content object from text."""
        return types.Content(
            role=role,
            parts=[types.Part(text=text)]
        )
    
    def _function_response_content(self, name: str, response: str) -> types.Content:
        """Create a Content object for a function response."""
        return types.Content(
            role="user",
            parts=[types.Part(
                function_response=types.FunctionResponse(
                    name=name,
                    response={"result": response}
                )
            )]
        )
    
    def _prepare_tools(self) -> list[types.Tool]:
        """Prepare tools in the expected format."""
        tools = []
        for tool_def in self.TOOLS:
            tools.append(
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool_def["name"],
                            description=tool_def["description"],
                            parameters=tool_def["parameters"]
                        )
                    ]
                )
            )
        return tools
    
    def ask(self, query: str) -> dict:
        """
        Ask a question about the CSV data using tool calling.
        Returns the answer and token usage information.
        """
        files_info = "\n".join([f"- {f}" for f in self.csv_files])
        systemPrompt = f"""You are a data analyst assistant. You have access to the following CSV files:
{files_info}

You have these tools available:
1. analyze_csv - Load and preview CSV data
2. filter_csv - Filter data by column values
3. aggregate_csv - Calculate sum, avg, count, min, max
4. group_by_csv - Group and aggregate data
5. sort_csv - Sort data by column
6. get_column_info - Get column metadata

Use the appropriate tools to answer the user's question. 
First analyze the CSV to understand its structure if you haven't already.
When showing data, use the tool to get the actual results.
Provide clear, concise answers based on the tool results."""

        # Build contents using proper Content objects
        contents = []
        
        # Add conversation history
        for item in self.conversation_history:
            contents.append(item)
        
        # Add current query
        contents.append(self._text_content(query, role="user"))
        
        # Prepare tools in the correct format
        tools = self._prepare_tools()
        
        # Create config
        config = types.GenerateContentConfig(
            system_instruction=self._text_content(systemPrompt, role="system"),
            tools=tools
        )
        
        # Call Gemini with tools
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=config
        )
        
        # Track token usage
        token_usage = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0
        }
        
        def update_tokens(resp):
            if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
                p = resp.usage_metadata.prompt_token_count or 0
                c = resp.usage_metadata.candidates_token_count or 0
                t = resp.usage_metadata.total_token_count or 0
                token_usage["prompt_token_count"] += p
                token_usage["candidates_token_count"] += c
                token_usage["total_token_count"] += t
                tokens.record_query(input_tokens=p, output_tokens=c)

        update_tokens(response)
        
        final_text = ""
        tool_results = []
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            if not response.candidates or not response.candidates[0].content.parts:
                break
                
            parts = response.candidates[0].content.parts
            has_function_call = any(part.function_call for part in parts if part.function_call)
            
            if not has_function_call:
                for part in parts:
                    if part.text:
                        final_text += part.text
                break
            
            # CRITICAL FIX: Append the raw, original model content block directly to history.
            # This preserves hidden internal fields like 'thought_signature' perfectly.
            contents.append(response.candidates[0].content)
            
            # Execute all function calls emitted in this step
            for part in parts:
                if part.function_call:
                    func_call = part.function_call
                    tool_name = func_call.name
                    args = {k: v for k, v in func_call.args.items()}
                    
                    tool_result = self._execute_tool(tool_name, args)
                    tool_results.append({
                        "tool": tool_name,
                        "arguments": args,
                        "result": tool_result
                    })
                    
                    # Add function result response to contents
                    contents.append(self._function_response_content(tool_name, tool_result))
            
            # Request subsequent loop execution
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=contents,
                config=config
            )
            update_tokens(response)
        
        # Sync structural chat memory cleanly
        self.conversation_history.append(self._text_content(query, role="user"))
        if final_text:
            self.conversation_history.append(self._text_content(final_text, role="model"))
        
        return {
            "answer": final_text or "No response generated",
            "tool_results": tool_results,
            "token_usage": token_usage,
            "session_token_usage": tokens.get_status()["total_tokens_used"],
            "provider": "gemini"
        }


def ask_with_tools(query: str, csv_files: list[str]) -> dict:
    """Convenience function to ask a question using tool calling."""
    tool_caller = CSVToolCaller(csv_files)
    return tool_caller.ask(query)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.tool_call <csv_file> [query]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Describe this CSV file"
    
    print(f"File: {csv_file}")
    print(f"Query: {query}\n")
    
    result = ask_with_tools(query, [csv_file])
    print(f"Answer: {result['answer']}\n")