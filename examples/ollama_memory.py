from subprocess import run

result = run(["amem", "search", "Test"], capture_output=True, text=True, check=True)
output = result.stdout

run(["ollama", "run", "llama3", f"Summarize: {output}"], check=True)
