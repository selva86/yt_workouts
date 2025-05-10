# Below are useful info to host this Agentic RAG app using MCP

_Note: Please refer to the video ('Building Agentic RAG using MCP') on my [youtube channel](youtube.com/@machinelearningplus) for a complete walkthrough._ 

## Step 1: Start the Qdrant container

Start the QDrant container
```
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

## Step 2: Set up Bright data account.

Open a free account in [brightdata](https://brightdata.com/) and setup a user-email and password. You will need this inside the `server2.py`.

## Step 3: Start the MCP server.

Clone the repo and open it in cursor IDE. Then go to settings > Cursor settings > MCP Servers. Click on 'Add new MCP server' and add the following code (assuming you have no other server running) to mcp.json.

__To know the location of 'uv'__

- For Mac / Linux: Use `which uv` or `where uv`
- For windows: It is usually present in `%USERPROFILE%/.local/bin/uv`, where `%USERPROFILE%` resolves to something like `c:\Users\username`.

```json
{
  "mcpServers": {
    "mcpRAG": {
      "command": "path/to/uv",
      "args": [
        "--directory",
        "absolute/path/to/projectdir",
        "run",
        "server2.py"
      ]
    }
  }
}
```

It should show the status in green and display the tools: `f1_faq_search_tool` and `bright_data_web_search_tool`.

You can now open the chat in cursor (Ctrl + L) and ask questions.