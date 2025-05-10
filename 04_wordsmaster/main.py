from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wordsmaster2",
              host="127.0.0.1",
              port=8080,
              timeout=30)

@mcp.tool()
def  get_word_count(text: str) -> int:
    """
    Count the number of words in the given text.

    Args:
        text (str): The input text to count words from

    Returns:
        int: The number of words in the text
    """
    word = text.split()
    return len(word)    


if __name__ == "__main__":
    print("Starting the mcp server in http://localhost:8080")
    mcp.run()