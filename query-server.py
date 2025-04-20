import requests
import json
import time
import asyncio
import concurrent.futures
from tabulate import tabulate

async def query_server(url, query, method_name):
    """Query a RAG server and return the response with timing information"""
    headers = {"Content-Type": "application/json"}
    data = {"query": query}
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                "method": method_name,
                "status": "Success",
                "response": result["response"],
                "time": elapsed,
                "time_reported": result.get("response_time", "N/A"),
                "doc_count": len(result.get("relevant_documents", [])),
                "full_response": result
            }
        else:
            return {
                "method": method_name,
                "status": f"Error: {response.status_code}",
                "response": None,
                "time": elapsed,
                "time_reported": "N/A",
                "doc_count": 0,
                "full_response": None
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "method": method_name,
            "status": f"Exception: {str(e)}",
            "response": None,
            "time": elapsed,
            "time_reported": "N/A",
            "doc_count": 0,
            "full_response": None
        }

async def query_all_servers(query):
    """Query all three RAG servers and return their responses"""
    servers = [
        {"url": "http://0.0.0.0:8000/query", "name": "Standard RAG"},
        {"url": "http://0.0.0.0:8001/query", "name": "RAG Fusion"},
        {"url": "http://0.0.0.0:8002/query", "name": "TF-IDF RAG"}
    ]
    
    tasks = [query_server(server["url"], query, server["name"]) for server in servers]
    
    results = await asyncio.gather(*tasks)
    return results

def display_results(results):
    """Display the results in a formatted way"""
    table_data = []
    for result in results:
        table_data.append([
            result["method"],
            result["status"],
            f"{result['time']:.2f}s",
            result["time_reported"],
            result["doc_count"]
        ])
    
    print("\n" + "="*80)
    print("SUMMARY OF RAG APPROACHES")
    print("="*80)
    print(tabulate(
        table_data,
        headers=["Method", "Status", "Total Time", "LLM Time", "Doc Count"],
        tablefmt="grid"
    ))
    
    for result in results:
        if result["response"]:
            print("\n" + "="*80)
            print(f"RESPONSE FROM: {result['method']}")
            print("="*80)
            print(result["response"])
            
    with open("rag_comparison_results.json", "w") as f:
        json_results = {r["method"]: r["full_response"] for r in results if r["full_response"]}
        json.dump(json_results, f, indent=2)
    print("\nFull results saved to rag_comparison_results.json")

def main():
    query = input("Enter your query: ")
    print(f"\nQuerying all RAG servers with: '{query}'")
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(query_all_servers(query))
    display_results(results)

if __name__ == "__main__":
    main()